import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MemoryRecurrent:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

        self.value_preds = []
        self.latent_a = []
        self.latent_a_s = []
        self.latent_v = []
        self.latent_v_s = []

    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

        del self.value_preds[:]
        del self.latent_a[:]
        del self.latent_a_s[:]
        del self.latent_v[:]
        del self.latent_v_s[:]

class RecurrentNet(nn.Module):
    def __init__(self, state_dim, output_dim, batch_size, hidden_size=256):
        super(RecurrentNet, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm_size = 32
        # self.gru = nn.GRU(input_size=state_dim, hidden_size=self.lstm_size)
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=self.lstm_size)
        self.output = nn.Sequential(
            nn.Linear(self.lstm_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, output_dim))

    def reset_hidden(self):
        self.hidden_cell = (torch.zeros(1, self.batch_size, self.lstm_size, device=device),
                            torch.zeros(1, self.batch_size, self.lstm_size, device=device))
        # self.hidden_cell = torch.zeros(1, self.batch_size, self.lstm_size, device=device)
    
    def forward(self, x):
        x = x.unsqueeze(0)
        _, self.hidden_cell = self.lstm(x, self.hidden_cell)
        out = self.output(self.hidden_cell[0][-1])
        return out

    def eval(self, x):
        # x = x.unsqueeze(0)
        out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        out = self.output(out)
        return out

class ActorCriticRecurrent(nn.Module):
    def __init__(self, state_dim, action_dim, batch_size, hidden_size):
        super(ActorCriticRecurrent, self).__init__()
        # action mean range -1 to 1
        self.actor =  RecurrentNet(state_dim, action_dim*2, batch_size, hidden_size=hidden_size)
        
        # critic
        self.critic = RecurrentNet(state_dim, 1, batch_size, hidden_size=hidden_size)

        self.std = nn.Parameter(torch.ones((1,), device=device))

        self.actor.reset_hidden()
        self.critic.reset_hidden()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size

    def forward(self):
        raise NotImplementedError
    

    def reset_hidden(self):
        self.actor.reset_hidden()
        self.critic.reset_hidden()

    def act(self, state, memory, store_latent=False):
        # if store_latent:
        #     memory.latent_a.append(self.actor.hidden_cell.detach())
        #     memory.latent_v.append(self.critic.hidden_cell.detach())
        action_mean, action_var = torch.split(self.actor(state), self.action_dim, dim=-1)
        cov_mat = torch.diag_embed(F.softplus(action_var))
        # cov_mat = torch.eye(5, device=device).unsqueeze(0).repeat(self.batch_size, 1, 1) * torch.exp(torch.log(self.std))
        # print(cov_mat.size(), 'cov mat size')
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()
    
    def sample(self, state, mean=False):
        with torch.no_grad():
            action_mean, action_var = torch.split(self.actor(state), self.action_dim, dim=-1)
            cov_mat = torch.diag_embed(F.softplus(action_var))
            # cov_mat = torch.eye(5, device=device).unsqueeze(0).repeat(self.batch_size, 1, 1) * torch.exp(torch.log(self.std))

            dist = MultivariateNormal(action_mean, cov_mat)
            if mean:
                action = action_mean
            else:
                action = dist.sample()
        
        return action.detach()


    def evaluate(self, state, action):
        action_mean, action_var = torch.split(self.actor.eval(state), self.action_dim, dim=-1)
        action_var = F.softplus(action_var)
            
#         action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        # cov_mat = torch.eye(5, device=device).unsqueeze(0).repeat(self.batch_size, 1, 1) * torch.exp(torch.log(self.std))

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic.eval(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPORecurrent:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, entropy,
                 batch_size, hidden_size, traj_length):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.traj_length = traj_length

        self.policy = ActorCriticRecurrent(state_dim, action_dim, batch_size, hidden_size).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCriticRecurrent(state_dim, action_dim, batch_size, hidden_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.entropy_bonus = entropy

    # def select_action(self, state, memory):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    #     return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def eval(self):
        self.policy.eval()

    def update(self, memory):
        #wandb.init() # NOT FOR LABEEBAH
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
#         rewards = torch.stack(rewards, dtype=torch.float32).to(device)
        rewards = torch.stack(rewards).squeeze(-1)#.detach()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

        
        # Optimize policy for K epochs:
        for k in range(self.K_epochs):
            # Evaluating old actions and values :
            print(f'{k} hg: ', 'MEM STATS 7: ', torch.cuda.memory_cached())
            self.policy.reset_hidden()
            # logprobs_t, state_values_t, dist_entropy_t = [], [] , []
            # for t in range(len(old_states)):
            # # Evaluating old actions and values :
            #     if t%self.traj_length==0:
            #         self.policy.actor.hidden_cell = memory.latent_a[t // self.traj_length]
            #         self.policy.critic.hidden_cell = memory.latent_v[t // self.traj_length]
            #     logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[t], old_actions[t])
            #     logprobs_t.append(logprobs)
            #     state_values_t.append(state_values)
            #     dist_entropy_t.append(dist_entropy)
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # logprobs = torch.stack(logprobs_t)
            # state_values = torch.stack(state_values_t)
            # dist_entropy = torch.stack(dist_entropy_t)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach()).squeeze()

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - self.entropy_bonus * dist_entropy
            entropy_loss = (self.entropy_bonus * dist_entropy).mean()
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            ''' NOT FOR LABEEBAH
        wandb.log({
            'total_loss': loss.mean(), 
            'advantages': advantages.mean(), 
            'surr1': surr1.mean(), 
            'surr2': surr2.mean(),
            'std': self.policy.std.item(),
            'entropy_loss': entropy_loss})
        '''
        # print('STD: ', self.policy.std)
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, filename):
        torch.save({
            'actor_critic': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['actor_critic'])
        self.policy_old.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        