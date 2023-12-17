import torch
from ppo import PPO, Memory
from environments import MetaGames
from ga import Auxiliary
import os
import argparse
import json
import wandb
from utils.setup_wandb import setup_wandb
from eval_mfos_ppo import eval_ppo
from statistics import mean

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, required=True)
parser.add_argument("--opponent", type=str, required=True)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--mamaml-id", type=int, default=0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--append_input", type=bool, default=False)
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--opp_lr", type=float, default=1)
parser.add_argument("--rand_opp", type=bool, default=False) # Randomly sample opponent learning rates
parser.add_argument("--collect_data", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=4096)
args = parser.parse_args()

setup_wandb(vars(args))


def inv_sigmoid(x):
    return -torch.log((1 / (x + 1e-8)) - 1)


if __name__ == "__main__":
    ############################################
    K_epochs = 4  # update policy for K epochs

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = args.lr # 0.002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    max_episodes = 1024 #2048
    batch_size = args.batch_size #4096
    random_seed = args.seed
    num_steps = 100

    save_freq = 512
    name = args.exp_name

    aux = Auxiliary().to(device)

    print(f"RUNNING NAME: {'runs/' + name}")
    if not os.path.isdir('runs/' + name):
        os.mkdir('runs/' + name)
        with open(os.path.join('runs/' + name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    #if not os.path.isdir('runs/' + name + '/policy'):
     #   os.mkdir('runs/' + name + '/policy')
      #  with open(os.path.join('runs/' + name + '/policy', "commandline_args.txt"), "w") as f:
       #     json.dump(args.__dict__, f, indent=2)

    if args.collect_data:
        if not os.path.isdir('runs/' + name + '/state_action'):
            os.mkdir('runs/' + name + '/state_action')
            with open(os.path.join('runs/' + name + '/state_action', "commandline_args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=2)

    #############################################
    # creating environment
    if args.seed != None:
        torch.manual_seed(random_seed) # Set seed for reproducability.

    env = MetaGames(batch_size, opponent=args.opponent, game=args.game, mmapg_id=args.mamaml_id, 
                    opp_lr=args.opp_lr, rand_opp=args.rand_opp)

    action_dim = env.d
    state_dim = (env.d * 2)
    if args.append_input:
        state_dim = (env.d * 2) + 2 # Changed way state is input

    memory = Memory()
    ppo = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, args.entropy)

    if args.checkpoint:
        ppo.load(args.checkpoint)

    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    rew_means = []
    state_action_data = [] # Store state-action pairs for behavioural clone

    for i_episode in range(1, max_episodes + 1):
        try:
            state, payout = env.reset()
            payout_probs = torch.cat([aux(payout[i].to(device)) for i in range(batch_size)])
        except ValueError or IndexError:
            state = env.reset()

        running_reward = torch.zeros(batch_size).cuda()
        running_opp_reward = torch.zeros(batch_size).cuda()

        last_reward = 0
        policy = []
        store_state = None
        store_action = None

        for t in range(num_steps):
            if args.append_input:
                state = torch.cat([state, payout_probs], axis=-1) # payout], axis=-1)
            # Running policy_old:

            action = ppo.policy_old.act(state, memory)

            state, reward, info, M = env.step(action) # This state has size (batch, 10), but we append the payout at each step (if chosen) before putting into PPO

            memory.rewards.append(reward)
            running_reward += reward.squeeze(-1)
            running_opp_reward += info.squeeze(-1)
            last_reward = reward.squeeze(-1)

            if t==num_steps-1: # Record policy for the final episode
                store_state = state.cpu().numpy().tolist()
                store_action = action.cpu().numpy().tolist()
             #   policy.append(state.cpu().numpy().tolist()) # Taken from Chris Lu notebooks paper plots-Copy2.ipynb

        if args.collect_data:
            # Save the state-action pair for the batch at the end of every meta-episode
            for i in range(batch_size):
                if args.append_input == True:
                    state_action_data.append(
                        {
                            'state_agent' + str(i) : torch.cat((state[i][:5],state[i][-2:]), -1).tolist(), # Appending auxilliary input
                            'action_agent' + str(i) : action[i].tolist()
                        }
                    )
                else:
                    state_action_data.append(
                        {
                            'state_agent' + str(i) : state[i][:5].tolist(),
                            'action_agent' + str(i) : action[i].tolist()
                        }
                    )

        ppo.update(memory)
        memory.clear_memory()

        print("=" * 100, flush=True)

        print(f"episode: {i_episode}", flush=True)

        print(f"loss: {-running_reward.mean() / num_steps}", flush=True)

        rew_means.append(
            {
                "rew": (running_reward.mean() / num_steps).item(),
                "opp_rew": (running_opp_reward.mean() / num_steps).item(),
            }
        )

        print(f"opponent loss: {-running_opp_reward.mean() / num_steps}", flush=True)

        wandb.log({"train_loss": -running_reward.mean() / num_steps, 
                   "train_opp_loss": -running_opp_reward.mean() / num_steps})
        
        for i, value in enumerate([*map(mean, zip(*store_state))][:5]):   
            wandb.log({"train_mean_state_"+str(i): value})
        for i, value in enumerate([*map(mean, zip(*store_action))][:5]):   
            wandb.log({"train_mean_action_"+str(i): value})


        if i_episode % save_freq == 0:
            ppo.save(os.path.join('runs/' + name, f"{i_episode}.pth"))
            with open(os.path.join('runs/' + name, f"out_{i_episode}.json"), "w") as f:
                json.dump(rew_means, f)
            #with open(os.path.join('runs/' + name + '/policy', f"out_{i_episode}.json"), "w") as f:
             #   json.dump(policy, f)
            if args.collect_data:
                with open(os.path.join('runs/' + name + '/state_action', f"out_{i_episode}.json"), "w") as f:
                    json.dump(state_action_data, f)
            print(f"SAVING! {i_episode}")

    ppo.save(os.path.join('runs/' + name, f"{i_episode}.pth"))
    with open(os.path.join('runs/' + name, f"out_{i_episode}.json"), "w") as f:
        json.dump(rew_means, f)
    #with open(os.path.join('runs/' + name + '/policy', f"out_{i_episode}.json"), "w") as f:
     #   json.dump(policy, f)
    if args.collect_data:
        with open(os.path.join('runs/' + name + '/state_action', f"out_{i_episode}.json"), "w") as f:
           json.dump(state_action_data, f)
    print(f"SAVING! {i_episode}")


    for eval_game in ["IPD", "random", "randIPD", "noisy_IPD"]:
        for lr in [3, 2.5, 2, 1.5, 1, 0.5, 0.05]:
            eval_ppo(args, game=eval_game, opp_lr=lr, checkpoint=max_episodes)