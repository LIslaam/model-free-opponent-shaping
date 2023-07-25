# Test a pretrained PPO agent
import torch
from ppo import PPO, Memory
from environments import MetaGames
import os
import argparse
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, required=True)
parser.add_argument("--opponent", type=str, required=True)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--exp-name", type=str, default="")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--mamaml-id", type=int, default=0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--append_input", type=bool, default=False)
args = parser.parse_args()

if __name__ == "__main__":
            ##############################
    K_epochs = 4  # update policy for K epochs

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    max_episodes = 256
    batch_size = 4096
    random_seed = args.seed
    num_steps = 100
    num_tests = 10 # How many times trained policy is tested on game

    save_freq = 250
    name = args.exp_name

    print(f"RUNNING NAME: {'runs/' + name + '/test_' + args.game  + '_seed' + str(args.seed)}")
    if not os.path.isdir('runs/' + name + '/test_' + args.game  + '_seed' + str(args.seed)):
        os.mkdir('runs/' + name + '/test_' + args.game  + '_seed' + str(args.seed))
        with open(os.path.join('runs/' + name + '/test_' + args.game  + '_seed' + str(args.seed), 
                               "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    if not os.path.isdir('runs/' + name + '/test_' + args.game  + '_seed' + str(args.seed) + '_policy'):
        os.mkdir('runs/' + name + '/test_' + args.game  + '_seed' + str(args.seed) + '_policy')
        with open(os.path.join('runs/' + name + '/test_' + args.game  + '_seed' + str(args.seed) + '_policy', 
                               "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    #################################################
    if args.seed != None:
        torch.manual_seed(random_seed)
    env = MetaGames(batch_size, opponent=args.opponent, game=args.game, mmapg_id=args.mamaml_id)
    memory = Memory()
    
    action_dim = env.d
    state_dim = env.d * 2
    if args.append_input:
        state_dim = (env.d * 2) + 8

    ppo = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, args.entropy)

    directory = "runs/" + name + '/'
    checkpoint_path = directory + "{}.pth".format(1024)# max_episodes)
    print("loading network from : " + checkpoint_path)

    ppo.load(checkpoint_path)
    rew_means = []

    for i_episode in range(1, max_episodes + 1):
        try:
            state, payout = env.reset()
        except ValueError:
            state = env.reset()

        running_reward = torch.zeros(batch_size).cuda()
        running_opp_reward = torch.zeros(batch_size).cuda()

        last_reward = 0
        policy = []

        for t in range(num_steps):
            if args.append_input:
                state = torch.cat([state, payout.to(device)], axis=-1)
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, info, M = env.step(action)

            memory.rewards.append(reward)
            running_reward += reward.squeeze(-1)
            running_opp_reward += info.squeeze(-1)
            last_reward = reward.squeeze(-1)

            if i_episode == max_episodes: # Record policy for the final episode
                policy.append(state.cpu().numpy().tolist()) # Taken from Chris Lu notebooks paper plots-Copy2.ipynb

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

        if i_episode % save_freq == 0:
            ppo.save(os.path.join('runs/' + name + '/test_' + args.game + '_seed' + str(args.seed), f"{i_episode}.pth"))
            with open(os.path.join('runs/' + name + '/test_' + args.game + '_seed' + str(args.seed), f"out_{i_episode}.json"), "w") as f:
                json.dump(rew_means, f)
            with open(os.path.join('runs/' + name + '/test_' + args.game + '_seed' + str(args.seed) + '_policy', 
                                    f"out_{i_episode}.json"), "w") as f:
                json.dump(policy, f)
            print(f"SAVING! {i_episode}")

    ppo.save(os.path.join('runs/' + name + '/test_' + args.game + '_seed' + str(args.seed), f"{i_episode}.pth"))
    with open(os.path.join('runs/' + name + '/test_' + args.game + '_seed' + str(args.seed), f"out_{i_episode}.json"), "w") as f:
        json.dump(rew_means, f)
    with open(os.path.join('runs/' + name + '/test_' + args.game + '_seed' + str(args.seed) + '_policy', 
                            f"out_{i_episode}.json"), "w") as f:
        json.dump(policy, f)
    print(f"SAVING! {i_episode}")
