import torch
from ga import BatchedPolicies
from environments import MetaGames
from ga import Auxiliary
import os
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, required=True)
parser.add_argument("--opponent", type=str, required=True)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--exp-name", type=str, default="")
parser.add_argument("--mamaml-id", type=int, default=0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--append_input", type=bool, default=False)
args = parser.parse_args()


if __name__ == "__main__":
    ############################################
    lr = 0.5
    num_species = 2048
    max_episodes = 64
    test_size = 2048
    batch_size = 128
    random_seed = None
    num_steps = 100
    save_freq = 16
    name = args.exp_name

    print(f"RUNNING NAME: {'runs/' + name}")
    if not os.path.isdir('runs/' + name):
        os.mkdir('runs/' + name)
        with open(os.path.join('runs/' + name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    #############################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.seed != None:
        torch.manual_seed(random_seed) # Set seed for reproducability.
    # creating environment
    env = MetaGames(num_species * batch_size, opponent=args.opponent, game=args.game, mmapg_id=args.mamaml_id)
    env_test = MetaGames(test_size, opponent=args.opponent, game=args.game, mmapg_id=args.mamaml_id)

    action_dim = env.d
    state_dim = (env.d * 2)
    if args.append_input:
        state_dim = (env.d * 2) + 2 # 2 outputs from Auxiliary NN

    _, payout = env.reset()
    agent = BatchedPolicies(state_dim, action_dim, num_species, payout=payout, device=device)
    aux = Auxiliary().to(device) # Use this to initialise state before using GA

    rew_means = []
    for i in range(1, max_episodes + 1):
        agent.create_species()
        rewards_nb = torch.zeros((num_species, batch_size), device=device)
        opp_rewards_nb = torch.zeros((num_species, batch_size), device=device)

        try:
            state, payout = env.reset()
            payout_probs = torch.Tensor.repeat(aux(payout), (num_species*batch_size,1))
            if args.append_input:
                state = torch.cat([state, payout_probs], axis=-1)
            state = state.view(num_species, batch_size, state_dim)
        except ValueError:
            state = env.reset().view(num_species, batch_size, state_dim)

        print(state.size())
        for n in range(num_steps):
            with torch.no_grad():
                action, payout_probs = agent(state, payout=payout)
                payout_probs = torch.Tensor.repeat(payout_probs, (num_species*batch_size,1))
                action = action.reshape(num_species * batch_size, action_dim)

            state, reward, info, M = env.step(action)
            if args.append_input:
                state = torch.cat([state, payout_probs], axis=-1)
            state = state.view(num_species, batch_size, state_dim)
            rewards_nb += reward.view(num_species, batch_size)
            opp_rewards_nb += info.view(num_species, batch_size)

        rewards_n = rewards_nb.mean(-1)
        opp_rewards_n = opp_rewards_nb.mean(-1)

        best = torch.argmax(rewards_n)
        running_reward = torch.zeros(test_size).cuda()
        running_opp_reward = torch.zeros(test_size).cuda()

        try:
            state, payout = env_test.reset()
            payout_probs = aux(payout)
        except ValueError:
            state = env_test.reset()

        for n in range(num_steps):
            with torch.no_grad():

                action = agent.forward_eval(state, payout, idx=best)

            state, reward, info, M = env_test.step(action)

            running_reward += reward.squeeze(-1)
            running_opp_reward += info.squeeze(-1)
        new_reward = running_reward.mean()
        new_opp_reward = running_opp_reward.mean()

        running_reward = torch.zeros(test_size).cuda()
        running_opp_reward = torch.zeros(test_size).cuda()

        try:
            state, payout = env_test.reset()
            payout_probs = aux(payout)
        except ValueError:
            state = env_test.reset()

        for n in range(num_steps):
            with torch.no_grad():
                if args.append_input:
                    state = torch.cat([state, payout_probs.to(device)], axis=-1)
                action = agent.forward_eval(state, payout, idx=None)

            state, reward, info, M = env_test.step(action)
            running_reward += reward.squeeze(-1)
            running_opp_reward += info.squeeze(-1)
        old_reward = running_reward.mean()
        old_opp_reward = running_opp_reward.mean()
        print("=" * 10)
        with torch.no_grad():

            if new_reward > old_reward:
                print("REPLACING!")
                agent.l_1 = torch.nn.Parameter(agent.fl1_nsd[best])
                agent.b_1 = torch.nn.Parameter(agent.fb1_nd[best])
                agent.l_2 = torch.nn.Parameter(agent.fl2_nda[best])
                agent.b_2 = torch.nn.Parameter(agent.fb2_na[best])
                rew_means.append(
                    {
                        "new_best": True,
                        "rew": (new_reward / num_steps).item(),
                        "opp_rew": (new_opp_reward / num_steps).item(),
                    }
                )
            else:
                print("NO BETTER FOUND!")
                rew_means.append(
                    {
                        "new_best": False,
                        "rew": (old_reward / num_steps).item(),
                        "opp_rew": (old_opp_reward / num_steps).item(),
                    }
                )

        print(f"New Reward: {new_reward / num_steps}")
        print(f"New Opp Reward: {new_opp_reward / num_steps}")

        print(f"Old Reward: {old_reward / num_steps}")
        print(f"Old Opp Reward: {old_opp_reward / num_steps}")

        #print(agent.state_dict())
        if i % save_freq == 0:
            torch.save(agent.state_dict(), os.path.join('runs/' + name, f"{i}.pth"))
            with open(os.path.join('runs/' + name, f"out_{i}.json"), "w") as f:
                json.dump(rew_means, f)
            print(f"SAVING! {i}")

    torch.save(agent.state_dict(), os.path.join('runs/' + name, f"{i}.pth"))
    with open(os.path.join('runs/' + name, f"out_{i}.json"), "w") as f:
        json.dump(rew_means, f)
    print(f"SAVING! {i}")
