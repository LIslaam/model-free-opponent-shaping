import os
import json
import torch
import argparse
from environments import NonMfosMetaGames
from utils import setup_wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inverse_sigmoid(x):
    return -torch.log((1 / (x + 1e-8)) - 1)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--game", type=str, default='IPD')
parser.add_argument("--agent", type=str, required=True)
parser.add_argument("--opponent", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--mamaml-id", type=int, default=0)
parser.add_argument("--lr", type=float, default=1)
parser.add_argument("--opp_lr", type=float, default=1)
parser.add_argument("--eval_game", type=str, default="IPD")
args = parser.parse_args()

setup_wandb(vars(args))

if __name__ == "__main__":
    batch_size = 8192
    num_steps = 100
    name = args.exp_name
    p1 = args.agent
    p2 = args.opponent


    if args.seed != None:
        torch.manual_seed(args.seed) # Set seed for reproducability.

    print(f"RUNNING NAME: {'runs/' + name}")
    if not os.path.isdir('runs/' + name):
        os.mkdir('runs/' + name)
        with open(os.path.join('runs/' + name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    if not os.path.isdir('runs/' + name + '/test_' + args.game  + '_seed' + str(args.seed)):
        os.mkdir('runs/' + name + '/test_' + args.game  + '_seed' + str(args.seed))
        with open(os.path.join('runs/' + name + '/test_' + args.game  + '_seed' + str(args.seed), 
                               "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    results = []
    if p1 == "MAMAML" or p2 == "MAMAML":
        for id in range(10):
            env = NonMfosMetaGames(batch_size, lr=args.lr, opp_lr=args.opp_lr, p1=args.agent, p2=args.opponent, 
                                    game=args.game, mmapg_id=args.mamaml_id)
            env.reset()
            running_rew_0 = torch.zeros(batch_size).cuda()
            running_rew_1 = torch.zeros(batch_size).cuda()
            for i in range(num_steps):
                _, r0, r1, M = env.step()
                running_rew_0 += r0.squeeze(-1)
                running_rew_1 += r1.squeeze(-1)
            mean_rew_0 = (running_rew_0.mean() / num_steps).item()
            mean_rew_1 = (running_rew_1.mean() / num_steps).item()

            results.append(
                {
                    "rew": mean_rew_0,
                    "opp_rew": mean_rew_1,
                    "mmapg_id": id,
                }
            )
            print("=" * 100)
            print(f"Done with game: {args.game}, p1: {p1}, p2: {p2}, id: {id}")
            print(f"r0: {mean_rew_0}, r1: {mean_rew_1}")
    else:
        env = NonMfosMetaGames(batch_size, lr=args.lr, opp_lr=args.opp_lr, p1=args.agent, p2=args.opponent, 
                                game=args.game, mmapg_id=args.mamaml_id)
        env.reset()
        running_rew_0 = torch.zeros(batch_size).cuda()
        running_rew_1 = torch.zeros(batch_size).cuda()
        for i in range(num_steps):
            state, r0, r1, M = env.step()
            running_rew_0 += r0.squeeze(-1)
            running_rew_1 += r1.squeeze(-1)
        mean_rew_0 = (running_rew_0.mean() / num_steps).item()
        mean_rew_1 = (running_rew_1.mean() / num_steps).item()

        policy1 = inverse_sigmoid(torch.tensor(state[:batch_size], requires_grad=True).to(device)) # Since state is sigmoided when returned by env.step()
        policy2 = inverse_sigmoid(torch.tensor(state[batch_size:], requires_grad=True).to(device))

        results.append({"rew": mean_rew_0, "opp_rew": mean_rew_1})
        print("=" * 100)
        print(f"Done with game: {args.game}, p1: {p1}, p2: {p2}")
        print(f"r0: {mean_rew_0}, r1: {mean_rew_1}")
        print(mean_rew_0)
        print(mean_rew_1)
    with open(os.path.join('runs/' + name, f"out.json"), "w") as f:
        json.dump(results, f)


############ EVAL SECTION ###############
print("Evaluating on IPD")
env = NonMfosMetaGames(batch_size, p1='STATIC', p2='STATIC', game='IPD')
env.reset(p1_th_ba=policy1, p2_th_ba=policy2)
running_rew_0 = torch.zeros(batch_size).cuda()
running_rew_1 = torch.zeros(batch_size).cuda()

for i in range(1):
    _, r0, r1, M = env.step()
    running_rew_0 += r0.squeeze(-1)
    running_rew_1 += r1.squeeze(-1)
mean_rew_0 = (running_rew_0.mean()).item()
mean_rew_1 = (running_rew_1.mean()).item()
print(f"r0: {mean_rew_0}, r1: {mean_rew_1}")
print(mean_rew_0)
print(mean_rew_1)

with open(os.path.join('runs/' + name + '/test_' + args.game + '_seed' + str(args.seed), "out_eval.json"), "w") as f:
    json.dump({"rew": mean_rew_0, "opp_rew": mean_rew_1}, f)
print(f"SAVING!")