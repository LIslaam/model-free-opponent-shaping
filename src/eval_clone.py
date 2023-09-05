import torch
from environments import MetaGames
from behavioural_clone import get_data, BehaviouralCloning
from ga import Auxiliary
import os
import argparse
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, required=True)
parser.add_argument("--opponent", type=str, required=True)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--mamaml-id", type=int, default=0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--append_input", type=bool, default=False)
parser.add_argument("--opp_lr", type=float, default=1)
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--rand_opp", type=bool, default=False) # Randomly sample opponent learning rates
args = parser.parse_args()


if __name__ == "__main__":
                    ###################################
    batch_size = 4096
    random_seed = args.seed
    num_steps = 100

    name = args.exp_name
    aux = Auxiliary().to(device)

    if args.rand_opp:
        opplr = str(args.opp_lr).replace('.','_')
        print(f"RUNNING NAME: {'runs/' + name + '/test_clone' + args.game  + '_opplr_' + opplr}")
        if not os.path.isdir('runs/' + name + '/test_clone' + args.game  + '_opplr_' + opplr):
            os.mkdir('runs/' + name + '/test_clone' + args.game  + '_opplr_' + opplr)
            with open(os.path.join('runs/' + name + '/test_' + args.game  + '_opplr_' + opplr, 
                                "commandline_args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=2)
    else:
        raise NotImplementedError

    directory = 'runs/' + name + '/state_action/out_' + str(args.checkpoint) + '.json'

    if args.seed != None:
        torch.manual_seed(random_seed)
    env = MetaGames(batch_size, opponent=args.opponent, game=args.game, mmapg_id=args.mamaml_id, 
                    opp_lr=args.opp_lr, rand_opp=args.rand_opp)
    

    state_data, action_data = get_data(directory)
    clone = BehaviouralCloning(state_data, action_data) # Getting the action from behavioural clone
    action = clone.run()
    print(action, action.size())

    # ONLY ONE EPISODE
    try:
        state, payout = env.reset()
        payout_probs = torch.cat([aux(payout[i].to(device)) for i in range(batch_size)])
    except ValueError:
        state = env.reset()

    running_reward = torch.zeros(batch_size).cuda()
    running_opp_reward = torch.zeros(batch_size).cuda()

    last_reward = 0
    rew_means = []

    for t in range(num_steps):
        if args.append_input:
            state = torch.cat([state, payout_probs.to(device)], axis=-1) #payout], axis=-1)
        # Running policy_old:
        state, reward, info, M = env.step(action) # Same action each time

        running_reward += reward.squeeze(-1)
        running_opp_reward += info.squeeze(-1)
        last_reward = reward.squeeze(-1)

        print("=" * 100, flush=True)

        print(f"loss: {-running_reward.mean() / num_steps}", flush=True)

        rew_means.append(
            {
                "rew": (reward.squeeze(-1).mean()).item(),
                "opp_rew": (info.squeeze(-1).mean()).item(),
            }
        )

        print(f"opponent loss: {-running_opp_reward.mean() / num_steps}", flush=True)


    if args.rand_opp:
        with open(os.path.join('runs/' + name + '/test_clone' + args.game + '_opplr_' + opplr, "out_eval.json"), "w") as f:
            json.dump(rew_means, f)
    print(f"SAVING!")
