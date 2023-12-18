# Test a pretrained PPO agent
import torch
from ppo import PPO, Memory
from environments import MetaGames
from ga import Auxiliary
from statistics import mean
import os
import argparse
import json
import wandb
from utils.setup_wandb import setup_wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_ppo(args, game, opp, opp_lr, rand_opp, checkpoint):
            ##############################
    
    args.rand_opp = rand_opp # Want to test against a fixed lr opponent
    args.opponent = opp # Override choice of opponent

    K_epochs = 4  # update policy for K epochs

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    batch_size = 4096
    random_seed = args.seed
    num_steps = 100

    save_freq = 250
    name = args.exp_name

    aux = Auxiliary().to(device)

    # if args.rand_opp:
    #     opplr = str(opp_lr).replace('.','_')
    #     print(f"RUNNING NAME: {'runs/' + name + '/test_clone' + game  + '_opplr_' + opplr}")
    #     if not os.path.isdir('runs/' + name + '/test_clone' + game  + '_opplr_' + opplr):
    #         os.mkdir('runs/' + name + '/test_clone' + game  + '_opplr_' + opplr)
    #         with open(os.path.join('runs/' + name + '/test_clone' + game  + '_opplr_' + opplr, 
    #                             "commandline_args.txt"), "w") as f:
    #             json.dump(args.__dict__, f, indent=2)

    # else:
    print(f"RUNNING NAME: {'runs/' + name + '/test_' + game  + '_' + opp + '_lr=' + str(opp_lr)}")
    if not os.path.isdir('runs/' + name + '/test_' + game  + '_' + opp + '_lr=' + str(opp_lr)):
        os.mkdir('runs/' + name + '/test_' + game  + '_' + opp + '_lr=' + str(opp_lr))
        with open(os.path.join('runs/' + name + '/test_' + game  + '_' + opp + '_lr=' + str(opp_lr), 
                            "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

        #if not os.path.isdir('runs/' + name + '/test_' + game  + '_seed' + str(args.seed) + '_policy'):
         #   os.mkdir('runs/' + name + '/test_' + game  + '_seed' + str(args.seed) + '_policy')
          #  with open(os.path.join('runs/' + name + '/test_' + game  + '_seed' + str(args.seed) + '_policy', 
           #                     "commandline_args.txt"), "w") as f:
            #    json.dump(args.__dict__, f, indent=2)

    #################################################
    if args.seed != None:
        torch.manual_seed(random_seed)
    env = MetaGames(batch_size, opponent=args.opponent, game=game, mmapg_id=args.mamaml_id, 
                    opp_lr=opp_lr, rand_opp=args.rand_opp)
    memory = Memory()
    
    action_dim = env.d
    state_dim = env.d * 2
    if args.append_input:
        state_dim = (env.d * 2) + 2 # New input

    ppo = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, args.entropy)

    directory = "runs/" + name + '/'
    checkpoint_path = directory + "{}.pth".format(checkpoint) # max episodes
    print("loading network from : " + checkpoint_path)

    ppo.load(checkpoint_path)
    rew_means = []

    # ONLY ONE EPISODE
    try:
        state, payout = env.reset()
        payout_probs = torch.cat([aux(payout[i].to(device)) for i in range(batch_size)])
    except ValueError:
        state = env.reset()

    running_reward = torch.zeros(batch_size).cuda()
    running_opp_reward = torch.zeros(batch_size).cuda()

    last_reward = 0
    policy = []

    for t in range(num_steps):
        if args.append_input:
            state = torch.cat([state, payout_probs.to(device)], axis=-1) #payout], axis=-1)
        # Running policy_old:
        action = ppo.policy_old.act(state, memory)
        state, reward, info, M = env.step(action)

        memory.rewards.append(reward)
        running_reward += reward.squeeze(-1)
        running_opp_reward += info.squeeze(-1)
        last_reward = reward.squeeze(-1)

        policy.append(state.cpu().numpy().tolist()) # Taken from Chris Lu notebooks paper plots-Copy2.ipynb

        if args.rand_opp:
            for i, value in enumerate([*map(mean, zip(*state.cpu().numpy().tolist()))][:5]):   
                wandb.log({"eval_"+opp+'_'+game+"_rand_opp"+"_state_"+str(i): value})
            for i, value in enumerate([*map(mean, zip(*action.cpu().numpy().tolist()))][:5]):   
                wandb.log({"eval_"+opp+'_'+game+"_rand_opp"+"_action_"+str(i): value})

            wandb.log({"eval_"+opp+'_'+game+"_rand_opp"+"_loss": -running_reward.mean() / num_steps})
            wandb.log({"eval_"+opp+'_'+game+"_rand_opp"+"_opp_loss": -running_opp_reward.mean() / num_steps})

        else:
            for i, value in enumerate([*map(mean, zip(*state.cpu().numpy().tolist()))][:5]):   
                wandb.log({"eval_"+opp+'_'+game+"_opp_lr="+str(opp_lr)+"_state_"+str(i): value})
            for i, value in enumerate([*map(mean, zip(*action.cpu().numpy().tolist()))][:5]):   
                wandb.log({"eval_"+opp+'_'+game+"_opp_lr="+str(opp_lr)+"_action_"+str(i): value})

            wandb.log({"eval_"+opp+'_'+game+"_opp_lr="+str(opp_lr)+"_loss": -running_reward.mean() / num_steps})
            wandb.log({"eval_"+opp+'_'+game+"_opp_lr="+str(opp_lr)+"_opp_loss": -running_opp_reward.mean() / num_steps})


        memory.clear_memory()

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
        ppo.save(os.path.join('runs/' + name + '/test_' + game  + '_' + opp + '_randopp', "eval.pth"))
        with open(os.path.join('runs/' + name + '/test_' + game  + '_' + opp + '_randopp', "out_eval.json"), "w") as f:
            json.dump(rew_means, f)
    else:
        ppo.save(os.path.join('runs/' + name + '/test_' + game  + '_' + opp + '_lr=' + str(opp_lr), "eval.pth"))
        with open(os.path.join('runs/' + name + '/test_' + game  + '_' + opp + '_lr=' + str(opp_lr), "out_eval.json"), "w") as f:
            json.dump(rew_means, f)
    #with open(os.path.join('runs/' + name + '/test_' + game + '_seed' + str(args.seed) + '_policy', 
     #                       f"out_eval.json"), "w") as f:
      #  json.dump(policy, f)
    print(f"SAVING!")

################################################## EVAL ONLY ##################################################

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

for eval_game in ["IPD", "random", "randIPD", "noisyIPD"]:
    for lr in [3, 2.5, 2, 1.5, 1, 0.5, 0.05]:
        for opponent in ["NL", "LOLA"]:
            eval_ppo(args, game=eval_game, opp=opponent, opp_lr=lr, checkpoint=1024)
