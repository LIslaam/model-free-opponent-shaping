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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_ppo(args, game, opp_lr):
            ##############################
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

    if args.rand_opp:
        opplr = str(opp_lr).replace('.','_')
        print(f"RUNNING NAME: {'runs/' + name + '/test_clone' + game  + '_opplr_' + opplr}")
        if not os.path.isdir('runs/' + name + '/test_clone' + game  + '_opplr_' + opplr):
            os.mkdir('runs/' + name + '/test_clone' + game  + '_opplr_' + opplr)
            with open(os.path.join('runs/' + name + '/test_clone' + game  + '_opplr_' + opplr, 
                                "commandline_args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=2)

    else:
        print(f"RUNNING NAME: {'runs/' + name + '/test_' + game  + '_seed' + str(args.seed)}")
        if not os.path.isdir('runs/' + name + '/test_' + game  + '_seed' + str(args.seed)):
            os.mkdir('runs/' + name + '/test_' + game  + '_seed' + str(args.seed))
            with open(os.path.join('runs/' + name + '/test_' + game  + '_seed' + str(args.seed), 
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
    checkpoint_path = directory + "{}.pth".format(args.checkpoint) # max episodes
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
        
        for i, value in enumerate([*map(mean, zip(*state.cpu().numpy().tolist()))][:5]):   
            wandb.log({"eval_"+game+"_opp_lr="+opp_lr+"_state_"+str(i): value})
        for i, value in enumerate([*map(mean, zip(*action.cpu().numpy().tolist()))][:5]):   
            wandb.log({"eval_"+game+"_opp_lr="+opp_lr+"_action_"+str(i): value})

        wandb.log({"eval_"+game+"_opp_lr="+opp_lr+"_loss": -running_reward.mean() / num_steps, 
                   "eval_"+game+"_opp_lr="+opp_lr+"_loss": -running_opp_reward.mean() / num_steps})


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
        ppo.save(os.path.join('runs/' + name + '/test_clone' + game + '_opplr_' + opplr, "eval.pth"))
        with open(os.path.join('runs/' + name + '/test_clone' + game + '_opplr_' + opplr, "out_eval.json"), "w") as f:
            json.dump(rew_means, f)
    else:
        ppo.save(os.path.join('runs/' + name + '/test_' + game + '_seed' + str(args.seed), "eval.pth"))
        with open(os.path.join('runs/' + name + '/test_' + game + '_seed' + str(args.seed), "out_eval.json"), "w") as f:
            json.dump(rew_means, f)
    #with open(os.path.join('runs/' + name + '/test_' + game + '_seed' + str(args.seed) + '_policy', 
     #                       f"out_eval.json"), "w") as f:
      #  json.dump(policy, f)
    print(f"SAVING!")
