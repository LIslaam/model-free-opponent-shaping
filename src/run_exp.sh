#!/bin/bash

#python3 src/main_mfos_ppo.py --game=noisyIPD --opponent=NL --exp_name=mfos_ppo_noisyipd_randopp_nl --rand_opp=True --seed=0
#python3 src/main_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_ipd_randopp_nl --rand_opp=True --seed=0
#python3 src/main_mfos_ppo.py --game=randIPD --opponent=NL --exp_name=mfos_ppo_randipd_randopp_nl --rand_opp=True --seed=0
#python3 src/main_mfos_ppo.py --game=random --opponent=NL --exp_name=mfos_ppo_random_randopp_nl --rand_opp=True --seed=0


#for s in {0..4..1}; do
 #  python src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_input_random_opplr_1_5_nl --append_input=True --opp_lr=1.5 --seed="$s" --checkpoint=1024

for lr in 3; do # 2.5 2 1.5 1 0.5 0.05
   python3 src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp_name=mfos_rnn_noisyipd_randopp_nl --seed=0 --checkpoint=1024 --opp_lr="$lr" --rand_opp=True
   python3 src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp_name=mfos_rnn_ipd_randopp_nl --seed=0 --checkpoint=1024 --opp_lr="$lr" --rand_opp=True
   python3 src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp_name=mfos_rnn_randipd_randopp_nl --seed=0 --checkpoint=1024 --opp_lr="$lr" --rand_opp=True
   python3 src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp_name=mfos_rnn_random_randopp_nl --seed=0 --checkpoint=1024 --opp_lr="$lr" --rand_opp=True


  #  python src/eval_mfos_ppo.py --game=noisyIPD --opponent=NL --exp-name=mfos_ppo_randnoisyipd_opplr_0_005_nl_run2 --opp_lr=0.005 #--seed="$s" --checkpoint=1024
   # python src/eval_mfos_ppo.py --game=noisyIPD --opponent=NL --exp-name=mfos_ppo_random_opplr_0_005_nl_run2 --opp_lr=0.005 #--seed="$s" --checkpoint=1024

done