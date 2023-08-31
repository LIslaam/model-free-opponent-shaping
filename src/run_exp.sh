#!/bin/bash

python3 src/main_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_input_random_opplr_2_nl --opp_lr=2 --append_input=True --seed=0
python3 src/main_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_input_random_opplr_1.5_nl --opp_lr=1.5 --append_input=True --seed=0
python3 src/main_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_input_random_opplr_0_5_nl --opp_lr=0.5 --append_input=True --seed=0
python3 src/main_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_input_random_opplr_0_05_nl --opp_lr=0.05 --append_input=True --seed=0

for s in {0..4..1}; do
   python3 src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_input_random_opplr_2_5_nl --opp_lr=2.5 --append_input=True --seed="$s" --checkpoint=1024
   python3 src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_input_random_opplr_2_nl --opp_lr=2 --append_input=True --seed="$s" --checkpoint=1024
   python3 src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_input_random_opplr_1_5_nl --opp_lr=1.5 --append_input=True --seed="$s" --checkpoint=1024
   python3 src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_input_random_opplr_0_5_nl --opp_lr=0.5 --append_input=True --seed="$s" --checkpoint=1024
   python3 src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_input_random_opplr_0_05_nl --opp_lr=0.05 --append_input=True --seed="$s" --checkpoint=1024


 #   python src/eval_mfos_ppo.py --game=noisyIPD --opponent=NL --exp-name=mfos_ppo_Fsyipd_opplr_0_005_nl_run2 --opp_lr=0.005 #--seed="$s" --checkpoint=1024
  #  python src/eval_mfos_ppo.py --game=noisyIPD --opponent=NL --exp-name=mfos_ppo_randnoisyipd_opplr_0_005_nl_run2 --opp_lr=0.005 #--seed="$s" --checkpoint=1024
   # python src/eval_mfos_ppo.py --game=noisyIPD --opponent=NL --exp-name=mfos_ppo_random_opplr_0_005_nl_run2 --opp_lr=0.005 #--seed="$s" --checkpoint=1024

done