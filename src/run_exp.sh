#!/bin/bash

#for s in {0..4..1}; do
python3 src/main_non_mfos.py --game=noisyIPD --agent=NL --opponent=NL --exp_name=nl_nl_noisyipd_seed0 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=NL --opponent=NL --exp_name=nl_nl_noisyipd_seed1 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=NL --opponent=NL --exp_name=nl_nl_noisyipd_seed2 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=NL --opponent=NL --exp_name=nl_nl_noisyipd_seed3 #--seed="$s" --checkpoint=512

python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=NL --exp_name=lola_nl_noisyipd_seed0 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=NL --exp_name=lola_nl_noisyipd_seed1 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=NL --exp_name=lola_nl_noisyipd_seed2 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=NL --exp_name=lolaipd_nl_noisyipd_seed3 #--seed="$s" --checkpoint=512

python3 src/main_non_mfos.py --game=noisyIPD --agent=STATIC --opponent=NL --exp_name=static_nl_noisyipd_seed0 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=STATIC --opponent=NL --exp_name=static_nl_noisyipd_seed1 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=STATIC --opponent=NL --exp_name=static_nl_noisyipd_seed2 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=STATIC --opponent=NL --exp_name=static_nl_noisyipd_seed3 #--seed="$s" --checkpoint=512

python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=STATIC --exp_name=lola_static_noisyipd_seed0 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=STATIC --exp_name=lola_static_noisyipd_seed1 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=STATIC --exp_name=lola_static_noisyipd_seed2 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=STATIC --exp_name=lola_static_noisyipd_seed3 #--seed="$s" --checkpoint=512

python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=LOLA --exp_name=lola_lola_noisyipd_seed0 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=LOLA --exp_name=lola_lola_noisyipd_seed1 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=LOLA --exp_name=lola_lola_noisyipd_seed2 #--seed="$s" --checkpoint=512
python3 src/main_non_mfos.py --game=noisyIPD --agent=LOLA --opponent=LOLA --exp_name=lola_lola_noisyipd_seed3 #--seed="$s" --checkpoint=512


 #   python src/eval_mfos_ppo.py --game=noisyIPD --opponent=NL --exp-name=mfos_ppo_Fsyipd_opplr_0_005_nl_run2 --opp_lr=0.005 #--seed="$s" --checkpoint=1024
  #  python src/eval_mfos_ppo.py --game=noisyIPD --opponent=NL --exp-name=mfos_ppo_randnoisyipd_opplr_0_005_nl_run2 --opp_lr=0.005 #--seed="$s" --checkpoint=1024
   # python src/eval_mfos_ppo.py --game=noisyIPD --opponent=NL --exp-name=mfos_ppo_random_opplr_0_005_nl_run2 --opp_lr=0.005 #--seed="$s" --checkpoint=1024

done