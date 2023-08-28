#!/bin/bash

python3 src/non_mfos.py --game=IPD --agent=NL --opponent=NL --exp_name=nl_nl_ipd_seed1 --seed=1
python3 src/non_mfos.py --game=IPD --agent=NL --opponent=NL --exp_name=nl_nl_ipd_seed2 --seed=2
python3 src/non_mfos.py --game=IPD --agent=NL --opponent=NL --exp_name=nl_nl_ipd_seed3 --seed=3

python3 src/non_mfos.py --game=IPD --agent=LOLA --opponent=NL --exp_name=lola_nl_ipd_seed1 --seed=1
python3 src/non_mfos.py --game=IPD --agent=LOLA --opponent=NL --exp_name=lola_nl_ipd_seed2 --seed=2
python3 src/non_mfos.py --game=IPD --agent=LOLA --opponent=NL --exp_name=lola_nl_ipd_seed3 --seed=3

python3 src/non_mfos.py --game=IPD --agent=STATIC --opponent=NL --exp_name=static_nl_ipd_seed1 --seed=1
python3 src/non_mfos.py --game=IPD --agent=STATIC --opponent=NL --exp_name=static_nl_ipd_seed2 --seed=2
python3 src/non_mfos.py --game=IPD --agent=STATIC --opponent=NL --exp_name=static_nl_ipd_seed3 --seed=3

python3 src/non_mfos.py --game=IPD --agent=LOLA --opponent=STATIC --exp_name=lola_static_ipd_seed0 --seed=0
python3 src/non_mfos.py --game=IPD --agent=LOLA --opponent=STATIC --exp_name=lola_static_ipd_seed1 --seed=1
python3 src/non_mfos.py --game=IPD --agent=LOLA --opponent=STATIC --exp_name=lola_static_ipd_seed2 --seed=2
python3 src/non_mfos.py --game=IPD --agent=LOLA --opponent=STATIC --exp_name=lola_static_ipd_seed3 --seed=3


#for s in {0..4..1}; do
 #   python src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp-name=mfos_ppo_randipd_opplr_2_5_nl --opp_lr=2.5 --seed="$s" --checkpoint=1024

done