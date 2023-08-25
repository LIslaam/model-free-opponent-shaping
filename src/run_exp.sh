#!/bin/bash

#for s in {0..500..1}; do
#    python src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp-name=mfos_rnn_noisyipd_x5e-2_nl  --checkpoint=2048 --seed="$s"


"""
python src/main_mfos_ppo.py --game=IPD --opponent=NL --exp-name=mfos_ppo_ipd_opplr_2_5_nl --opp_lr=2.5
python src/main_mfos_ppo.py --game=IPD --opponent=NL --exp-name=mfos_ppo_ipd_opplr_2_nl --opp_lr=2
python src/main_mfos_ppo.py --game=IPD --opponent=NL --exp-name=mfos_ppo_ipd_opplr_1_5_nl --opp_lr=1.5
python src/main_mfos_ppo.py --game=IPD --opponent=NL --exp-name=mfos_ppo_ipd_opplr_0_5_nl --opp_lr=0.5

python src/main_mfos_ppo.py --game=randIPD --opponent=NL --exp-name=mfos_ppo_randipd_opplr_2_5_nl --opp_lr=2.5
python src/main_mfos_ppo.py --game=randIPD --opponent=NL --exp-name=mfos_ppo_randipd_opplr_2_nl --opp_lr=2
python src/main_mfos_ppo.py --game=randIPD --opponent=NL --exp-name=mfos_ppo_randipd_opplr_1_5_nl --opp_lr=1.5
python src/main_mfos_ppo.py --game=randIPD --opponent=NL --exp-name=mfos_ppo_randipd_opplr_0_5_nl --opp_lr=0.5
"""

python src/main_mfos_ppo.py --game=random --opponent=NL --exp-name=mfos_ppo_random_opplr_0_05_nl --opp_lr=0.05
python src/main_mfos_ppo.py --game=random --opponent=NL --exp-name=mfos_ppo_random_opplr_2_nl --opp_lr=2
python src/main_mfos_ppo.py --game=random --opponent=NL --exp-name=mfos_ppo_random_opplr_1_5_nl --opp_lr=1.5
python src/main_mfos_ppo.py --game=random --opponent=NL --exp-name=mfos_ppo_random_opplr_0_5_nl --opp_lr=0.5


"""
for s in {0..4..1}; do
    python src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp-name=mfos_ppo_random_seed0_nl --checkpoint=1024  --seed="$s"
    python src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp-name=mfos_ppo_random_seed1_nl --checkpoint=1024  --seed="$s"
    python src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp-name=mfos_ppo_random_seed2_nl --checkpoint=1024  --seed="$s"
    python src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp-name=mfos_ppo_random_seed3_nl --checkpoint=1024  --seed="$s"
    python src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp-name=mfos_ppo_random_seed4_nl --checkpoint=1024  --seed="$s"

    python src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp-name=mfos_rnn_random_seed0_nl --checkpoint=2048  --seed="$s"
    python src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp-name=mfos_rnn_random_seed1_nl --checkpoint=2048  --seed="$s"
    python src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp-name=mfos_rnn_random_seed2_nl --checkpoint=2048  --seed="$s"
    python src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp-name=mfos_rnn_random_seed3_nl --checkpoint=2048  --seed="$s"
    python src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp-name=mfos_rnn_random_seed4_nl --checkpoint=2048  --seed="$s"
"""
done