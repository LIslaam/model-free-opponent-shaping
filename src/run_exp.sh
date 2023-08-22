#!/bin/bash

for s in {0..500..1}; do
    python src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp-name=mfos_ppo_noisyipd_x5e-2_seed4_nl --checkpoint=1024 --seed="$s"
done