#!/bin/bash

for s in {0..1000..1}; do
    python src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp-name=mfos_rnn_noisyipd_x1e0_nl --checkpoint=2048 --seed="$s"
done