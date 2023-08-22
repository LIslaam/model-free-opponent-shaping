#!/bin/bash

for s in {0..500..1}; do
    python src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp-name=mfos_rnn_input_noisyipd_x5e-2_nl --append_input=True --checkpoint=2048 --seed="$s"
done