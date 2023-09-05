#!/bin/bash


for lr in 3 2.5 2 1.5 1 0.5 0.05; do
  python3 src/eval_clone.py --game=IPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_ppo_input_ipd_randopp_nl --rand_opp=True --seed=0 --append_input=True --opp_lr="$lr" --checkpoint=512
#python src/main_mfos_ppo.py --game=noisyIPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_ppo_input_noisyipd_randopp_nl --rand_opp=True --collect_data=True --append_input=True
#python src/main_mfos_ppo.py --game=noisyIPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_ppo_noisyipd_randopp_nl --rand_opp=True --collect_data=True


#for s in {0..4..1}; do
 #  python src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_ipd_opplr_0_05_nl_run3 --opp_lr=0.05 --seed="$s" --checkpoint=1024

   #python3 src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp_name=mfos_rnn_ipd_randopp_nl_run2 --rand_opp=True --seed=1 --opp_lr="$lr" --checkpoint=2048

  # python3 src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp_name=mfos_rnn_ipd_randopp_nl --seed=0 --checkpoint=1024 --opp_lr="$lr" --rand_opp=True
  # python3 src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp_name=mfos_rnn_randipd_randopp_nl --seed=0 --checkpoint=1024 --opp_lr="$lr" --rand_opp=True
   #python3 src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp_name=mfos_rnn_random_randopp_nl --seed=0 --checkpoint=1024 --opp_lr="$lr" --rand_opp=True

done