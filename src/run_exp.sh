#!/bin/bash

for s in {0..4..1}; do
  python3 src/main_mfos_ppo.py --game=noisyIPD --opponent=NL --exp_name=mfos_ppo_noisyipd_x0_05_nl_seed"$s" --seed="$s"
done
for s in {0..2..1}; do

done

#python3 src/main_mfos_rnn.py --game=IPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_newrnn_input_ipd_randopp_nl --rand_opp=True --seed=5 --append_input=True --collect_data=True
#python3 src/main_mfos_rnn.py --game=randIPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_newrnn_input_randipd_randopp_nl --rand_opp=True --seed=5 --append_input=True --collect_data=True
#python3 src/main_mfos_rnn.py --game=random --opponent=NL --exp_name=DATA_COLLECTION_mfos_newrnn_input_random_randopp_nl --rand_opp=True --seed=5 --append_input=True --collect_data=True
#python3 src/main_mfos_rnn.py --game=noisyIPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_newrnn_input_noisyipd_randopp_nl --rand_opp=True --seed=5 --append_input=True --collect_data=True

#python3 src/main_mfos_ppo.py --game=IPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_ppo_input_ipd_nl --seed=0 --append_input=True --collect_data=True
#python3 src/main_mfos_ppo.py --game=randIPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_ppo_input_randipd_nl --seed=0 --append_input=True --collect_data=True
#python3 src/main_mfos_ppo.py --game=random --opponent=NL --exp_name=DATA_COLLECTION_mfos_ppo_input_random_nl --seed=0 --append_input=True --collect_data=True
#python3 src/main_mfos_ppo.py --game=noisyIPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_ppo_input_noisyipd_nl --seed=0 --append_input=True --collect_data=True

#for lr in 1; do # 3 2.5 2 1.5 1 0.5 0.05
#python3 src/eval_clone.py --game=IPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_ppo_input_ipd_randopp_nl --rand_opp=True --append_input=True --opp_lr="$lr" --checkpoint=2048
#python src/main_mfos_ppo.py --game=noisyIPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_ppo_input_noisyipd_randopp_nl --rand_opp=True --collect_data=True --append_input=True
#python src/main_mfos_ppo.py --game=noisyIPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_ppo_noisyipd_randopp_nl --rand_opp=True --collect_data=True

#for s in 1; do #{0..4..1}
 # python3 src/eval_clone.py --game=IPD --opponent=NL --exp_name=DATA_COLLECTION_mfos_ppo_input_ipd_nl --seed="$s" --append_input=True --checkpoint=2048
 #  python src/eval_mfos_ppo.py --game=IPD --opponent=NL --exp_name=mfos_ppo_ipd_opplr_0_05_nl_run3 --opp_lr=0.05 --seed="$s" --checkpoint=1024

  # python3 src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp_name=mfos_rnn_ipd_randopp_nl --seed=0 --checkpoint=1024 --opp_lr="$lr" --rand_opp=True
  # python3 src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp_name=mfos_rnn_randipd_randopp_nl --seed=0 --checkpoint=1024 --opp_lr="$lr" --rand_opp=True
   #python3 src/eval_mfos_rnn.py --game=IPD --opponent=NL --exp_name=mfos_rnn_random_randopp_nl --seed=0 --checkpoint=1024 --opp_lr="$lr" --rand_opp=True

