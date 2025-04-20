#!/bin/zsh
python3 model_calculator.py --model 70e9 --data 1e12
python3 compute_optimal.py --loss 1.947 --inference_tokens 2e12
python3 cost_optimal.py \
    --loss 1.947 \
    --inference_reqs 1e10 \
    --input_tok 1000 \
    --output_tok 250 \
    --train_gpu_type 'A100_80' \
    --train_gpu_cost 1.40 \
    --inference_gpu_type 'A100_40' \
    --inference_dtype int8 \
    --inference_gpu_cost 0.60 \
    --train_mfu 0.5 \
    --prefill_mfu 0.4 \
    --generation_mfu 0.2
python3 model_calculator.py --chinchilla --compute 1e24
python3 model_calculator.py --chinchilla --model 70e9
python3 model_calculator.py --model 13e9 --data 1e12
python3 model_calculator.py --model 7e9 --compute 1e23
python3 cost_calculator.py --model 70e9 --data 10e12
python3 cost_calculator.py \
    --model 30e9 \
    --chinchilla \
    --train_gpu_cost 1.50 \
    --inference_gpu_type 'A100_40' \
    --inference_dtype int8 \
    --inference_gpu_cost 0.60 \
    --input_tok 500 \
    --output_tok 100 \
    --inference_reqs 1e9
python3 huber_optimize.py --data trainingresults.csv
