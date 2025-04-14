This is the codebase for "Beyond Chinchilla Optimal: Accounting for Inference in Language Model Scaling Laws."

Paper Link: https://arxiv.org/abs/2401.00448

Published in ICML 2024. An earlier version was published in a NeurIPS 2023 Workshop.

# Summary
The core idea behind this work is that LLM developers who are going to pretrain and serve a large language model should train their models to minimize their costs over the *model's entire lifetime*. Traditional scaling laws only factor in the cost of pretraining. However, if developers can estimate their inference demand ahead of time, they should train models smaller and longer than traditional (Chinchilla) scaling laws recommend since they have to pay for serving costs as well.

This repository contains the **code** and **data** for our paper. The **code** computes exactly **how to size your models and training datasets properly** to reach a model quality target and run inference, assuming you can estimate your inference demand beforehand. The **data** includes the final loss values and downstream evalution results from the **47 training runs** used to validate our scaling laws.

# Installation
Clone the repo and run:
```shell
pip install -r requirements.txt
```
# Scripts
## 1. compute_optimal.py
```
python compute_optimal.py --loss <your model loss> --tokens <expected inference demand> --<optional-args> ...
```
A script that takes in a model quality (loss) and expected inference demand (tokens), and tells you the minimum
number of FLOPs required to train a model to reach that quality and then run inference for your expected inference demand.
This **minimizes compute** over the **model's entire lifetime** (training + inference).

Example use case:

You are planning on pretraining your own model and deploying it at wide scale. You want to ensure you're doing it compute optimally. 
You should:
1. Determine the quality of model you want to train, as measured by loss on your training dataset.

    (a) Typically, downstream evaluation results are highly correlated to loss value. However, the process of converting from a downstream metric you care about to the loss required to achieve this metric will have to be done separately from this repository.

    (b) Alternatively, you can use our script `model_calculator.py` and pass in the model size and dataset length you're considering training to get the model's expected loss according to scaling laws. For example, if you have a trillion tokens and are considering a 70B model, you'd run `python model_calculator.py --model 70e9 --data 1e12` to get this loss value (`1.947`).

2. Run `compute_optimal.py` with your loss value and your expected inference demand. This script will return the model size and number of training tokens you'll need to reach your desired quality (loss) that requires the least compute over the model lifetime, including inference. We also show the expected savings over training a model according to the Chinchilla scaling laws, and then deploying it.

    Continuining our above example, if you expect a total of 2 trillion tokens of inference demand over the model's lifetime, you would run `python compute_optimal.py --loss 1.947 --inference_tokens 2e12`

    Excerpt from Output:
    ```shell
    > python compute_optimal.py --loss 1.947 --inference_tokens 2e12
    > ...
    *** (Lifetime) Compute-Optimal Model ***
    ----------------------------------------
    Model Size (params):     2.418e+10
    Training Data (tok):     2.657e+12
    ......
    The Compute-Optimal model should have 70.96% of the parameters of an equal-quality Chinchilla-style model.
    The Compute-Optimal model should be trained on 146.77% of the tokens of an equal-quality Chinchilla-style model.
    The Compute-Optimal model should cost 95.22% of an equal-quality Chinchilla-style model.
    ```

Run `python compute_optimal.py --help` for a full list of arguments.
### Estimating Inference Demand
Some tips for estimating lifetime inference demand. 

Ask yourself the following questions:
- How long do I expect to serve this model (e.g. 6 months, 1 year) before it is replaced by another?
- How many requests do I expect per day?
    - This can be broken down into "How many users do I expect to use my model?" and "How often do I expect each user to use the model?"
- How long do I expect the average request to be? (Input + Output tokens)

## 2. cost_optimal.py
This script returns the **minimum cost** to train a model to reach your given quality (loss) and then run inference for your expected inference demand. While `compute_optimal.py` minimizes FLOPs, `cost_optimal.py` minimizes costs, which matters more for real-world developers.

This script takes in a model quality (loss), expected inference demand over the model's lifetime, and hardware configuration.
Inference demand has three components:
- Number of inference requests
- Average input tokens per request
- Average output tokens per request

The hardware configuration includes:
- Training and Inference GPU type (A100 or H100)
- Training and inference GPU cost per hour
- Training and Inference dtype (FP8, INT8, or BF16)
- Training, prefill, and decode utilization (MFU).

We provide reasonable defaults for the hardware configuration, so these parameters are optional and will default to the ones we have chosen if not provided. Usage of the script is similar to `compute_optimal.py`, but with these extra parameters.

Examples:
- What is the most cost-effective way to train a model with loss 1.947 and deploy it, assuming I am training on A100-80GBs @ 1.40/hr and running inference on A100-40GBs @ 0.60/hr in INT8. I expect a total of 10B inference requests over the model's lifetime, with each request averaging 1000 input tokens and 250 output tokens. I expect my training utilization (MFU) to be 50%, my prefill utilization to be 40%, and my decode utilization to be 20%.

    - ```python
        python cost_optimal.py --loss 1.947 --inference_reqs 1e10 --input_tok 1000 --output_tok 250 --train_gpu_type A100_80 --train_gpu_cost 1.40 --inference_gpu_type 'A100_40' --inference_dtype int8 --inference_gpu_cost 0.60 --train_mfu 0.5 --prefill_mfu 0.4 --generation_mfu 0.2
        ```
    - Excerpt from Output:
        ```shell
        *** Cost-Optimal Model ***
        ----------------------------------------
        Model Size (params):     2.053e+10
        Training Data (tok):     3.302e+12
        ......
        The Cost-Optimal model should have 60.23% of the parameters of an equal-quality Chinchilla-style model.
        The Cost-Optimal model should be trained on 182.41% of the tokens of an equal-quality Chinchilla-style model.
        The Cost-Optimal model should cost 88.76% of an equal-quality Chinchilla-style model.
        ```


## 3. model_calculator.py
A helper script that calculates model properties based on the Chinchilla scaling laws.
You can provide:
- Model size (`--model`) and training data size (`--data`) --> Output: Expected final training loss, based on Chinchilla, and amount of compute required to train the model.
- Model size and training compute budget (`--compute`) --> Output: Training data length you can afford to train your model on with your budget, and expected final training loss
- Training data size and training compute budget --> Output: Model size you can afford to train with your compute budget and data, and expected final training loss

In other words, provide two of {Compute budget, Model parameters, Training tokens}, and this script will return the missing value, and the expected loss.

Alternatively, if you provide the flag
- `--chinchilla`, then you only need to provide a single value of the four: Compute Budget, model parameters, training tokens, or loss. The code will then assume that you wish to train your model so that it minimizes *training computation* according to the Chinchilla scaling laws (i.e. it is Chinchilla-optimal).

Examples of questions this script can answer:
- What is the absolute best quality model I can train with 1e24 training flops?
    - `python model_calculator.py --chinchilla --compute 1e24`

- What is the training-compute-optimal amount of training data for a 70B model?
    - `python model_calculator.py --chinchilla --model 70e9`
- What loss can I expect if I train a 13B model on 1 trillion tokens?
    - `python model_calculator.py --model 13e9 --data 1e12`
- How much data can I afford to train on if I have a 1e23 training FLOP budget and I want a 7B model?
    - `python model_calculator.py --model 7e9 --compute 1e23`

If you just want to play around with the Chinchilla scaling laws, use this script with the `--chinchilla` flag.

## 4. cost_calculator.py
A helper script, similar to `model_calculator.py`, except that it also calculates the total cost. In addition to taking in the inputs of `model_calculator.py`, it also allows for hardware configuration inputs:
- Training and Inference GPU type (A100 or H100)
- Training and inference GPU cost per hour
- Training and Inference dtype (FP8, INT8, or BF16)
- Training, prefill, and decode utilization (MFU).
We provide reasonable defaults for the hardware configuration, so these parameters are optional and will default to the ones we have chosen if not provided.

You can also provide the expected inference demand over the model's lifetime to get the lifetime model costs, including inference. Inference demand has three components:
- Number of inference requests
- Average input tokens per request
- Average output tokens per request
We again provide reasonable defaults.

Example questions you can answer with this script?
- How much would it cost to train a 70B model on 10 trillion tokens on an H100?
    - `python cost_calculator.py --model 70e9 --data 10e12`

- How much would it cost to train a 30B Chinchilla-optimal model an H100 @ $1.50 per hour, and then run inference on 1 billion requests, with an average of 500 input and 100 output tokens per request on an A100 40GB @ $0.60 per hour?

    - ```python
        python cost_calculator.py --model 30e9 --chinchilla --train_gpu_cost 1.50 --inference_gpu_type 'A100_40' --inference_dtype int8 --inference_gpu_cost 0.60 --input_tok 500 --output_tok 100 --inference_reqs 1e9
        ```
    - Output:
    ```python
        Model Size (params):	 3.000e+10
        Training Data (tok):	 1.556e+12
        Final Training loss:	 1.958253360475841
        Train Cost ($):		 3.146e+05
        Train FLOPs:		 2.801e+23
        Chinchilla Style:	 True
        Coefficients:		 alpha = 0.336        beta = 0.283        A = 406.4        B = 410.7        E = 1.69
        Input tokens per request: 500
        Output tokens per request: 100
        Inference requests: 1.000e+09
        Total (Train + Inference) FLOPs 3.161e+23
        Total (Train + Inference) Cost ($): 6.378e+05
    ```

# Advanced Features
By default, all scripts in this repo assume the original coefficients from the Chinchilla paper: $\alpha=0.336, \beta=0.283, A=406.4, B=410.7, E=1.69$. These coefficients were determined by a curve-fitting procedure, where the authors fit the Chinchilla equation to data from ~400 empirical training runs. These training runs were all conducted on the same dataset, which is a non-public dataset containing mostly internet text and code.

If your data is very similar to this (or you are just playing around with scaling laws), then you can use the scripts in this repo as-is, with these original coefficients. However, if you have a different training dataset (and most people do), you should use coefficients which are fit to your dataset for better scaling predictions. 

To fit coefficients for your data:
1. Collect empirical training by training many models across many different sizes and training data lengths, collecting final loss values (typically smoothed over the last few batches). This may be expensive, depending on the scales you wish to test.
2. Fit the coefficients of the Chinchilla formula to this empirical data. We provide our code for this fitting procedure in `huber_optimize.py`. This code is an implementation of the original algorithm from the Chinchilla paper.
    - Example usage: `python huber_optimize.py --data trainingresults.csv`

Once you have your own coefficients, so that the scripts apply better to your dataset, you can pass in: 
`--alpha`, `--beta`, `--A`, `--B`, and `--E` 
to any of the scripts and we will use your coeffients in all our calculations, rather than the original Chinchilla coefficients.

In practice, this means the total FLOPs and total costs we calculate (and therefore the model size and training data length we suggest) change based on the coefficients you provide. It is important that your coefficients be reasonably accurate for your dataset, or else our recommendations ("You should train for a model of 'p' parameters on 'x' tokens to achieve 'l' loss") may not align with your actual results. 

Typically, labs will train a few small models on their dataset and to determine their coefficients, and then extrapolate these scaling curves out to a much larger model. This results in reasonably accurate scaling predictions without a large expense (relative to the training cost of the larger model).

# Data
In `trainingresults.csv`, we provide the smoothed final loss values and downstream evalution results from our paper's 47 training runs of MPT-style models across various model sizes (150M - 6B) and training data lengths (10--10,000 tokens/parameter). These models were used in our paper to validate our inference-adjusted scaling laws.

# Citation
If you use this code, please cite our paper:
```
@inproceedings{
    beyondchinchilla,
    title={Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws},
    author={Nikhil Sardana and Jacob Portes and Sasha Doubov and Jonathan Frankle},
    booktitle={Fourty-first International Conference on Machine Learning},
    year={2024},
    url={https://arxiv.org/abs/2401.00448}
}
```

# License
This code is licensed under the Apache-2.0 license.
