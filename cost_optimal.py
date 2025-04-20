from scipy.optimize import newton
from compute_optimal import sf, total_flops
from model_calculator import ChinchillaConstants
import argparse
from utils import params_from_loss_and_tokens, add_coefficients_to_parser
from cost_calculator import CostModel, cost_per_flop, total_cost, gpu_cost_per_hour


def calculate_cost_optimal_model(cost_model: CostModel, T_in, T_out) -> CostModel:
    """
    For a given model and inference tokens, this function
    calculates the cost-optimal model that achieves the same quality (cost_model.loss),
    accounting for the real-world costs of training + inference.
    Using this method, one can easily compare the total compute for training + inference for
    the input and returned CostModels to find the overall cost difference.
    Arguments:
        cost_model: CostModel object
        T_in = Number of input tokens
        T_out = Number of output tokens
    Returns:
        A new CostModel object, representing the optimal model accounting for the cost
        of inference.
    """
    D_opt = newton(
        cost_model.cost_optimal_train_tokens,
        1e8,
        tol=1e-5,
        maxiter=100,
        rtol=1e-8,
        args=(T_in, T_out, cost_model.loss),
        disp=True,
    )
    approximation_error = cost_model.cost_optimal_train_tokens(
        D_opt, T_in, T_out, cost_model.loss
    )
    assert (
        approximation_error < 1e-10
    ), f"Approximation failure: Could not approximate training tokens for given Inference demand ({T_in}, {T_out}) and Loss ({cost_model.loss})."
    N_opt = params_from_loss_and_tokens(cost_model.constants, cost_model.loss, D_opt)
    cost_optimal_model = CostModel(
        cost_model.C_tr,
        cost_model.C_i,
        cost_model.U_tr,
        cost_model.U_in,
        cost_model.U_out,
        chinchilla_style=False,
        constants=cost_model.constants,
        parameters=N_opt,
        training_tokens=D_opt,
    )
    return cost_optimal_model


if __name__ == "__main__":
    """
    Understand the cost-optimal way (i.e. minimum cost) to train a model of `args.loss` quality and run
    `inference_reqs` of inference.
    """
    default_constants = ChinchillaConstants()
    parser = argparse.ArgumentParser(
        description="""
            Understand the lifetime cost-optimal way to train a model of `args.loss` quality and run `inference_reqs` of inference.
            Use the arguments below to set your expected inference demand, in terms of:
                - average input tokens per request
                - average output tokens per request
                - number of total liftetime inference requests.
            You can also set your hardware configuration (GPU type, inference and training GPU cost per FLOP, MFU for training,
            prefill, and generation).
            This script also returns the total cost to train a model ofo the same quality (and run inference requests) for a model
            trained to its Chinchilla point. It returns the ratios of the "cost-optimal" method and the Chinchilla-"Optimal" method.
            By default, this script is configured to show you the cost-optimal way to train
            a Chinchilla-70B quality model, and then deploy it for 2e12 inference tokens, assuming:
            (a) You are training on H100s @ $2.00/hr in BF16 at 50% MFU
            (b) You are running inference on H100s @ $2.00/hr in FP8 at 50% MFU for prefill, and 1% MFU for generation
            (c) Your requests have an average of 70 input and 215 output tokens.
            """
    )

    parser.add_argument(
        "--loss",
        type=float,
        required=True,
        help="Loss (quality) of model you wish to train.",
    )

    # Hardware configuration
    parser.add_argument(
        "--train_gpu_cost",
        type=float,
        required=False,
        help="Train GPU cost per hour, in dollars. Optional, we default to typical hourly GPU rates if not provided.",
    )
    parser.add_argument(
        "--inference_gpu_cost",
        type=float,
        required=False,
        help="Train GPU cost per hour, in dollars. Optional, we default to typical hourly GPU rates if not provided.",
    )

    parser.add_argument(
        "--train_gpu_type",
        type=str,
        required=False,
        default="H100",
        choices=["A100_40", "A100_80", "H100"],
        help="Train GPU cost per hour, in dollars. Defaults to H100.",
    )
    parser.add_argument(
        "--inference_gpu_type",
        type=str,
        required=False,
        default="H100",
        choices=["A100_40", "A100_80", "H100"],
        help="Inference GPU cost per hour, in dollars. Defaults to H100.",
    )

    parser.add_argument(
        "--train_dtype",
        type=str,
        required=False,
        default="bf16",
        choices=["bf16", "fp8"],
        help="Train GPU dtype. Defaults to bf16.",
    )
    parser.add_argument(
        "--inference_dtype",
        type=str,
        required=False,
        default="fp8",
        choices=["bf16", "fp8", "int8"],
        help="Inference GPU dtype. Defaults to fp8.",
    )

    parser.add_argument(
        "--train_mfu",
        type=float,
        required=False,
        default=0.5,
        help="Training model flops utilization. Defaults to 0.5 (50%).",
    )
    parser.add_argument(
        "--prefill_mfu",
        type=float,
        required=False,
        default=0.5,
        help="Inference prefill model flops utilization. Defaults to 0.5 (50%).",
    )
    parser.add_argument(
        "--generation_mfu",
        type=float,
        required=False,
        default=0.01,
        help="Inference generation model flops utilization. Defaults to 0.01 (1%).",
    )

    # Inference request structure
    parser.add_argument(
        "--input_tok",
        type=int,
        required=False,
        default=70,
        help="Average number of input tokens per request. Defaults to average (70) from LMSys data.",
    )
    parser.add_argument(
        "--output_tok",
        type=int,
        required=False,
        default=215,
        help="Average number of output tokens per request. Defaults to average (215) from LMSys data.",
    )
    parser.add_argument(
        "--inference_reqs",
        type=float,
        required=False,
        default=10000000000,
        help="Total number of inference requests across the model's lifetime. Defaults to 10 billion",
    )

    # Chinchilla Coefficients
    parser = add_coefficients_to_parser(parser)

    args = parser.parse_args()

    custom_constants = ChinchillaConstants(
        alpha=args.alpha, beta=args.beta, A=args.A, B=args.B, E=args.E
    )

    cost_train = cost_per_flop(
        args.train_gpu_type, args.train_dtype, args.train_gpu_cost
    )
    cost_inference = cost_per_flop(
        args.inference_gpu_type, args.inference_dtype, args.inference_gpu_cost
    )

    chinchilla_model = CostModel(
        cost_train,
        cost_inference,
        args.train_mfu,
        args.prefill_mfu,
        args.generation_mfu,
        chinchilla_style=True,
        constants=custom_constants,
        loss=args.loss,
    )

    input_tok_per_req = args.input_tok
    output_tok_per_req = args.output_tok
    total_input_tokens = args.inference_reqs * input_tok_per_req
    total_output_tokens = args.inference_reqs * output_tok_per_req

    assert total_input_tokens > 0, "Number of input tokens must be greater than zero."

    # Calculate cost-optimal model that matches original model's quality (loss)
    cost_optimal_model = calculate_cost_optimal_model(
        chinchilla_model, total_input_tokens, total_output_tokens
    )

    # Calculate total (train + inference) flops for each model, original and optimal
    total_flops_chinchilla = total_flops(
        chinchilla_model.params,
        chinchilla_model.train_tokens,
        total_input_tokens + total_output_tokens,
    )
    total_flops_optimal = total_flops(
        cost_optimal_model.params,
        cost_optimal_model.train_tokens,
        total_input_tokens + total_output_tokens,
    )

    # Calculate total cost for each model, original and optimal
    total_cost_chinchilla = total_cost(
        chinchilla_model, total_input_tokens, total_output_tokens
    )
    total_cost_optimal = total_cost(
        cost_optimal_model, total_input_tokens, total_output_tokens
    )

    print("----------------------------------------")
    print("*** Configuration ***")
    print("----------------------------------------")
    print(
        "Inference requests:\t\t",
        sf(
            (total_input_tokens + total_output_tokens)
            / (input_tok_per_req + output_tok_per_req)
        ),
    )
    print("Avg. Input Tokens per Request:\t", input_tok_per_req)
    print("Avg. Output Tokens per Request:\t", output_tok_per_req)
    print("Training GPU Type:\t\t", args.train_gpu_type)
    print("Training GPU dtype:\t\t", args.train_dtype)
    print(
        "Training GPU Cost per hour:\t",
        (
            args.train_gpu_cost
            if args.train_gpu_cost is not None
            else gpu_cost_per_hour(args.train_gpu_type)
        ),
    )
    print("Inference GPU Type:\t\t", args.inference_gpu_type)
    print("Inference GPU dtype:\t\t", args.inference_dtype)
    print(
        "Inference GPU Cost per hour:\t",
        (
            args.inference_gpu_cost
            if args.inference_gpu_cost is not None
            else gpu_cost_per_hour(args.inference_gpu_type)
        ),
    )
    print("----------------------------------------")
    print("*** Baseline: Chinchilla-Style Model ***")
    print("----------------------------------------")
    print(chinchilla_model)
    print("Total (Train + Inference) FLOPs:", sf(total_flops_chinchilla))
    print("Total (Train + Inference) Cost ($):", total_cost_chinchilla)
    print("----------------------------------------")
    print("----------------------------------------")
    print("*** Cost-Optimal Model ***")
    print("----------------------------------------")
    print(cost_optimal_model)
    print("Total (Train + Inference) FLOPs:", sf(total_flops_optimal))
    print("Total (Train + Inference) Cost ($):", total_cost_optimal)
    print("----------------------------------------")
    print("*** Conclusion ***")
    print(
        f"The Cost-Optimal model should have {round(100 * cost_optimal_model.params/chinchilla_model.params, 2)}% of the parameters of an equal-quality Chinchilla-style model."
    )
    print(
        f"The Cost-Optimal model should be trained on {round(100 * cost_optimal_model.train_tokens/chinchilla_model.train_tokens, 2)}% of the tokens of an equal-quality Chinchilla-style model."
    )
    print(
        f"The Cost-Optimal model should cost {round(100 * total_cost_optimal/total_cost_chinchilla, 2)}% of an equal-quality Chinchilla-style model."
    )
