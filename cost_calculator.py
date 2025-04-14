from compute_optimal import sf
from model_calculator import Model, ChinchillaConstants, total_flops
import argparse
from utils import FLOPMode, add_coefficients_to_parser

A100_INT8_OPS = 6.24e14
A100_FP16_OPS = 3.12e14

H100_FP8_OPS = 1.978e15
H100_FP16_OPS = 9.89e14

A100_80_DEFAULT_COST = 1.50  # cost per gpu per hour
A100_40_DEFAULT_COST = 1.10  # cost per gpu per hour
H100_DEFAULT_COST = 2.00  # cost per gpu per hour


def gpu_cost_per_hour(gpu_type):
    if gpu_type == "A100_40":
        return A100_40_DEFAULT_COST
    elif gpu_type == "A100_80":
        return A100_80_DEFAULT_COST
    elif gpu_type == "H100":
        return H100_DEFAULT_COST
    else:
        raise ValueError(f"Does not support {gpu_type} gpu type")


def cost_per_flop(gpu_type, dtype, provided_cost_per_hour=None):
    flops_per_second = 0
    if gpu_type == "A100_40" or gpu_type == "A100_80":
        if dtype == "int8":
            flops_per_second = A100_INT8_OPS
        elif dtype == "bf16" or dtype == "fp16":
            flops_per_second = A100_FP16_OPS
        else:
            raise ValueError(f"A100 does not support dtype {dtype}")
    else:  # H100
        if dtype == "fp8" or dtype == "int8":
            flops_per_second = H100_FP8_OPS
        elif dtype == "bf16" or dtype == "fp16":
            flops_per_second = H100_FP16_OPS
        else:
            raise ValueError(f"H100 does not support dtype {dtype}")

    if provided_cost_per_hour is not None:
        cost_per_hour = provided_cost_per_hour
    else:
        cost_per_hour = gpu_cost_per_hour(gpu_type)
    return cost_per_hour / (60 * 60 * flops_per_second)


def cost(params, tokens, mode: FLOPMode, cost_per_flop, mfu):
    return (params * tokens * mode * cost_per_flop) / mfu


class CostModel(Model):
    """Extension of Model class that takes in extra parameters
    for the cost of training and inference. This allows us to keep track
    of all the variables we need to minimize real-world cost, rather than FLOPS.
    """

    def __init__(
        self,
        training_cost,  # cost per training FLOP
        inference_cost,  # cost per inference FLOP
        training_mfu,
        prefill_mfu,
        generation_mfu,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.C_tr = training_cost
        self.C_i = inference_cost
        self.U_tr = training_mfu
        self.U_in = prefill_mfu
        self.U_out = generation_mfu

    def cost_optimal_train_tokens(self, x, T_in, T_out, L):
        """
        Find the cost-optimal number of pre-training tokens (D), assuming we want to train a
        model up to loss L, and then run inference with T_in total input (prefill) tokens
        and T_out total output (generation) tokens over the model lifetime.
        This method takes into account the compute utilization and cost multipliers provided
        to the class.
        We call this method using a solver (e.g. Newton's method) to find root (optimal D) since no analytic solutions exist.
        Arguments:
            T_in = Number of input tokens
            T_out = Number of output tokens
            L = loss (model quality)
        """
        mult_i = self.C_i * (T_in / self.U_in + T_out / self.U_out)
        mult_tr = self.C_tr / self.U_tr
        coeff_1 = (
            (self.constants.beta * self.constants.B) / self.constants.alpha
        ) + self.constants.B
        coeff_2 = (self.constants.beta * self.constants.B * mult_i) / (
            3 * self.constants.alpha * mult_tr
        )
        diff = self.constants.E - L
        return (
            coeff_1 * x ** (-1 * self.constants.beta)
            + coeff_2 * x ** ((-1 * self.constants.beta) - 1)
            + diff
        )

    def __repr__(self):
        x = f"Model Size (params):\t {sf(self.params)}"
        x += f"\nTraining Data (tok):\t {sf(self.train_tokens)}"
        x += f"\nFinal Training loss:\t {self.loss}"
        x += f"\nTrain Cost ($):\t\t {sf(cost(self.params, self.train_tokens, FLOPMode.TRAINING, self.C_tr, self.U_tr))}"
        x += f"\nTrain FLOPs:\t\t {sf(self.train_flops)}"
        x += f"\nChinchilla Style:\t {self.chinchilla_style}"
        x += f"\nCoefficients:\t\t {self.constants}"
        return x


def total_cost(model: CostModel, T_in, T_out):
    # Train + Prefill + Decode cost
    return (
        cost(
            model.params, model.train_tokens, FLOPMode.TRAINING, model.C_tr, model.U_tr
        )
        + cost(model.params, T_in, FLOPMode.INFERENCE, model.C_i, model.U_in)
        + cost(model.params, T_out, FLOPMode.INFERENCE, model.C_i, model.U_out)
    )


if __name__ == "__main__":
    """
    Calculate the properties of a model given some of its attributes and real-world costs.
    """
    default_constants = ChinchillaConstants()
    parser = argparse.ArgumentParser(
        description="""
                                    Calculate the properties of a model given some of its attributes. 
                                    If your model is Chinchilla-optimal (--chinchilla), provide one of {loss, model, data, compute}. 
                                    Otherwise, provide two of Compute Budget, model parameters, and training tokens.
                                    If you provide the number of inference requests (and input and output tokens per request) you wish to run the model on,
                                    this script computes the total cost to run the model across training + inference.
                                    Additionally, provide the training (and inference, if applicable) hardware configurations to estimate 
                                    the overall cost.
                                     """
    )

    # Provide some details of the model configuration
    parser.add_argument(
        "--loss",
        type=float,
        required=False,
        help="Loss (quality) of model you wish to train.",
    )
    parser.add_argument(
        "--model", type=float, required=False, help="Size of model you wish to train."
    )
    parser.add_argument(
        "--data",
        type=float,
        required=False,
        help="Training data length (tokens) of model you wish to train.",
    )
    parser.add_argument(
        "--compute",
        type=float,
        required=False,
        help="Training compute (FLOPS) of model you wish to train.",
    )
    parser.add_argument(
        "--chinchilla",
        type=bool,
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="If the model you would originally like to train obeys the Chinchilla-style ratios, you can set this flag to true and provide only one piece of information about the model (loss, model size, or training data).",
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

    # Infereence Request details
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
        default=0,
        help="Optional. Total number of inference requests across the model's lifetime. Defaults to zero if not provided.",
    )

    # Chinchilla coefficients
    parser = add_coefficients_to_parser(parser)

    # Parse the arguments
    args = parser.parse_args()

    custom_constants = ChinchillaConstants(
        alpha=args.alpha, beta=args.beta, A=args.A, B=args.B, E=args.E
    )

    cost_train = cost_per_flop(args.train_gpu_type, args.train_dtype)
    cost_inference = cost_per_flop(args.inference_gpu_type, args.inference_dtype)

    model = CostModel(
        cost_train,
        cost_inference,
        args.train_mfu,
        args.prefill_mfu,
        args.generation_mfu,
        chinchilla_style=args.chinchilla,
        constants=custom_constants,
        loss=args.loss,
        parameters=args.model,
        training_tokens=args.data,
        training_compute=args.compute,
    )

    print(model)

    if args.inference_reqs > 0:
        total_input_tokens = args.inference_reqs * args.input_tok
        total_output_tokens = args.inference_reqs * args.output_tok

        # Calculate total flops across training and inference
        combined_flops = total_flops(
            model.params, model.train_tokens, total_input_tokens + total_output_tokens
        )

        # Calculate total cost across training and inference
        combined_cost = total_cost(model, total_input_tokens, total_output_tokens)

        print("Input tokens per request:", args.input_tok)
        print("Output tokens per request:", args.output_tok)
        print(
            "Inference requests:",
            sf(
                (total_input_tokens + total_output_tokens)
                / (args.input_tok + args.output_tok)
            ),
        )
        print("Total (Train + Inference) FLOPs", sf(combined_flops))
        print("Total (Train + Inference) Cost ($):", sf(combined_cost))
