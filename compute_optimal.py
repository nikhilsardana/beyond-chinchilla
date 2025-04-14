from scipy.optimize import newton
import argparse
from model_calculator import Model, ChinchillaConstants, total_flops
from utils import sf, params_from_loss_and_tokens, add_coefficients_to_parser


def calculate_compute_optimal_model(model: Model, T) -> Model:
    """
    For a given Model (model) and inference tokens (T), this function
    computes the optimal model that achieves the same quality (model.loss),
    where optimal = minimum lifetime compute over training + inference.
    Using this method, one can easily compare the total compute for training + inference for
    the input and returned Models to find the overall compute savings.
    Arguments:
        model: Model object
        T = Number of inference tokens
    Returns:
        A new Model object, representing the optimal model accounting for training
        and inference compute.

    """
    D_opt = newton(
        model.compute_optimal_train_tokens,
        1e8,
        tol=1e-5,
        maxiter=100,
        rtol=1e-8,
        args=(T, model.loss),
        disp=True,
    )
    approximation_error = model.compute_optimal_train_tokens(D_opt, T, model.loss)
    assert (
        approximation_error < 1e-10
    ), "Approximation failure: Could not approximate training tokens for given T and Q."
    N_opt = params_from_loss_and_tokens(model.constants, model.loss, D_opt)
    compute_optimal_model = Model(
        chinchilla_style=False,
        constants=model.constants,
        parameters=N_opt,
        training_tokens=D_opt,
    )
    return compute_optimal_model


def run(argv=None):
    """
    Understand the compute-optimal way (i.e. minimum cost) to train a model of `args.loss` quality and run
    `inference_tokens` of inference.
    """
    parser = argparse.ArgumentParser(
        description="""
                                     Calculate the compute-optimal (minimum FLOPs) way over a model's lifetime to train and run inference on a model of a certain quality
                                     and given inference demand. You provide the quality of model (--loss) you wish to train, and how much
                                     inference demand you expect (--inference_tokens), and this script will tell you (based on the Chinchilla scaling laws)
                                     how large to make your model and how long to train it. 
                                     Use chinchilla.py to calculate loss values for various model sizes, data lengths, or compute budgets.
                                     """
    )

    # Arguments
    parser.add_argument(
        "--loss",
        type=float,
        required=True,
        help="Loss (quality) of model you wish to train.",
    )
    parser.add_argument(
        "--inference_tokens",
        type=float,
        required=False,
        default=2e12,
        help="Total number of lifetime inference tokens (Input+output) across all requests.",
    )

    parser = add_coefficients_to_parser(parser)

    args = parser.parse_args(argv)

    chinchilla_model = Model(
        True,
        ChinchillaConstants(
            alpha=args.alpha, beta=args.beta, A=args.A, B=args.B, E=args.E
        ),
        loss=args.loss,
    )
    inference_tokens = args.inference_tokens

    optimal_model = calculate_compute_optimal_model(chinchilla_model, inference_tokens)

    total_flops_opt = total_flops(
        optimal_model.params, optimal_model.train_tokens, inference_tokens
    )
    total_flops_chinchilla = total_flops(
        chinchilla_model.params, chinchilla_model.train_tokens, inference_tokens
    )
    print("----------------------------------------")
    print("*** Baseline: Chinchilla-Style Model ***")
    print("----------------------------------------")
    print(chinchilla_model)
    print("Total (Train + Inference) FLOPs:", sf(total_flops_chinchilla))
    print("----------------------------------------")
    print("----------------------------------------")
    print("*** (Lifetime) Compute-Optimal Model ***")
    print("----------------------------------------")
    print(optimal_model)
    print("Total (Train + Inference) FLOPs:", sf(total_flops_opt))
    print("----------------------------------------")
    print("*** Conclusion ***")
    print(
        f"The Compute-Optimal model should have {round(100 * optimal_model.params/chinchilla_model.params, 2)}% of the parameters of an equal-quality Chinchilla-style model."
    )
    print(
        f"The Compute-Optimal model should be trained on {round(100 * optimal_model.train_tokens/chinchilla_model.train_tokens, 2)}% of the tokens of an equal-quality Chinchilla-style model."
    )
    print(
        f"The Compute-Optimal model should cost {round(100 * total_flops_opt/total_flops_chinchilla, 2)}% of an equal-quality Chinchilla-style model."
    )


if __name__ == "__main__":
    run()
