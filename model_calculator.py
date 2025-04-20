import argparse
from utils import (
    ChinchillaConstants,
    FLOPMode,
    FLOPs,
    chinchilla_params_from_compute,
    chinchilla_tokens_from_compute,
    loss_from_params_and_data,
    chinchilla_compute_from_tokens,
    chinchilla_compute_from_params,
    chinchilla_params_from_tokens,
    chinchilla_tokens_from_params,
    params_from_compute_and_tokens,
    tokens_from_compute_and_params,
    chinchilla_params_from_loss,
    chinchilla_tokens_from_loss,
    sf,
    add_coefficients_to_parser,
)


class Model:
    """
    Class that keeps track of parameters, training tokens, loss, and compute
    for a model.

    Computes missing values based on model characteristics provided (e.g. training compute from params & tokens).
    """

    def __init__(
        self,
        chinchilla_style: bool,
        constants: ChinchillaConstants,
        training_compute=None,
        parameters=None,
        training_tokens=None,
        loss=None,
    ):

        # Set chinchilla_style to True if model satisfies Chinchilla Ratios
        self.chinchilla_style = chinchilla_style
        self.constants = constants

        if self.chinchilla_style:
            if training_compute != None:
                self.train_flops = training_compute
                self.params = chinchilla_params_from_compute(
                    self.constants, self.train_flops
                )
                self.train_tokens = chinchilla_tokens_from_compute(
                    self.constants, self.train_flops
                )
                self.loss = loss_from_params_and_data(
                    self.constants, self.params, self.train_tokens
                )
            elif parameters != None:
                self.params = parameters
                self.train_flops = chinchilla_compute_from_params(
                    self.constants, self.params
                )
                self.train_tokens = chinchilla_tokens_from_params(
                    self.constants, self.params
                )
                self.loss = loss_from_params_and_data(
                    self.constants, self.params, self.train_tokens
                )
            elif training_tokens != None:
                self.train_tokens = training_tokens
                self.train_flops = chinchilla_compute_from_tokens(
                    self.constants, self.train_tokens
                )
                self.params = chinchilla_params_from_tokens(
                    self.constants, self.train_tokens
                )
                self.loss = loss_from_params_and_data(
                    self.constants, self.params, self.train_tokens
                )
            elif loss != None:
                self.loss = loss
                self.params = chinchilla_params_from_loss(self.constants, loss)
                self.train_tokens = chinchilla_tokens_from_loss(self.constants, loss)
                self.train_flops = chinchilla_compute_from_params(
                    self.constants, self.params
                )
            else:
                raise ValueError(
                    "Must provide either Compute budget, number of model parameters, number of training tokens, or loss to fully define a chinchilla-style model."
                )

        else:
            if (
                (training_compute == None and parameters == None)
                or (training_compute == None and training_tokens == None)
                or (parameters == None and training_tokens == None)
            ):
                raise ValueError(
                    "For non-chinchilla-style models, must provide two of Compute Budget, model parameters, or training tokens to fully define a model."
                )

            self.train_flops = (
                training_compute
                if training_compute != None
                else FLOPs(parameters, training_tokens, FLOPMode.TRAINING)
            )
            self.params = (
                parameters
                if parameters != None
                else params_from_compute_and_tokens(training_compute, training_tokens)
            )
            self.train_tokens = (
                training_tokens
                if training_tokens
                else tokens_from_compute_and_params(training_compute, parameters)
            )
            self.loss = loss_from_params_and_data(
                self.constants, self.params, self.train_tokens
            )

    def __repr__(self):
        x = f"Model Size (params):\t {sf(self.params)}"
        x += f"\nTraining Data (tok):\t {sf(self.train_tokens)}"
        x += f"\nFinal Training loss:\t {self.loss}"
        x += f"\nTrain FLOPs:\t\t {sf(self.train_flops)}"
        x += f"\nChinchilla Style:\t {self.chinchilla_style}"
        x += f"\nCoefficients:\t\t {self.constants}"
        return x

    def compute_optimal_train_tokens(self, x, T, L):
        """
        Find the optimal number of tokens (D) to train on if you are going to create a model
        of quality L (pre-training loss L) and run inference for T tokens over its lifetime.
        This method is used by a solver (e.g. Newton's method) to find root (optimal D).
        We cannot use a formula similar to utils.tokens_from_loss() because there is no analytical formula when T > 0.
        """
        coeff_1 = (
            (self.constants.beta * self.constants.B) / self.constants.alpha
        ) + self.constants.B
        coeff_2 = (T * self.constants.beta * self.constants.B) / (
            3 * self.constants.alpha
        )
        loss_diff = self.constants.E - L
        return (
            coeff_1 * x ** (-1 * self.constants.beta)
            + coeff_2 * x ** ((-1 * self.constants.beta) - 1)
            + loss_diff
        )


def total_flops(model_size, training_tokens, inference_tokens):
    """
    Returns total number of FLOPs to train a model of `model_size` params on
    `training_tokens` tokens, and then run inference on `inference_tokens`.
    """
    return FLOPs(model_size, training_tokens, FLOPMode.TRAINING) + FLOPs(
        model_size, inference_tokens, FLOPMode.INFERENCE
    )


def run(argv=None):
    """
    Calculate the properties of a model given some of its attributes.
    """
    parser = argparse.ArgumentParser(
        description="""
                                    Calculate the properties of a model given some of its attributes.
                                    If your model is Chinchilla-optimal (--chinchilla), provide one of {loss, model, data, compute}.
                                    Otherwise, provide two of Compute Budget, model parameters, and training tokens.
                                    If you provide the inference tokens you wish to run the model on,
                                    this script also computes the total FLOPs to run the model across training + inference.
                                     """
    )
    # Provide some details of the model configuration
    parser.add_argument(
        "--loss", type=float, required=False, help="Loss of model you wish to train."
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
        help="If the model you would originally like to train obeys the Chinchilla ratios, you can set this flag to true and provide only one piece of information about the model (loss, model size, or training data). Defaults to False.",
    )

    # Inference demand
    parser.add_argument(
        "--inference_tokens",
        type=float,
        required=False,
        default=0,
        help="Optional. Total number of lifetime inference tokens (Input+output) across all requests. Defaults to zero if not provided.",
    )

    # Chinchilla coefficients
    parser = add_coefficients_to_parser(parser)

    args = parser.parse_args(argv)
    constants = ChinchillaConstants(args.alpha, args.beta, args.A, args.B, args.E)
    model = Model(
        args.chinchilla,
        constants,
        training_compute=args.compute,
        parameters=args.model,
        training_tokens=args.data,
        loss=args.loss,
    )
    print(model)

    if args.inference_tokens:
        combined_flops = total_flops(
            model.params, model.train_tokens, args.inference_tokens
        )
        print("Inference Tokens:", args.inference_tokens)
        print("Total (Train + Inference) FLOPs", sf(combined_flops))


if __name__ == "__main__":
    run()
