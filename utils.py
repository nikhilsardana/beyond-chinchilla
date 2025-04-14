"""
Helper methods, classes, and data structures.
"""

from enum import IntEnum
from argparse import ArgumentParser


class ChinchillaConstants:
    """
    Class for Chinchilla coefficients/constants
    Defaults to constants from Chinchilla paper (law 3)
    In practice, these need to be adjusted to your dataset.
    See huber_loss.py for how to find constants after you have collected empirical data.
    """

    def __init__(self, alpha=0.336, beta=0.283, A=406.4, B=410.7, E=1.69):
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.B = B
        self.E = E

    def __repr__(self):
        return f"alpha = {self.alpha}\
        beta = {self.beta}\
        A = {self.A}\
        B = {self.B}\
        E = {self.E}"


# Multipliers for FLOP Calculations
class FLOPMode(IntEnum):
    TRAINING = 6.0
    INFERENCE = 2.0


def FLOPs(params, tokens, mode: FLOPMode):
    """
    FLOPs(N, D) = 6ND for training,
                  2ND for inference
    """
    return params * tokens * mode


def sf(num):
    """
    Nice scientific formatting.
    """
    return "{:.3e}".format(num)


def chinchilla_tokens_from_compute(cc: ChinchillaConstants, C):
    """
    Compute D (pre-training tokens) from compute budget (C)
    """
    b = (cc.alpha) / (cc.alpha + cc.beta)
    first_term = ((cc.beta * cc.B) / (cc.alpha * cc.A)) ** (1 / (cc.alpha + cc.beta))
    second_term = (C / 6) ** b
    return first_term * second_term


def chinchilla_params_from_compute(cc: ChinchillaConstants, C):
    """
    Compute N (params) from compute budget (C)
    """
    a = (cc.beta) / (cc.alpha + cc.beta)
    first_term = ((cc.alpha * cc.A) / (cc.beta * cc.B)) ** (1 / (cc.alpha + cc.beta))
    second_term = (C / 6) ** a
    return first_term * second_term


def chinchilla_compute_from_params(cc: ChinchillaConstants, N):
    """
    Compute training FLOPs from model parameters (N)
    """
    G = ((cc.alpha * cc.A) / (cc.beta * cc.B)) ** (1 / (cc.alpha + cc.beta))
    a = (cc.beta) / (cc.alpha + cc.beta)
    return 6 * ((N / G) ** (1 / a))


def chinchilla_compute_from_tokens(cc: ChinchillaConstants, D):
    """
    Compute training FLOPs from training tokens (D)
    """
    G = ((cc.alpha * cc.A) / (cc.beta * cc.B)) ** (1 / (cc.alpha + cc.beta))
    b = (cc.alpha) / (cc.alpha + cc.beta)
    return (6 * D * G) ** (1 / b)


def chinchilla_tokens_from_params(cc: ChinchillaConstants, N):
    """
    Compute the number of Chinchilla-optimal training tokens
    for a given model size (N parameters).
    """
    return chinchilla_tokens_from_compute(cc, chinchilla_compute_from_params(cc, N))


def chinchilla_params_from_tokens(cc: ChinchillaConstants, D):
    """
    Compute the number of Chinchilla-optimal parameters
    for a given token duration (D tokens).
    """
    return chinchilla_params_from_compute(cc, chinchilla_compute_from_tokens(cc, D))


def tokens_from_compute_and_params(C, N):
    """
    Compute training tokens from training FLOPs (C) and model parameters (N)
    """
    return C / (N * 6)


def params_from_compute_and_tokens(C, D):
    """
    Compute model parameters from training FLOPs (C) and training tokens (D)
    """
    return C / (D * 6)


### Training only, no inference
def tokens_from_loss_and_params(cc: ChinchillaConstants, L, N):
    """
    Compute minimum number of tokens a model of N params must be trained on
    to reach L model loss (quality).
    """
    partial_result = cc.B / (L - cc.E - cc.A / (N**cc.alpha))
    return partial_result ** (1 / cc.beta)


def params_from_loss_and_tokens(cc: ChinchillaConstants, L, D):
    """
    Compute minimum number of model params we need to reach L model loss
    after D training tokens.
    """
    partial_result = cc.A / (L - cc.E - cc.B / (D**cc.beta))
    return partial_result ** (1 / cc.alpha)


def chinchilla_tokens_from_loss(cc: ChinchillaConstants, L):
    """
    Compute Chinchilla-optimal D (pre-training tokens) from pre-training loss
    """
    numerator = cc.B * ((cc.beta / cc.alpha) + 1)
    denominator = L - cc.E
    return (numerator / denominator) ** (1 / cc.beta)


def chinchilla_params_from_loss(cc: ChinchillaConstants, L):
    """
    Compute Chinchilla-optimal N (params) from pre-training loss
    """
    numerator = cc.A * ((cc.alpha / cc.beta) + 1)
    denominator = L - cc.E
    return (numerator / denominator) ** (1 / cc.alpha)


def loss_from_params_and_data(cc: ChinchillaConstants, N, D):
    """
    Compute model loss from parameters and tokens.
    Formula from Chinchilla paper.
    """
    return cc.E + cc.A / (N**cc.alpha) + cc.B / (D**cc.beta)


def add_coefficients_to_parser(parser: ArgumentParser):
    default_constants = ChinchillaConstants()
    parser.add_argument(
        "--A",
        type=float,
        required=False,
        default=default_constants.A,
        help="Optional. 'A' coefficient in Chinchilla formula. Defaults to coefficient from original paper (406.4). Pass in value if you have fit scaling laws to your own dataset.",
    )
    parser.add_argument(
        "--B",
        type=float,
        required=False,
        default=default_constants.B,
        help="Optional. 'B' coefficient in Chinchilla formula. Defaults to coefficient from original paper (410.7). Pass in value if you have fit scaling laws to your own dataset.",
    )
    parser.add_argument(
        "--E",
        type=float,
        required=False,
        default=default_constants.E,
        help="Optional. 'E' coefficient in Chinchilla formula. Defaults to coefficient from original paper (1.69). Pass in value if you have fit scaling laws to your own dataset.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=default_constants.alpha,
        help="Optional. 'alpha' coefficient in Chinchilla formula. Defaults to coefficient from original paper (0.336). Pass in value if you have fit scaling laws to your own dataset.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=False,
        default=default_constants.beta,
        help="Optional. 'beta' coefficient in Chinchilla formula. Defaults to coefficient from original paper (0.283). Pass in value if you have fit scaling laws to your own dataset.",
    )
    return parser
