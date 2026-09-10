# -*- coding: utf-8 -*-
"""
libRC: a reservoir computing research framework.

##  Glossary ##
    # Reservoir Parameters
    N           -   dimension of reservoir / number of nodes in reservoir
    activ       -   activation function, defaults to np.tanh
    bias        -   adds a bias output node to the W fit (advised AGAINST using this)

    # Connection Matrix and Parameters
    A           -   adjacency/connection matrix with spectral radius rho
    rho         -   spectral radius
    density     -   percentage of non-zero nodes in A
    degree      -   number of connections per node in A

    # Input Matrix and Parameters
    B           -   input weight matrix (unnormalized)
    sigma       -   scaling/normalization factor

    # Output Matrix and Parameters
    mask        -   to specify which index of the output should be used as input
    D           -   dimension of the input data
    W           -   output weight matrix
    alpha       -   ridge parameter for least squares fit

    # Continuous Time Parameters (diffRC only)
    ds          -   reservoir time step (constant or vector)
    integrator  -   time stepping method

    # Listening Time Series
    M           -   number of steps of the incoming data
    r           -   reservoir state / internal representation (N x M)
    y_in        -   (external) input data (D x M)
    y_target    -   (external) target/output data
    y_est       -   reservoir estimate when fitting r to y (D x M)

    # Echoing Time Series
    M_echo      -   number of steps to echo (value not stored explicitly in object)
    r_echo      -   state of the autonomous reservoir
    y_echo      -   prediction of the original dataset using the autonomous reservoir

    # Inference Time Series
    M_infer     -   number of steps to infer (value not stored explicitly in object)
    y_drive     -   (external) signal that continues to be presented to the reservoir
    r_infer     -   reservoir state if the signal continues to be presented after training
    y_infer     -   state reconstruction assuming that signal continues to be presented
                    after training

@author: Adrian Wong
"""
from .reservoir import Reservoir, mapRC, diffRC
from . import connectivity
from . import integrators

__all__ = ["Reservoir", "mapRC", "diffRC", "connectivity", "integrators"]
