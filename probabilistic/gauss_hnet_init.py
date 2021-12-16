#!/usr/bin/env python3
# Copyright 2020 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :probabilistic/gauss_hnet_init.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/09/2020
# @version        :1.0
# @python_version :3.6.9
"""
Custom hypernet initialization for Gaussian target networks
-----------------------------------------------------------

The module :mod:`probabilistic.gauss_hnet_init` provides functions to
initialize hypernetworks suitable for target networks implementing the wrapper
:class:`probabilistic.gauss_mnet_interface.GaussianBNNWrapper`.
"""
import contextlib
import math
import os
import torch
import warnings
from warnings import warn

from hnets.mlp_hnet import HMLP
from hnets.chunked_mlp_hnet import ChunkedHMLP
from hnets.hnet_perturbation_wrapper import HPerturbWrapper
from hnets.structured_mlp_hnet import StructuredHMLP
from probabilistic import GaussianBNNWrapper

def gauss_hyperfan_init(hnet, mnet=None, mean_var=0.003, rho_var=0.083,
                        use_xavier=False, uncond_var=1., cond_var=1.,
                        keep_hyperfan_mean=False, eps=1e-5):
    r"""Initialize a hypernetwork to produce Gaussian output weights.

    The class :class:`probabilistic.gauss_mnet_interface.GaussianBNNWrapper`
    initializes its weights such that mean parameters :math:`\mu` follow a
    uniform distribution :math:`\mathcal{U}(-0.1, 0.1)` (with variance
    :math:`0.1^2 / 3 \approx 0.003`), whereas encoded variance parameters
    :math:`\rho` follow by default :math:`\mathcal{U}(-3, -2)` (with variance
    :math:`0.5^2 / 3 \approx 0.083`).

    The initialization doesn't distinguish further between the types of weight
    tensors to be initialized (i.e., bias vectors and weight tensors and others
    are initialized the same way).

    Inspired by the
    `hyperfan init <https://openreview.net/forum?id=H1lma24tPB>`__ paper and our
    implementations in
    :meth:`hnets.mlp_hnet.HMLP.apply_hyperfan_init` and
    :meth:`hnets.chunked_mlp_hnet.ChunkedHMLP.apply_chunked_hyperfan_init`
    we initialize the hypernetwork in a way such that its output variances match
    the ones provided by arguments ``mean_var`` and ``rho_var``.

    Note:
        This function assumes that constructor argument ``apply_rho_offset``
        was activated in class
        :class:`probabilistic.gauss_mnet_interface.GaussianBNNWrapper`. Hence,
        this function does not modify the expected value of the outputs
        interpreted as :math:`\rho`. Instead, it assumes that all input
        embeddings have an expected value of zero.

    Note:
        This function can only initialize hypernetwork instances from class
        :class:`hnets.mlp_hnet.HMLP` and
        :class:`hnets.chunked_mlp_hnet.ChunkedHMLP`.

    **How the expected value of outputs could be shifted?**

    For future reference, we state here how the expected value of hypernet
    outputs could be shifted, even though, this function does **not** implement
    such functionality.

    Let's denote the hypernetwork input by
    :math:`\mathbf{e} \in \mathbb{R}^{n_e}`, and it's output by
    :math:`\mathbf{w} \in \mathbb{R}^m`. The last hypernet hidden layer will
    be denoted by :math:`\mathbf{x} \in \mathbb{R}^n`.

    Importantly, we invoke all Xavier assumptions except that we assume that
    expected values are zero. Instead, we need to fine-tune expected values to
    achieve our goals.

    First, we need to initialize the hidden layers such that input variance
    and expected value are preserved, i,.e.,
    :math:`\mathbb{E}[e] = \mathbb{E}[x]` and
    :math:`\text{Var}(e) = \text{Var}(x)`.

    Wlog we assume that there is only one hidden layer such that
    :math:`\mathbf{x} = V \mathbf{e}`. It holds that

    .. math::

        \text{Var}(x) = n_e \text{Var}(v) \text{Var}(e) + \
            n_e \text{Var}(v) \mathbb{E}[e]^2 + \
            n_e \text{Var}(e) \mathbb{E}[v]^2

    Enforcing that :math:`\text{Var}(e) = \text{Var}(x)` we arrive at the
    expression

    .. math::

        \text{Var}(v) = \frac{ \
            \text{Var}(e) - n_e \text{Var}(e) \mathbb{E}[v]^2}{ \
            n_e \big( \text{Var}(e) + \mathbb{E}[e]^2 \big)}

    which enforces the condition :math:`\mathbb{E}[v]^2 < \frac{1}{n_e}`.

    The expected value can be written as
    :math:`\mathbb{E}[x] = n_e \mathbb{E}[v] \mathbb{E}[e]`. To enforce that
    :math:`\mathbb{E}[e] = \mathbb{E}[x]` we need to set
    :math:`\mathbb{E}[v] \equiv \frac{1}{n_e}`. Together with the above
    condition of :math:`\mathbb{E}[v]^2 < \frac{1}{n_e}` we need to ensure that
    :math:`n_e > 1`.

    Note:

        An actual implementation of the above scheme would need to respect more
        details, for instance that the input embedding will in general be a
        concatenation of several embeddings with potentially different
        expected values and different varaiances.

    Note:

        The above scheme can be relaxed, since nothing requires us to enforce
        :math:`\mathbb{E}[e] = \mathbb{E}[x]` and
        :math:`\text{Var}(e) = \text{Var}(x)`. We just need to be able to
        compute the expected value and variance of the last hidden layer. But
        as shown below, the conditions are easy to enforce by manipulating the
        input without complicated expressions when initialized this way.

    Now, we can turn to the initialization of the last hidden layer to achieve
    the desired target expected value :math:`\mathbb{E}[w]` and target variance
    :math:`\text{Var}(w)`. We assume the output is computed as follows
    :math:`\mathbf{w} = H \mathbf{x}`.

    The expected value :math:`\mathbb{E}[w] = n \mathbb{E}[h] \mathbb{E}[x]`
    can be set to the target by setting

    .. math::

        \mathbb{E}[h] = \frac{\mathbb{E}[w]}{n \mathbb{E}[x]}

    This necessitates that :math:`\mathbb{E}[x] \neq 0` (in the above scheme,
    that would be given by ensuring that :math:`\mathbb{E}[e] \neq 0`).

    The variance

    .. math::

        \text{Var}(w) = n \text{Var}(h) \text{Var}(x) + \
            n \text{Var}(h) \mathbb{E}[x]^2 + n \text{Var}(x) \mathbb{E}[h]^2

    can be rearranged to

    .. math::

        \text{Var}(h) = \frac{ \
            \text{Var}(w) - n \text{Var}(x) \mathbb{E}[h]^2}{ \
            n \big( \text{Var}(x) + \mathbb{E}[x]^2 \big)}

    This equation dictates the following condition

    .. math::

        \text{Var}(w) > n \text{Var}(x) \mathbb{E}[h]^2

    Which could be enforced by choosing a proper last hidden layer variance
    :math:`\text{Var}(x) < \frac{\text{Var}(w)}{n \mathbb{E}[h]^2}`

    Args:
        hnet (hnets.mlp_hnet.HMLP or hnets.chunked_mlp_hnet.ChunkedHMLP): The
            hypernetwork to be initialized.
        mnet (probabilistic.gauss_mnet_interface.GaussianBNNWrapper, optional):
            The main network that will be the target of the initialization.
            Only required to execute some sanity checks.
        mean_var (float): The desired target variance for mean values (first
            half of hypernet outputs).
        rho_var (float): The desired target varaince for :math:`\rho` values
            (second half of hypernetwork outputs).
        use_xavier (bool): Whether Kaiming (``False``) or Xavier (``True``)
            init should be used to initialize the hidden layers of the main
            network.
        uncond_var (float): See argument ``uncond_var`` of method
            :meth:`hnets.mlp_hnet.HMLP.apply_hyperfan_init`.
        cond_var (float): See argument ``cond_var`` of method
            :meth:`hnets.mlp_hnet.HMLP.apply_hyperfan_init`.
        keep_hyperfan_mean (bool): This function uses internally the methods
            ``apply_hyperfan_init`` or ``apply_chunked_hyperfan_init``. If this
            option is ``True``, then the hyperfan init from this methods for
            means is kept and only the ones for :math:`\rho` is modified by this
            method.

            Note:
                Argument ``mean_var`` will be ignored if ``True``.
        eps (float): See argument ``eps`` of corresponding method
            ``apply_chunked_hyperfan_init``.
    """
    if isinstance(hnet, HPerturbWrapper):
        hnet = hnet.internal_hnet
    assert isinstance(hnet, (HMLP, ChunkedHMLP, StructuredHMLP))

    if mnet is not None:
        assert isinstance(mnet, GaussianBNNWrapper)
        assert mnet.rho_offset_applied, 'This function does not influence ' + \
            'the mean of the outputted weight distribution.'

    assert (keep_hyperfan_mean or mean_var > 0) and rho_var > 0
    assert cond_var > 0 and uncond_var > 0

    ####################
    ### MLP Hypernet ###
    ####################
    if isinstance(hnet, HMLP):
        # We set the weight matrices to zero while adjusing the
        # bias vector init only. Not sure whether this will cause practical
        # issues.
        w_val_arg = [None] * len(hnet.target_shapes)
        w_var_arg = [None] * len(hnet.target_shapes)
        b_val_arg = [None] * len(hnet.target_shapes)
        b_var_arg = [None] * len(hnet.target_shapes)
        for i in range(len(hnet.target_shapes)):
            if i < len(hnet.target_shapes) // 2: # Mean parameters
                if not keep_hyperfan_mean:
                    w_val_arg[i] = 0
                    w_var_arg[i] = 0
                    b_val_arg[i] = 0
                    b_var_arg[i] = mean_var
            else:
                w_val_arg[i] = 0
                w_var_arg[i] = 0
                b_val_arg[i] = 0
                b_var_arg[i] = rho_var

        hnet.apply_hyperfan_init(method='in', use_xavier=use_xavier,
            uncond_var=uncond_var, cond_var=cond_var, mnet=mnet,
            w_val=w_val_arg, w_var=w_var_arg, b_val=b_val_arg, b_var=b_var_arg)

    ###############################
    ### Structured MLP Hypernet ###
    ###############################
    elif isinstance(hnet, StructuredHMLP):
        for ii, int_hnet in enumerate(hnet.internal_hnets):
            w_val_arg = [None] * len(int_hnet.target_shapes)
            w_var_arg = [None] * len(int_hnet.target_shapes)
            b_val_arg = [None] * len(int_hnet.target_shapes)
            b_var_arg = [None] * len(int_hnet.target_shapes)
            for i in range(len(int_hnet.target_shapes)):
                if ii < len(hnet.internal_hnets) // 2: # Mean hnets
                    if not keep_hyperfan_mean:
                        w_val_arg[i] = 0
                        w_var_arg[i] = 0
                        b_val_arg[i] = 0
                        b_var_arg[i] = mean_var
                else:
                    w_val_arg[i] = 0
                    w_var_arg[i] = 0
                    b_val_arg[i] = 0
                    b_var_arg[i] = rho_var

            int_hnet.apply_hyperfan_init(method='in', use_xavier=use_xavier,
                uncond_var=uncond_var, cond_var=cond_var, mnet=mnet,
                w_val=w_val_arg, w_var=w_var_arg, b_val=b_val_arg,
                b_var=b_var_arg)

    ########################
    ### Chunked Hypernet ###
    ########################
    elif isinstance(hnet, ChunkedHMLP):
        target_vars = [mean_var] * (len(hnet.target_shapes) // 2) + \
                      [rho_var] * (len(hnet.target_shapes) // 2)

        hnet.apply_chunked_hyperfan_init(method='in', use_xavier=use_xavier,
            uncond_var=uncond_var, cond_var=cond_var, eps=eps, mnet=mnet,
            target_vars=target_vars)

if __name__ == '__main__':
    pass


