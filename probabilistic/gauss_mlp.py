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
# @title          :probabilistic/gauss_mlp.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/13/2020
# @version        :1.0
# @python_version :3.6.9
"""
MLP that implements the local reparametrization trick
-----------------------------------------------------

The module :mod:`probabilisic.gauss_mlp` contains a MLP reimplementation that
implements the local reparametrization trick as described in

    Kingma et al., "Variational Dropout and the Local Reparameterization Trick",
    2015. https://arxiv.org/abs/1506.02557
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from mnets import MLP
from probabilistic import prob_utils as putils

class GaussianMLP(MLP):
    r"""Multi-layer-perceptron with fully-factorized Gaussian weight posterior.

    This class assumes that the weight posterior is a fully-factorized Gaussian,
    such that the local reparametrization trick can be applied. Hence, assuming
    a batch size :math:`M`, a layer input size of :math:`N_{in}` and a layer
    output size of :math:`N_{out}` a linear layer performs the following
    operation

    .. math::

        Y = X W

    with inputs :math:`X \in \mathbb{R}^{M \times N_{in}}`, weights
    :math:`W \in \mathbb{R}^{N_{in} \times N_{out}}` and outputs
    :math:`Y \in \mathbb{R}^{M \times N_{out}}`.

    Elements in :math:`W` are expected to arise from the following factorized
    Gaussian

    .. math::

        q_{\phi}(w_{ij}) = \mathcal{N} (\mu_{ij}, \sigma_{ij}^2)

    As shown by `Kingma et al. <https://arxiv.org/abs/1506.02557>`__, the
    variance of the likelihood estimator will decrease to zero with increasing
    batch size if a different weight sample is drawn for every sample in the
    mini-batch.

    They show a more computational efficient solution is to sample activations,
    which follow the following factorized Gaussian given the above weight
    posterior

    .. math::

        q_{\phi}(y_{mj}) = \mathcal{N} (\gamma_{mj}, \delta_{mj}) \quad \
            \text{with} \quad \
            \gamma_{mj} = \sum_{i=1}^{N_{in}} x_{mi} \mu_{ij} \quad \
            \delta_{mj} = \sum_{i=1}^{N_{in}} x_{mi}^2 \sigma_{ij}^2

    Hence, we can simply sample activations using the above distribution and
    the reparametrization trick

    .. math::

        y_{mj} = \gamma_{mj} + \sqrt{\delta_{mj}} \zeta_{mj} \quad \text{, } \
            \zeta_{mj} \sim \mathcal{N}(0, 1)

    What about bias vectors :math:`b \in \mathbb{R}^{N_{out}}`? We can easily
    incorporating them in the above formulas, by just assuming that we append an
    additional column to :math:`X` containing ones and an additional row to
    :math:`W` containing the bias vector. In this case, the formulas become

    .. math::

        \gamma_{mj} = \sum_{i=1}^{N_{in}} x_{mi} \mu_{ij} + \
            \hat{\mu}_{j} \quad \
            \delta_{mj} = \sum_{i=1}^{N_{in}} x_{mi}^2 \sigma_{ij}^2 + \
            \hat{\sigma}_{j}^2

    where we assume that
    :math:`q_{\phi}(b_{j}) = \mathcal{N} (\hat{\mu}_{j}, \hat{\sigma}_{j}^2)`.

    Note:
        This class will not hold mean and variance weights (even though, it
        is exoected that they are passed to the :meth:`forward`). Therefore, we
        recommend using an instance of this class always wrapped by class
        :class:`probabilistic.gauss_mnet_interface.GaussianBNNWrapper`.

    Args:
        (....): See constructor documentation of class :class:`mnets.mlp.MLP`.
    """
    def __init__(self, n_in=1, n_out=1, hidden_layers=[10, 10],
                 activation_fn=nn.ReLU(), use_bias=True, no_weights=False,
                 init_weights=None, verbose=True):
        # We don't need to change anything from the base class constructor,
        # we just need to modify the forward method.
        MLP.__init__(self, n_in=n_in, n_out=n_out, hidden_layers=hidden_layers,
                 activation_fn=activation_fn, use_bias=use_bias,
                 no_weights=no_weights, init_weights=init_weights,
                 dropout_rate=-1, use_spectral_norm=False,
                 use_batch_norm=False, use_context_mod=False, out_fn=None,
                 verbose=verbose)

    def forward(self, x, mean, rho, logvar_enc=False, mean_only=False,
                sample=None, rand_state=None):
        r"""Compute the output :math:`y` of this network given the input
        :math:`x` by drawing a different weight sample for every sample in the
        input batch.

        This is achieved by using the local reparametrization trick.

        Args:
            (....): See docstring of method
                :meth:`mnets.mlp.MLP.forward`. We
                provide some more specific information below.
            mean (list): The set of mean parameters. The shapes of the tensors
                are expected to follow attribute
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.
            rho (list): The set of :math:`\rho` parameters. The shapes of the
                tensors are expected to follow attribute
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.
            logvar_enc (bool): See docstring of function
                :func:`probabilistic.prob_utils.decode_and_sample_diag_gauss`.
            mean_only (bool): If ``True``, the hidden activation will simply
                be set to :math:`\gamma`.
            sample (optional): If a specific weight sample is provided, the
                local raparametrization trick will be disabled and the super
                class ``forward`` method will be called where the sample is
                passed as argument ``weights``.
            rand_state (torch.Generator, optional): This generator would be used
                to realize the reparametrization trick (i.e., the activation
                sampling), if specified. This way, the behavior of the forward
                pass is easily reproducible.

        Returns:
            The output of the network.
        """
        if sample is not None:
            # Do not use local reparam trick!
            return MLP.forward(self, x, weights=sample)

        assert len(mean) == len(self.param_shapes) and \
            len(rho) == len(self.param_shapes)

        for i, s in enumerate(self.param_shapes):
            assert np.all(np.equal(s, list(mean[i].shape))) and \
                np.all(np.equal(s, list(rho[i].shape)))

        _, var = putils.decode_diag_gauss(rho, return_var=True,
                                          logvar_enc=logvar_enc)

        w_means = []
        b_means = []
        w_vars = []
        b_vars = []

        for i, p in enumerate(mean):
            if self.has_bias and i % 2 == 1:
                b_means.append(p)
                b_vars.append(var[i])
            else:
                w_means.append(p)
                w_vars.append(var[i])

        ###########################
        ### Forward Computation ###
        ###########################
        hidden = x

        for l in range(len(w_means)):
            mu = w_means[l]
            sigma2 = w_vars[l]

            if self.has_bias:
                mu_hat = b_means[l]
                sigma2_hat = b_vars[l]
            else:
                mu_hat = None
                sigma2_hat = None

            # Compute means and variances of activations.
            gamma = F.linear(hidden, mu, bias=mu_hat)

            if mean_only:
                hidden = gamma
            else:
                delta = F.linear(hidden.pow(2), sigma2, bias=sigma2_hat)

                # Sample via reparametrization trick.
                if rand_state is None:
                    hidden = Normal(gamma, delta.sqrt()).rsample()
                else:
                    eps = torch.normal(torch.zeros_like(gamma), 1.,
                                       generator=rand_state)
                    hidden = gamma + eps * delta.sqrt()

            # Only for hidden layers.
            if l < len(w_means) - 1:
                # Non-linearity
                if self._a_fun is not None:
                    hidden = self._a_fun(hidden)

        return hidden

if __name__ == '__main__':
    pass


