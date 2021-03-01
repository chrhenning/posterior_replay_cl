#!/usr/bin/env python3
# Copyright 2019 Christian Henning
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
# @title           :probabilistic/prob_utils.py
# @author          :ch, mc
# @contact         :henningc@ethz.ch
# @created         :08/25/2019
# @version         :1.0
# @python_version  :3.6.8
"""
Helper functions when working with Bayesian Neural Networks
-----------------------------------------------------------
"""
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from warnings import warn

def decode_and_sample_diag_gauss(mean, rho, logvar_enc=False, generator=None,
                                 is_radial=False):
    r"""Sample from a Gaussian distribution with diagonal covariance.

    The covariance of the distribution is given in an encoded form :math:`\rho`.
    This method will assume that the standard deviation can be retrieved via
    a softplus operation

    .. math::

        \sigma = \text{softplus} (\rho)

    Except if it is assumed that :math:`rho` (``rho``) encodes the log-variance
    (parameter ``logvar_enc``), in which case the standard deviation is
    retrieved via

    .. math::

        \sigma = \exp (\frac{1}{2} \rho)

    The samples are drawn according to :math:`\mathcal{N}(\mu, I \sigma^2)`,
    except if ``is_radial`` is ``True``, in which case they are sampled
    according to

    .. math::

        \mathbf{w} = \mathbf{\mu} + \mathbf{\sigma} \odot \
        \frac{\epsilon_{MFVI}}{\mid\mid \epsilon_{MFVI} \mid\mid} \cdot r

    where :math:`\epsilon_{MFVI} \sim \mathcal{N} (0, \mathbf{I})`,
    :math:`r \sim \mathcal{N} (0, 1)`, and the norm of :math:`\epsilon_{MFVI}`
    is computed per parameter tensor (e.g., for the weights or the biases of
    each layer). Also :math:`r` is resampled for every parameter tensor.

    When drawing many samples, it might be more efficient to use a combination
    of functions :func:`decode_diag_gauss` and :func:`sample_diag_gauss`.

    Args:
        mean (list): List of tensors (:class:`torch.Tensor`), that represent
            the mean of the Gaussian distribution.
        rho (list): List of tensors, that represent the encoded variance
            :math:`\rho`.
        logvar_enc (bool): Whether the encoded variances ``rho`` represent
            log-variances.
        generator (torch.Generator, optional): A generator can be passed to
            obtain more control over the reproducibility of the random sampling
            process.
        is_radial (bool, optional): If ``True``, the weights will be sampled
            according to a radial distribution, and not a Gaussian one.

    Returns:
        (list): A sample from the desired distribution, retrieved via the
        reparametrization trick.
    """
    assert(len(mean) == len(rho))

    sample = []
    device = mean[0].device if len(mean) > 0 else None
    for i in range(len(mean)):
        if logvar_enc:
            std = torch.exp(0.5 * rho[i])
        else:
            std = F.softplus(rho[i])

        eps = torch.normal(torch.zeros_like(std), 1., generator=generator)
        if not is_radial:
            sample.append(mean[i] + eps * std)
        else:
            r = torch.normal(torch.tensor(0.).to(device), 1.,
                             generator=generator)
            sample.append(mean[i] + std * r * eps / torch.norm(eps, p=2))

    return sample

def decode_diag_gauss(rho, logvar_enc=False, return_var=False,
                      return_logvar=False):
    r"""Decode the standard deviation for a Gaussian distribution with diagonal
    covariance.

    We consider a Gaussian distribution where the covariance is encoded in
    :math:`\rho` (``rho``) as real numbers. We can extract the standard
    deviation from :math:`\rho` as described in the documentation of
    :func:`decode_and_sample_diag_gauss`.

    Args:
        (....): See docstring of function :func:`decode_and_sample_diag_gauss`.
        return_var (bool, optional): If ``True``, the variance :math:`\sigma^2`
            will be returned as well.
        return_logvar (bool, optional): If ``True``, the log-variance
            :math:`\log\sigma^2` will be returned as well.

    Returns:
        (tuple): Tuple containing:

        - **std** (list): The standard deviation :math:`\sigma`.
        - **var** (list, optional): The variance :math:`\sigma^2`. See argument
          ``return_var``.
        - **logvar** (list, optional): The log-variance :math:`\log\sigma^2`.
          See argument ``return_logvar``.
    """
    ret_std = []
    ret_var = []
    ret_logvar = []

    for i in range(len(rho)):
        if logvar_enc:
            std = torch.exp(0.5 * rho[i])
            logvar = rho[i]
        else:
            std = F.softplus(rho[i])
            logvar = 2 * torch.log(std)

        ret_std.append(std)
        ret_logvar.append(logvar)

        if return_var:
            ret_var.append(std**2)

    if return_var and return_logvar:
        return ret_std, ret_var, ret_logvar
    elif return_var:
        return ret_std, ret_var
    elif return_logvar:
        return ret_std, ret_logvar

    return ret_std

def sample_diag_gauss(mean, std, generator=None, is_radial=False):
    """Get a sample from a multivariate Gaussian distribution with diagonal
    covariance matrix.

    Samples are produced using the reparametrization trick.

    Note: 
        If ``is_radial`` is set to ``True``, samples will instead by obtained
        from a radial BNN distribution instead of a multivariate Gaussian.
        For details refer to docstring of :func:`decode_and_sample_diag_gauss`.

    Args:
        mean: A list of tensors. See return value of method
            :func:`extract_mean_std`.
        std: A list of tensors with the same shapes as `mean`. See return value
            of method :func:`extract_mean_std`.
        generator (torch.Generator, optional): A generator can be passed to
            obtain more control over the reproducibility of the random sampling
            process.
        is_radial (bool, optional): If ``True``, the weights will be sampled
            according to a radial distribution, and not a Gaussian one.

    Returns:
        A list of tensors, where each is a sample from the diagonal Gaussian
        distributions defined by entries of `mean` and `std`.
    """
    sample = []
    device = mean[0].device if len(mean) > 0 else None
    for i, m in enumerate(mean):
        eps = torch.normal(torch.zeros_like(m), 1., generator=generator)

        if not is_radial:
            sample.append(m + eps * std[i])
        else:
            r = torch.normal(torch.tensor(0.).to(device), 1., \
                             generator=generator)
            sample.append(m +  std[i] * r * eps / torch.norm(eps, p=2))

    return sample

def kl_diag_gaussians(mean_a, logvar_a, mean_b, logvar_b):
    r"""Compute the KL divergence between 2 diagonal Gaussian distributions.

    .. math::
        KL \big( p_a(\cdot) \mid\mid  p_b(\cdot) \big)

    Args:
        mean_a: Mean tensors of the first distribution (see argument `mean` of
            method :func:`sample_diag_gauss`).
        logvar_a: Log-variance tensors with the same shapes as the `mean_a`
            tensors.
        mean_b: Same as `mean_a` for second distribution.
        logvar_b: Same as `logvar_a` for second distribution.

    Returns:
        The analytically computed KL divergence between these distributions.
    """
    mean_a_flat = torch.cat([t.view(-1) for t in mean_a])
    logvar_a_flat = torch.cat([t.view(-1) for t in logvar_a])
    mean_b_flat = torch.cat([t.view(-1) for t in mean_b])
    logvar_b_flat = torch.cat([t.view(-1) for t in logvar_b])

    ### Using our own implementation ###
    kl = 0.5 * torch.sum(-1 + \
        (logvar_a_flat.exp() + (mean_b_flat - mean_a_flat).pow(2)) / \
        logvar_b_flat.exp() + logvar_b_flat - logvar_a_flat)

    return kl

def kl_diag_gauss_with_standard_gauss(mean, logvar):
    """Compute the KL divergence between an arbitrary diagonal Gaussian
    distributions and a Gaussian with zero mean and unit variance.

    Args:
        mean: Mean tensors of the distribution (see argument `mean` of
            method :func:`sample_diag_gauss`).
        logvar: Log-variance tensors with the same shapes as the `mean`
            tensors.

    Returns:
        The analytically computed KL divergence between these distributions.
    """
    mean_flat = torch.cat([t.view(-1) for t in mean])
    logvar_flat = torch.cat([t.view(-1) for t in logvar])
    var_flat = logvar_flat.exp()

    return -0.5 * torch.sum(1 + logvar_flat - mean_flat.pow(2) - var_flat)

def square_wasserstein_2(mean_a, logvar_a, mean_b, logvar_b):
    r"""Compute the square of the Wasserstein-2 distance between 2 diagonal 
    Gaussian distributions.

    Args:
        mean_a: Mean tensors of the first distribution (see argument `mean` of
            method :func:`sample_diag_gauss`).
        logvar_a: Log-variance tensors with the same shapes as the `mean_a`
            tensors.
        mean_b: Same as `mean_a` for second distribution.
        logvar_b: Same as `logvar_a` for second distribution.

    Returns:
        The analytically computed square of the Wasserstein distance between 
        these distributions.
    """
    mean_a_flat = torch.cat([t.view(-1) for t in mean_a])
    var_a_flat = torch.cat([t.view(-1) for t in logvar_a]).exp()
    mean_b_flat = torch.cat([t.view(-1) for t in mean_b])
    var_b_flat = torch.cat([t.view(-1) for t in logvar_b]).exp()

    square_wasserstein = torch.norm((mean_a_flat - mean_b_flat), p=2)**2 + \
        (var_a_flat + var_b_flat - 2*torch.sqrt(var_a_flat*var_b_flat)).sum()

    return square_wasserstein

def kl_radial_bnn_with_diag_gauss(mean_a, std_a, mean_b, std_b,
                                  ce_sample_size=10, generator=None):
    r"""Compute the KL divergence between one radial BNN distribution and one
    diagonal Gaussian distribution.

    For a radial BNN distribution :math:`p_a(\cdot)` and a Gaussian distribution
    :math:`p_b(\cdot)`, the expression is given by

    .. math::
        KL \big( p_a(\cdot) \mid\mid  p_b(\cdot) \big) = \
        \int p_a(\cdot) \log{p_a(\cdot)} \,d\cdot - \
        \int p_a(\cdot) \log{p_b(\cdot)} \,d\cdot = \
        - h(p_a(\cdot)) + h(p_a(\cdot), p_b(\cdot))

    where :math:`h(p_a(\cdot))` denotes the entropy of :math:`p_a(\cdot)` and 
    :math:`h(p_a(\cdot), p_b(\cdot))` the cross-entropy between 
    :math:`p_a(\cdot)` and :math:`p_b(\cdot)`.

    **Entropy**

    Since :math:`p_a(\cdot)` is a radial BNN distribution its entropy can be 
    computed analytically according to

    .. math::
        h(p_a(\cdot)) = \sum_{i} \log [\sigma^{a}_i] + const

    as described in Eq.5 from
    `Farquhar et al. <https://arxiv.org/abs/1907.00865>`__, where :math:`i` is
    a sum over the weights.

    **Cross-entropy**

    The cross-entropy term can be estimated taking :math:`N` Monte-Carlo samples 
    from :math:`p_a(\cdot)` and averaging their log-probability under 
    :math:`p_b(\cdot)`.

    .. math::
        h(p_a(\cdot), p_b(\cdot)) \approx \
            - \frac{1}{N} \sum_{n}^{N} \log[p_b(\cdot^{(n)})]

    Args:
        mean_a: Mean tensors of the radial BNN distribution (see argument `mean` 
            of method :func:`sample_diag_gauss`).
        std_a: The standard deviation :math:`\sigma` with the same shapes as the 
            `mean_a` tensors.
        mean_b: Same as `mean_a` for diagonal Gaussian distribution.
        std_b: Same as `std_a` for diagonal Gaussian distribution.
        ce_sample_size (int, optional): The number of weight samples to draw to
            estimate the cross-entropy term of the KL.
        generator (torch.Generator, optional): A generator can be passed to
            obtain more control over the reproducibility of the random sampling
            process.

    Returns:
        The computed KL divergence between these distributions.
    """

    ### Analytically compute entropy term.
    # Since we sum the log values of the standard deviation, we can simply
    # flatten the parameter tensors and compute the log and sum on this vector.
    entropy = - torch.log(torch.cat([t.view(-1) for t in std_a])).sum()

    ### Estimate cross-entropy term.
    cross_entropy = 0
    for n in range(ce_sample_size):
        # Take a weight sample from :math:`p_a(w)`.
        sample = sample_diag_gauss(mean_a, std_a, generator=generator,
                                   is_radial=True)
        for i in range(len(mean_b)):
            # Define the normal distribution for the prior.
            p_b = Normal(mean_b[i], std_b[i])
            # Note, all weights are independent from one another (no
            # correlations) modelled in our weight distributions. Thus, we can
            # just sum all log-probs within and across parameter tensors.
            cross_entropy += p_b.log_prob(sample[i]).sum()

    cross_entropy /= ce_sample_size

    return entropy - cross_entropy

def sample_diag_gaus_from_hnet(hnet_outputs):
    """This method uses the reparametrization trick to sample a set of main
    network weights assuming the output of the hypernetwork represents the
    mean and log-variances of a diagonal Gaussian distribution.

    When drawing many samples, it might be more efficient to use a combination
    of methods :func:`extract_mean_std` and :func:`sample_diag_gauss`.

    .. deprecated:: 1.0
        Please use a main network wrapper such as
        :class:`probabilistic.gauss_mnet_interface.GaussianBNNWrapper` or the
        function :func:`decode_and_sample_diag_gauss` rather than working with
        the hypernet output directly.

    Args:
        hnet_outputs: A list of tensors. The first half of this list is
            interpreted as mean values and the second half as log variance
            values.

    Returns:
        A sample of the distribution represented by the hypernet output.
    """
    warn('Please use a main network wrapper such as class' +
        '"probabilistic.gauss_mnet_interface.GaussianBNNWrapper" or the' +
        'function "decode_and_sample_diag_gauss" rather than working with ' +
        'the hypernet output directly.', DeprecationWarning)

    assert(len(hnet_outputs) % 2 == 0)
    n = len(hnet_outputs) // 2

    mu = hnet_outputs[n:]
    logvar = hnet_outputs[:n]

    sample = []
    for i in range(n):
        std = torch.exp(0.5 * logvar[i])
        sample.append(Normal(mu[i], std).rsample())

    return sample

def extract_mean_std(hnet_outputs, return_logvar=False):
    """Extract mean and standard deviation for a multivariate Gaussian
    distribution with diagonal covaiance matrix from a hypernetwork that outputs
    mean and log-variance.

    .. deprecated:: 1.0
        Please use a main network wrapper such as
        :class:`probabilistic.gauss_mnet_interface.GaussianBNNWrapper` or the
        function :func:`decode_diag_gauss` rather than working with
        the hypernet output directly.

    Args:
        hnet_outputs: See docstring of method :func:`sample_diag_gaus_weights`.
        return_logvar (optional): If set, a third value will be returned,
            corresponding to the log-variance.

    Returns:
        Two lists of tensors: `mean` and `std`.
    """
    warn('Please use a main network wrapper such as class' +
        '"probabilistic.gauss_mnet_interface.GaussianBNNWrapper" or the' +
        'function "decode_diag_gauss" rather than working with ' +
        'the hypernet output directly.', DeprecationWarning)

    assert(len(hnet_outputs) % 2 == 0)
    n = len(hnet_outputs) // 2

    mean = hnet_outputs[n:]
    logvar = hnet_outputs[:n]
    std = [torch.exp(0.5 * logvar[i]) for i in range(n)]

    if return_logvar:
        return mean, std, logvar
    return mean, std

if __name__ == '__main__':
    pass


