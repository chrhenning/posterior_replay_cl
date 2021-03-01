#!/usr/bin/env python3
# Copyright 2020 Maria Cervera
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
# @title           :probabilistic/ssge_utils.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :08/12/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Using the Spectral Stein Estimator CL for implicit weight distributions
-----------------------------------------------------------------------

In the script :mod:`probabilistic.ssge_utils` we use the Spectral Stein
Gradient Esimator to approximate the score function (i.e. the gradient of the
log density of our implicit distribution with respect to the location), as
described in

    Shi, Jiaxin, Shengyang Sun, and Jun Zhu. "A spectral approach to gradient 
    estimation for implicit distributions." ICML, 2018.
    https://arxiv.org/abs/1806.02925

By directly estimating the gradients, instead of the implicit distribution
itself, we avoid having a GAN-like training procedure as is the case with AVB.
"""
import numpy as np
import torch
from torch.autograd import grad
from warnings import warn

class SSGE():
    """Spectral Stein Gradient Estimator.

    This class contains methods that are necessary for making an estimation of
    the score function according to the SSGE algorithm. Details of this method
    can be found in

        Shi, Jiaxin, Shengyang Sun, and Jun Zhu. "A spectral approach to
        gradient estimation for implicit distributions." ICML, 2018.
        https://arxiv.org/abs/1806.02925

    Args:
        kernel (str): The type of kernel. Only RBF (``'rbf'``) is implemented!
        sigma (float): The width of the RBF kernel.
    """
    def __init__(self, kernel='rbf', kernel_width=1.):
        self._kernel = kernel
        self._sigma = kernel_width
        if kernel != 'rbf':
            raise NotImplementedError

    def rbf_kernel(self, x1, x2):
        r"""Compute the radial basis function kernel between two vectors.

        The value is given by

        .. math::

            \exp( - \frac{\lVert x1 - x2 \rVert_2^2}{2 \sigma^2})

        where :math:`\lVert \cdot \rVert_2` denotes the euclidean norm.

        Note that this function can also handle matrices.

        Args:
            x1 (torch.Tensor): The first vector of length ``D`` or matrix of
                dimensions ``[B1, D]`` where ``B1`` are the number of samples.
            x2 (torch.Tensor): The second vector of length ``D`` or matrix of
                dimensions ``[B2, D]`` where ``B2`` are the number of samples.

        Returns:
            (float): The value of the RBF kernel between the two vectors or
                matrices. If the inputs are vectors, this matrix returns a
                scalar. Else, it returns a tensor of shape ``[B1, B2]``.
        """
        assert x1.shape[-1] == x2.shape[-1]
        if len(x1.shape) == 1:
            squared_norm = (x1 - x2).pow(2).sum()
            #norm = torch.norm(x1 - x2, p=2)
        else:
            squared_norm = (x1[:,None,:] - x2[None,:,:]).pow(2).sum(dim=2)
            #norm = torch.cdist(x1[None, :,:], x2[None, :,:], p=2)[0]
        return torch.exp(- squared_norm/(2 * self._sigma**2))

    def grad_rbf_kernel(self, x1, x2, K=None):
        r"""Compute the gradient of the RBF kernel wrt ``x1``.

        The value is given by

        .. math::

            - \frac{1}{\sigma^2}(x1 - x2) \exp( - \frac{\lVert x1 - \
            x2 \rVert_2^2}{2 \sigma^2}) = \
            - \frac{1}{\sigma^2}(x1 - x2) k(x1, x2)

        where :math:`\lVert \cdot \rVert_2` denotes the euclidean norm.

        Note that this function can also handle input matrices.

        Args:
            (....): See docstring of method :meth:`rbf_kernel`.
            K (torch.Tensor, optional): The kernel matrix
                ``self.rbf_kernel(x1, x2)``. Can be passed if computed already
                for computational efficiency. Otherwise it will be recomputed.

        Returns:
            (float): The value of the RBF kernel gradient between the two
                vectors. If the inputs are vectors, this method returns a vector
                of dimensionality ``D``. Else, it returns a tensor of shape
                ``[B1, D, B2]``.
        """
        assert x1.shape[-1] == x2.shape[-1]

        if K is None:
            K = self.rbf_kernel(x1, x2)

        if len(x1.shape) == 1:
            return - (x1 - x2) / self._sigma**2 * K
        else:
            grad_K = - (x1[:,None,:] - x2[None,:,:]) / self._sigma**2 * \
                K[:,:,None]

            return grad_K.permute(0, 2, 1)

    def compute_eigenvectors(self, K):
        """Compute the eigenvalues and eigenvectors of a certain matrix.

        Args:
            K (torch.Tensor): The input matrix of dimensions ``[M, M]``.

                Note:
                    ``K`` is assumed to be symmetric.

        Returns:
            (tuple): Tuple containing:

            - **lamb**: The eigenvalues, sorted in descending order.
            - **u**: The corresponding eigenvectors. Each of them has
              dimensionality ``M``.
        """
        lamb, u = torch.symeig(K, eigenvectors=True)

        # We want descending and not ascending order.
        lamb = torch.flip(lamb, [0])
        u = torch.flip(u, [1])

        return lamb, u

    def compute_eigenfunctions(self, x, x_est, lamb, u):
        r"""Apply Nystrom method to compute eigenfunctions of the kernel matrix.

        Given a set of input samples :math:`x`, this function computes ``J``
        eigenfunctions, where the j-th eigenfunction :math:`\psi_j(x)` is
        approximated by

        .. math::

            \hat{\psi}_j(x) = \frac{\sqrt(M)}{\lambda_j} \sum_{m=1}^M u_{jm} \
            K(x, x^m)

        where :math:`u_j` and :math:`\lambda_j` are the corresponding
        eigenvectors and eigenvalues, ``M`` indicates the number of samples
        used for the estimation, and :math:`x_m` are the samples used for the
        estimation.

        Args:
            x (torch.Tensor): The sample for which to compute the 
                eigenfunctions. It has dimensionality ``[N, D]``.
            x_est (torch.Tensor): The samples used for the estimation. It has
                dimensions ``[M, D]``.
            lambda (torch.Tensor): The eigenvalues.
            u (torch.Tensor): The eigenvectors.

        Returns:
            (torch.Tensor): A tensor of dimensionality ``[N, J]``, where each
                column corresponds to a different eigenfunction, sorted
                according to the corresponding eigenvalue. The rows correspond
                to the estimations for different ``x`` samples.
        """
        M = x_est.shape[0]

        # Estimate the first J eigenfunctions using Nystrom's method.
        psi = np.sqrt(M) / lamb * torch.mm(self.rbf_kernel(x, x_est), u)

        return psi

    def compute_beta_coeffs(self, lamb, u, x_est, K=None):
        r"""Compute the coefficients for the spectral series of the gradient.

        These coefficients can be estimated according to

        .. math::

            \hat{\beta{ij}} = - \frac{1}{M} \sum_{m=1}^M \
                \nabla_{x_i} \hat{\psi}_j(\mathbf{x}^m) = \
                - \frac{1}{\sqrt{M} \lambda_j} \sum_{m=1}^M  \sum_{z=1}^M \
                \nabla_{x_i} k(\mathbf{x}^z, \mathbf{x}^m) u_{jm}

        Args:
            lamb (torch.Tensor): The eigenvalues.
            u (torch.Tensor): The eigenvectors.
            x_est (torch.Tensor or None): The samples to be used to estimate
                the score function. It has dimensions ``[M, D]`` where ``M``
                is the number of samples for the estimate.
            K (torch.Tensor, optional): See argument ``K`` of method
                :meth:`grad_rbf_kernel`.

        Returns:
            (torch.Tensor): The beta coefficients. It has dimensions ``[D, J]``.
        """
        M = x_est.shape[0]

        # Compute the gradient of the RBF kernel wrt our samples for estimation.
        grad_K = self.grad_rbf_kernel(x_est, x_est, K=K)

        # Compute the coefficients from the gradient and the eigenpairs.
        beta = - 1 / (np.sqrt(M) * lamb) *  torch.einsum("abc,cd->bd", \
            (grad_K, u))

        return beta

    def select_eigenvectors(self, lamb, u, J=-1, J_thr=1.):
        """Select the number of eigenvalues and eigenvectors.

        One of two criterions can be used: directly providing the number of
        eigenvalues via ``J``, or setting a max ratio for the eigenvalues via
        ``J_thr``.

        Args:
            J (int, optional): The maximum number of eigenfunctions to use for
                the spectral series of the gradient.
            J_thr (float, optional): The threshold for the ratio of eigenvalues
                to be used.

        Returns:
            (tuple): Tuple containing:

            - **lamb**: The ``J`` first eigenvalues, sorted in decreasing order.
            - **u**: The corresponding eigenvectors. Each of them has
              dimensionality ``M``.
        """
        # Determine the number of eigenvectors to be used.
        if J_thr != 1.:
            if (lamb < 0).sum() > 0:
                # At initilization, eigenvalues can be slightly negative due to
                # numerical instabilities.
                if (lamb < -1e-5).sum() > 0:
                    # Should never happen since we work with sym. matrices.
                    raise ValueError('Cant use a cumulative sum if some ' +
                        'eigenvalues are negative.')
                lamb[lamb < 0] = 0
            # Use the trick in (3.2) from the original paper.
            lamb_ratios = torch.cumsum(lamb, 0) / lamb.sum()
            lamb_valid_ratios = np.where(lamb_ratios.detach().cpu() < J_thr)[0]
            if len(lamb_valid_ratios) == 0:
                warn('The provided threshold for eigenvalue ratios is too ' +
                     'low and leads to zero eigenvalues being selected. A ' +
                     'higher threshold should have been chosen. Setting J=1!')
                J = 1
            else:
                J = lamb_valid_ratios[-1] + 1

        if J != -1:
            # Truncate to only J values.
            lamb = lamb[:J] # [J]
            u = u[:, :J] # [M, J]

        return lamb, u

    def estimate_gradient(self, x, x_est=None, x_sup=None, J=-1, J_thr=1.,
                          max_m=-1):
        r"""Estimate the value of the score function on the provided samples.

        This function returns an estimate of the score function values
        computed on the provided samples. It is given by

        .. math::

            g(\mathbf{x}) = \nabla_{\mathbf{x}} \log q(\mathbf{x})

        where :math:`x` corresponds to an individual input sample and has
        a dimensionality of ``D``. Its i-th element can be expanded into the
        spectral series

        .. math::

            g_i(\mathbf{x}) = \sum_{j=1}^\infty \beta_{ij} \psi_j(\mathbf{x})

        where :math:`\psi_j(\mathbf{x})` is the j-th eigenfunction and
        :math:`\beta_{ij}` is the i-th coefficient of for the series of the
        j-th eigenfunction.

        Args:
            x (torch.Tensor): The samples at which to evaluate the gradient.
                It has dimensions ``[N, D]``, where ``N`` is the number of
                samples and ``D`` their dimensionality.
                If ``x_est`` is ``None``, these samples will also be used to
                build the score function estimator.
            x_est (torch.Tensor or None): If provided, these samples will be
                used to estimate the score function. It has dimensionality
                ``[M, D]``. If ``None``, the samples used for estimating the
                score function will consist of ``x`` together with the
                supplementary samples provided in ``x_est`` (if provided).
            x_sup (torch.Tensor or None): The supplementary samples to be used
                to estimate the score function. Only relevant if ``x_est`` is
                ``None``. If ``M <= N``, then it will be ``None`` and the ``x`` 
                tensor will be also used to estimate the gradient. Else, it has
                dimensions ``[M-N, D]`` where ``M`` is the number of samples for
                the gradient estimate.
            J (int, optional): The maximum number of eigenfunctions to use for
                the spectral series of the gradient.
            J_thr (float, optional): The threshold for the ratio of eigenvalues
                to be used.
            max_m (int, optional): The maximum number ``M`` of samples to use
                for the estimation from ``x`` if ``x_sup`` is not provided.

        Returns:
            (torch.Tensor): The estimated gradient. It has dimensionality 
                ``[N, D]``, and its average can be computed by averaging along
                the first dimension.
        """
        # Only one of them might be provided.
        assert x_est is None or x_sup is None

        # Construct the tensors for estimation ``x_est`` and for evaluation
        # ``x_eval`` from the provided inputs.
        if x_est is None and x_sup is None:
            # Use x if no extra samples for the estimation are provided.
            x_est = x
            # Set the maximum number of samples.
            if max_m != -1 and max_m < x_est.shape[0]:
                x_est = x_est[:max_m, :]
            # M = max_m
        elif x_est is None:
            x_est = torch.cat((x, x_sup))

        # Use our samples to construct the Gram matrix.
        K = self.rbf_kernel(x_est, x_est)

        # Estimate its eigenvectors (lamb) and eigenvalues (u).
        lamb, u = self.compute_eigenvectors(K) # [M], [M, M]

        # Truncate the number of eigenvectors and eigenvalues.
        lamb, u = self.select_eigenvectors(lamb, u, J=J, J_thr=J_thr)#[J],[M, J]

        # Compute the eigenfunctions.
        psi = self.compute_eigenfunctions(x, x_est, lamb, u) # [N, J]

        # Compute the coefficients.
        beta = self.compute_beta_coeffs(lamb, u, x_est, K=K) # [D, J]

        # Estimate the gradient for each of the input samples.
        gradient = torch.matmul(psi, beta.t()) # [N, D]

        return gradient

def get_heuristic_kernel_width(x1, x2=None, max_m=-1):
    """Heuristically define the kernel width.

    In the original paper they define the kernel width as the median of
    the pairwise distances between all samples.

    Note:
        If ``x2`` is ``None``, then the pairwise distance matrix is symmetric
        with zero-elements on the diagonal. However, we compute the median
        across all elements of this matrix as this seems to also have been
        handled that way in the original
        `implementation <https://github.com/thjashin/spectral-stein-grad/blob/\
e40133806e5d8a9c94b89c77529cd4a892a09ef1/estimator/base.py#L57>`__.

    Args:
        x1 (torch.Tensor): The samples. The first dimension corresponds
            to the number of samples ``N1``.
        x2 (torch.Tensor or None): The extra samples. The first dimension
            corresponds to the number of samples ``N2``.
        max_m (int): The maximum number of estimation samples to be used.
            Is only relevant if ``x2`` is ``None``.

    Returns:
        (float): The kernel width.
    """
    if x2 is not None:
        dist_x1 = torch.cdist(x1[None, :,:],
                              x1[None, :,:], p=2)[0].flatten()
        # If there are extra samples, we compute their pairwise distance,
        # as well as its distance with previous samples.
        dist_x2 = torch.cdist(x2[None, :,:],
                              x2[None, :,:], p=2)[0].flatten()
        dist_x1x2 = torch.cdist(x1[None, :,:],
                                x2[None, :,:], p=2)[0].flatten()
        pairwise_dist = torch.cat((dist_x1, dist_x2))
        pairwise_dist = torch.cat((pairwise_dist, dist_x1x2))

        # Because a pairwise distance matrix would be of size (N1+N2)**2,
        # we need to add twice the distances between x1 and x2 samples
        # in order to correctly obtain the median. I.e., we use all elements of
        # all 4 sub matrices
        #    x1-x1 | x1-x2
        #    -------------
        #    x2-x1 | x2-x2
        # Note, correctly we should apply `torch.tril_indices` to `dist_x1` and
        # and `dist_x2`.
        pairwise_dist = torch.cat((pairwise_dist, dist_x1x2))
    else: 
        # Set the maximum number of samples.
        if max_m != -1:
            x1 = x1[:max_m, :]
        pairwise_dist = torch.cdist(x1[None, :,:],
                                    x1[None, :,:], p=2)[0].flatten()

    # Compute the median of the distances.
    return pairwise_dist.median().item()

def generate_weight_sample(config, shared, device, hnet, theta, num_samples=1,
                           ret_format='squeezed'):
    """Generate weight samples from latent embedding.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions.
        device: Torch device.
        hnet: The hypernetwork.
        theta: The weights passed to ``hnet`` when drawing samples (can be
            ``None`` if internally maintaned weights of ``hnet`` should be 
            used).
        num_samples (int): How many samples should be generated.
        ret_format (str): See argument ``ret_format`` of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
    """
    # Sample z from a Gaussian distribution
    z = torch.normal(torch.zeros(num_samples, shared.noise_dim),
                     config.latent_std).to(device)

    # Compute the weights (i.e. h(z,\theta)).
    weights = hnet.forward(uncond_input=z, weights=theta, ret_format=ret_format)

    return weights


def estimate_entropy_grad(config, shared, device, logger, hnet, hnet_theta,
                          prior_theta=None):
    r"""Estimate the gradient of the entropy term of the ELBO via SSGE.

    This function uses SSGE to estimate the gradient of the entropy of the
    implicit distribution that is present in the ELBO. Specifically, this
    function estimates

    .. math::

        \nabla_{\theta} \mathbb{E}_{q_\theta(W)} [ \log q_\theta(W) ]

    where :math:`q_\theta(W)` is the implicit distribution, and :math:`\theta`
    are, in our case, the parameters of the hypernetwork. Note, while the ELBO
    contains the entropy, we consider the negative entropy (without minus sign)
    as we minimize the negative ELBO.

    This expression is equal to

    .. math::

        \mathbb{E}_{p(Z)} \big[ \nabla_{W} (\log q_\theta(W)) \nabla_{\theta} \
        h(Z, \theta) \big]

    where :math:`p(Z)` is a standard gaussian and :math:`h(Z, \theta)` is the
    function performed by the hypernetwork.

    Note that the second term
    :math:`\nabla_{\theta} h(Z, \theta)` within the expected value can be
    obtained using autograd while the first term
    :math:`\nabla_{W} (\log q_\theta(W))` needs to be estimated using the SSGE
    algorithm.

    The number of samples used for these estimations is denoted by ``N``.

    **Implementation and derivation details**

    This section should help developers to better understand the internals of
    the function. We first derive the formula above:

    .. math::

        \nabla_{\theta} \mathbb{E}_{q_\theta(W)} [ \log q_\theta(W) ] &=
        \mathbb{E}_{p(Z)} \big[ \nabla_{\theta}
        \log q_\theta \big( h(Z, \theta) \big) \big] \\
        &= \mathbb{E}_{p(Z)} \big[ \nabla_{\theta}
        \log q_\theta \big( h(Z, \hat{\theta}) \big)
        \Big\lvert_{\hat{\theta}=\theta} \big] +
        \mathbb{E}_{p(Z)} \big[ \nabla_{\theta}
        \log q_{\hat{\theta}} \big( h(Z, \theta) \big)
        \Big\lvert_{\hat{\theta}=\theta} \big] \\
        &= \mathbb{E}_{q_\theta(W)} \big[ \nabla_{\theta}
        \log q_\theta (W) \big] +
        \mathbb{E}_{p(Z)} \big[ \big( \nabla_{\theta} h(Z, \theta) \big)
        \big( \nabla_W \log q_{\theta} (W) \big)
        \Big\lvert_{W = h(Z, \theta)} \big] \\
        &= \nabla_{\theta} \mathbb{E}_{q_\theta(W)} [1] +
        \mathbb{E}_{p(Z)} \big[ \big( \nabla_{\theta} h(Z, \theta) \big)
        \big( \nabla_W \log q_{\theta} (W) \big)
        \Big\lvert_{W = h(Z, \theta)} \big] \\

    The first term evaluates to zero and the second term is approximated via
    a Monte-Carlo estimate.

    .. math::

        \nabla_{\theta} \mathbb{E}_{q_\theta(W)} [ \log q_\theta(W) ] \approx
        \frac{1}{N} \sum_{n=1}^N
        \big( \nabla_{\theta} h(z_n, \theta) \big)
        \big( \nabla_W \log q_{\theta} (W) \big) \Big\lvert_{W = h(z_n, \theta)}

    The hypernetwork derivative is a matrix
    :math:`\nabla_{\theta} h(z_n, \theta) \in \mathbb{R}^{n_\theta \times n_W}`
    and the score function is a row vector
    :math:`\nabla_W \log q_{\theta} (W) \in \mathbb{R}^{n_W \times 1}`.

    Internally, we work with efficient parallized matrix operations rather
    than computing the sum post-hoc. To understand this, assume :math:`n_W = 1`,
    such that the batch of score estimates can be considered a row vector
    :math:`s_\theta \in \mathbb{R}^{N \times 1}` similarly the batch of hypernet
    gradients becomes a matrix
    :math:`\nabla_{\theta} h \in \mathbb{R}^{n_\theta \times N}`. Performing
    the matrix-vector product here leads to the desired result that terms are
    first multiplied and then summed. In practice, we work with tensors
    :math:`s_\theta \in \mathbb{R}^{N \times n_W}` and
    :math:`\nabla_{\theta} h \in \mathbb{R}^{n_\theta \times N \times n_W}`,
    but the procedure is analoguous.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions.
        device: Torch device.
        logger: The logger.
        hnet: The hypernetwork, representing an implicit distribution from
            which to sample weights.
        hnet_theta: The weights passed to ``hnet`` when drawing samples (can be
            ``None`` if internally maintaned weights of ``hnet`` should be 
            used).
        prior_theta: A distribution, that represents an implicit prior. Only
            relevant when doing prior-focused CL in order to generate the
            samples for computing the gradient of the cross-entropy in the ELBO.

    Returns:
        (list): The estimate of the gradient of entropy term of the ELBO with
            respect to the hypernetwork parameters. It has same length and
            dimensionalities as ``hnet_theta``.
    """
    # Determine whether we are estimating the cross-entropy or entropy term.
    # For the cross-entropy term, parameters for the prior are provided.
    is_cross_entropy_term = False
    if prior_theta is not None:
       is_cross_entropy_term = True

    # Detach the hypernetwork parameters from outer optimization.
    if hnet_theta is None:
        # Note, we assume that `hnet` has no conditional parameters, i.e., that
        # hnet.internal_params is the same as calling hnet.unconditional_params.
        assert hnet.internal_params is not None
        hnet_theta = hnet.internal_params
    theta = []
    for hti in hnet_theta:
        hti = hti.clone().detach()
        hti.requires_grad = True
        theta.append(hti)

    ### Get the weight samples from the current theta (i.e. from the posterior).
    samples_post = generate_weight_sample(config, shared, device, hnet, theta,
        num_samples=config.train_sample_size, ret_format='flattened')

    ### Get the weight samples from the prior, if necessary.
    samples_prior = None
    if is_cross_entropy_term:
        # Samples will be used to estimate the gradient in prior-focused CL.
        samples_prior = generate_weight_sample(config, shared, device, hnet,
            prior_theta, num_samples=config.ssge_sample_size,
            ret_format='flattened')

    ### Compute the first term using SSGE
    samples_eval = samples_post # the samples where to evaluate the function
    samples_est = None # the samples for estimating the score function
    samples_est_sup = None # the supplementary samples for estimation

    with torch.no_grad():

        # Determine the samples to use for gradient estimation.
        if is_cross_entropy_term:
            # In this case, the score function is estimated with samples from
            # the prior, and evaluated samples from the posterior.
            samples_est = samples_prior
        else:
            # Else, use samples from the posterior to estimate and evaluate.
            # For memory reasons, we reuse the samples that were generated for
            # evaluation and use them for estimation as well. If the number of
            # samples for estimation samples is smaller than that for
            # evaluation we select a subset of them. Else, we generate a set of
            # supplementary samples ``samples_est_sup``.
            if config.ssge_sample_size > samples_eval.shape[0]:
                num_sup = config.ssge_sample_size - samples_eval.shape[0]
                samples_est_sup = generate_weight_sample(config, shared, device,
                    hnet, theta, num_samples=num_sup, ret_format='flattened')

        # Extract kernel width
        if config.heuristic_kernel:
            if is_cross_entropy_term:
                rbf_kernel_width = get_heuristic_kernel_width(samples_est)
            else:
                rbf_kernel_width = get_heuristic_kernel_width(samples_eval,
                    x2=samples_est_sup, max_m=config.ssge_sample_size)
        else:
            rbf_kernel_width = config.rbf_kernel_width

        ssge = SSGE(kernel_width=rbf_kernel_width)
        # Get score function estimate using SSGE.
        gradient = ssge.estimate_gradient(samples_eval,
                                          x_est=samples_est,
                                          x_sup=samples_est_sup,
                                          max_m=config.ssge_sample_size,
                                          J=config.num_ssge_eigenvals,
                                          J_thr=config.thr_ssge_eigenvals)

    ### Compute the second term using autograd
    theta_grad = [torch.zeros_like(thetai) for thetai in theta]
    for i, thetai in enumerate(theta):
        # Compute the gradient of the weights wrt theta. Note, multiplication
        # with score function estimate and sum across samples happens
        # implicitly.
        retain_graph = False if i == len(theta) - 1 else True
        theta_grad[i] += grad(outputs=samples_post, inputs=thetai, \
            grad_outputs=gradient, retain_graph=retain_graph)[0]
        # Multiply accumulated gradient by 1/N (factor from MC estimate).
        theta_grad[i] /= samples_post.shape[0]

    return theta_grad


def get_prior_loss_term(config, shared, batch_size, device, hnet, theta_current,
                        dist_prior):
    r"""Get the cross-entropy term within the ELBO.

    This functions returns the cross-entropy term from the (negative) ELBO
    performing a Monte-Carlo estimate:

    .. math::

        - \mathbb{E}_{q_\theta(W)} [\log p(W)]

    where :math:`p(W)` is the prior distribution and :math:`q_\theta(W)` is
    (for instance) the implicit variational approximation, which we can sample
    from. Note, the prior :math:`p(W)` is assumed to be explicit with an
    analytically tractable density.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions.
        batch_size (int): How many samples should be used for estimating the
            expected value.
        device: PyTorch device.
        hnet: The hypernetwork, representing an implicit distribution from
            which to sample weights. Is used to draw samples from the current
            implicit distribution ``theta_current`` (which may be ``None`` if
            internal weights should be selected).
        theta_current: The weights passed to ``hnet`` when drawing samples from
            the current implicit distribution (can be ``None`` if internally
            maintained weights of ``hnet`` should be used).
        dist_prior: A distribution, that represents a prior. Can be explicit, or
            implicit.

    Returns:
        (float): The prior loss component.
    """
    w_samples = generate_weight_sample(config, shared, device, hnet,
        theta_current, num_samples=batch_size, ret_format='flattened')

    # Compute the log probabilities of the current samples under the prior.
    log_prob = dist_prior.log_prob(w_samples).sum(dim=1)

    # Compute the average across the samples to estimate expectation.
    mean_log_prob = log_prob.mean()

    return - mean_log_prob


def get_prior_grad_term(config, shared, device, logger, batch_size, hnet,
                        theta_current, prior_theta):
    r"""Get the gradient of the cross-entropy within the ELBO.

    This function returns the following term from the gradient of the
    (negative) ELBO

    .. math::

        - \nabla_{\theta} \mathbb{E}_{q_\theta(W)} (\log p(W)) 

    where :math:`p(W)` is the prior distribution and :math:`q_\theta(W)` is the
    implicit distribution representing the variational approximation. This
    calculation of the gradient of an implicit prior with respect to
    :math:`\theta` is useful, for example, when doing Variational Continual
    Learning using as prior an implicit distribtion parametrized by a
    hypernetwork.

    The above term is equal to 

    .. math::

        \mathbb{E}_{p(Z)} \big[ \nabla_w \log p(W) \nabla_{\theta} \
        h(Z, \theta) \big]

    where :math:`w = h(Z, \theta)` is the function performed by the
    hypernetwork. This function therefore requires to utilize SSGE to estimate
    the score function of the prior distribution.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions.
        device: PyTorch device.
        logger: The logger.
        batch_size (int): How many samples should be used for evaluating the
            expectation.
        hnet: The hypernetwork, representing an implicit distribution from
            which to sample weights. Is used to draw samples from the current
            implicit distribution ``theta_current`` (which may be ``None`` if
            internal weights should be selected).
        theta_current: The weights passed to ``hnet`` when drawing samples from
            the current implicit distribution (can be ``None`` if internally 
            maintaned weights of ``hnet`` should be used).
        prior_theta: Same as ``theta_current`` but for the last implicit
            distribution, i.e. the current prior in a prior-focused approach.

    Returns:
        (float): The prior grad component.
    """
    cross_entropy_ssge_grad = estimate_entropy_grad(config, shared, device,
        logger, hnet, theta_current, prior_theta=prior_theta)

    return cross_entropy_ssge_grad