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
# @title          :probabilistic/gauss_mnet_interface.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/07/2020
# @version        :1.0
# @python_version :3.6.9
"""
A main network wrapper that handles Gaussian network weights
------------------------------------------------------------

The module :mod:`probabilistic.gauss_mnet_interface` contains a wrapper for
class :class:`mnets.mnet_interface.MainNetInterface` is implemented, that
duplicates internal weights to represent mean :math:`\mu` and variance
:math:`\sigma^2`. Additionally, the :meth:`GaussianBNNWrapper.forward` method
takes care of sampling from the Gaussian distribution.
"""
import numpy as np
import torch
import torch.nn as nn
from warnings import warn

from mnets.mnet_interface import MainNetInterface
from probabilistic import prob_utils as putils
from probabilistic.gauss_mlp import GaussianMLP

class GaussianBNNWrapper(nn.Module, MainNetInterface):
    r"""Convert a Main Network into a BNN with Gaussian weight distribution.

    This class takes an existing main network object and dublicates its weights,
    such that they can be interpreted as means :math:`\mu` and
    `squashed-variances` :math:`\rho` (the real valued parameter :math:`\rho`
    will be transformed into a positive real value that represents the std as
    described below) of a Gaussian distribution. Note, the weight distribution
    is fully-factorized, i.e., correlations between weights cannot be learned.

    Weights :math:`\rho` are translated into a standard deviatation
    :math:`\sigma` via a softplus function as recommended
    `here <https://arxiv.org/pdf/1505.05424.pdf>`__.

    For initialization, we follow the implementation from this
    `BbB example code`_ and initialize mean values uniformly via
    :math:`\mathcal{U}(-0.1, 0.1)` and :math:`\rho` values via
    :math:`\mathcal{U}(-3, -2)`.

    Note:
        If `is_radial` is ``True``, the network is not a BNN with a Gaussian
        weight distribution anymore, but a Radial BNN where the weights are
        sampled according to

        .. math::

            \mathbf{w} = \mathbf{\mu} + \mathbf{\sigma} \odot \
            \frac{\epsilon_{MFVI}}{\mid\mid \epsilon_{MFVI} \mid\mid} \cdot r

        as described in `Farquhar et al. <https://arxiv.org/abs/1907.00865>`__,
        where :math:`\epsilon_{MFVI} \sim \mathcal{N} (0, \mathbf{I})` and
        :math:`r \sim \mathcal{N} (0, 1)`.

    Attributes:
        param_shapes_meta (list): See documentation of attribute
            :attr:`mnets.mnet_interface.MainNetInterface.param_shapes_meta`.
            In addition, dictionaries were added the key ``'dist_param'`` with
            values ``'mean'`` or ``'rho'``.
        mean (torch.nn.ParameterList): Any internally stored parameters that
            represent the mean :math:`\mu` parameters of the Gaussian weight
            distribution. These correspond to the internally stored weights
            (see :attr:`mnets.mnet_interface.MainNetInterface.internal_params`)
            of the original main network ``mnet`` passed to the constructor.
            Note, this attribute together with the attribute :attr:`rho` will
            make up this class its attribute
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params`.
        rho (torch.nn.ParameterList): Any internally stored parameters that
            represent the encoded std :math:`\sigma` parameters of the
            Gaussian weight distribution.
        rho_offset_applied (bool): Whether constructor argument
            ``apply_rho_offset`` was activated.
        orig_param_shapes (list): The attribute
            :attr:`mnets.mnet_interface.MainNetInterface.param_shapes` of the
            original main network ``mnet`` provided to the constructor. I.e.,
            the shapes of either the means or variances.
        logvar_encoding (bool): Value of constructor argument
            ``logvar_encoding``.

    Args:
        mnet (mnets.mnet_interface.MainNetInterface): An existing network.
            Its internal weights (attribute
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params`) as
            well as shape attributes (such as
            :attr:`mnets.mnet_interface.MainNetInterface.param_shapes` and
            :attr:`mnets.mnet_interface.MainNetInterface.hyper_shapes_learned`)
            are duplicated.
        no_mean_reinit (bool): If ``True``, the original internal weights of
            ``mnet`` are not reinitialized. Instead, the current values of
            these weights are kept (for instance, the standard init that was
            used when ``mnet`` was created).
        logvar_encoding (bool): If ``True``, a log-variance encoding for
            :math:`\rho` is used. I.e., instead of extracting the std like this
            :math:`\sigma = \text{softplus}(\rho)`, we retrieve it via
            :math:`\sigma = \exp \frac{1}{2} \rho`. Hence,
            :math:`\rho \equiv \log \sigma^2`.
        apply_rho_offset (bool): A constant offset (of ``-2.5``) will be applied
            to weights corresponding to encoded variances :math:`\rho` if
            ``True``. Therefore, the initialization of internally maintained
            :math:`\rho` weights (see attribute :attr:`rho`) is changed from
            :math:`\mathcal{U}(-3, -2)` to :math:`\mathcal{U}(-0.5, 0.5)`, i.e.,
            zero-mean.

            Note, this argument results in computational overhead, but provides
            benefits when using hypernetworks. For instance, a hypernetwork is
            often initialized to have zero-mean outputs. Thus, if we add this
            constant (negative) offset to the hypernet output, we simulate a
            negative output mean of the hypernet.
        is_radial (bool): Indicates whether the instantiated network should be a
            Radial BNN.

    .. _BbB example code:
        https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Bayes_By_Backprop/model.py#L38
    """
    def __init__(self, mnet, no_mean_reinit=False, logvar_encoding=False,
                 apply_rho_offset=False, is_radial=False):
        # FIXME find a way using super to handle multiple inheritance.
        #super(GaussianBNNWrapper, self).__init__()
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        assert isinstance(mnet, MainNetInterface)
        assert not isinstance(mnet, GaussianBNNWrapper)

        if is_radial:
            print('Converting network into BNN with radial weight '+
                  'distribution ...')
        else:
            print('Converting network into BNN with diagonal Gaussian weight '+
                  'distribution ...')

        self._mnet = mnet
        self._logvar_encoding = logvar_encoding
        self._apply_rho_offset = apply_rho_offset
        self._rho_offset = -2.5
        self._is_radial = is_radial

        # Take over attributes of `mnet` and modify them if necessary.
        self._mean_params = None
        self._rho_params = None
        if mnet.internal_params is not None:
            self._mean_params = mnet.internal_params
            self._rho_params = nn.ParameterList()

            for p in self._mean_params:
                self._rho_params.append(nn.Parameter(torch.Tensor(p.size()),
                                                     requires_grad=True))

            # Initialize weights.
            if not no_mean_reinit:
                for p in self._mean_params:
                    p.data.uniform_(-0.1, 0.1)

            for p in self._rho_params:
                if apply_rho_offset:
                    # We will subtract 2.5 from `rho` in the forward method.
                    #p.data.uniform_(-0.5, 0.5)
                    p.data.uniform_(-3-self._rho_offset, -2-self._rho_offset)
                else:
                    p.data.uniform_(-3, -2)

            self._internal_params = nn.ParameterList()
            for p in self._mean_params:
                self._internal_params.append(p)
            for p in self._rho_params:
                self._internal_params.append(p)

        # Simply duplicate `param_shapes` and `hyper_shapes_learned`.
        self._param_shapes = mnet.param_shapes + mnet.param_shapes
        if mnet._param_shapes_meta is not None:
            self._param_shapes_meta = []
            old_wlen = 0  if self.internal_params is None \
                else len(mnet.internal_params)
            for dd in mnet._param_shapes_meta:
                dd['dist_param'] = 'mean'
                self._param_shapes_meta.append(dd)

            for dd_old in mnet._param_shapes_meta:
                dd = dict(dd_old)
                dd['index'] += old_wlen
                dd['dist_param'] = 'rho'
                self._param_shapes_meta.append(dd)

        if mnet._hyper_shapes_learned is not None:
            self._hyper_shapes_learned = mnet._hyper_shapes_learned + \
                mnet._hyper_shapes_learned
        if mnet._hyper_shapes_learned_ref is not None:
            self._hyper_shapes_learned_ref = \
                list(mnet._hyper_shapes_learned_ref)
            old_plen = len(mnet.param_shapes)
            for ii in mnet._hyper_shapes_learned_ref:
                self._hyper_shapes_learned_ref.append(ii + old_plen)

        self._hyper_shapes_distilled = mnet._hyper_shapes_distilled
        if self._hyper_shapes_distilled is not None:
            # In general, that shouldn't be an issue, as those distilled values
            # are just things like batchnorm stats. But it might be good to
            # inform the user about the fact that we are not considering this
            # attribute as special.
            warn('Class "GaussianBNNWrapper" doesn\'t modify the existing ' +
                 'attribute "hyper_shapes_distilled".')

        self._has_bias = mnet._has_bias
        self._has_fc_out = mnet._has_fc_out
        # Note, it's still true that the last two entries of
        # `hyper_shapes_learned` are belonging to the output layer. But those
        # are only the variance weights. So, we would forget about the mean
        # weights when setting this quantitiy to true.
        self._mask_fc_out = False #mnet._mask_fc_out
        self._has_linear_out = mnet._has_linear_out

        # We don't modify the following attributed, but generate warnings
        # when using them.
        self._layer_weight_tensors = mnet._layer_weight_tensors
        self._layer_bias_vectors = mnet._layer_bias_vectors
        self._batchnorm_layers = mnet._batchnorm_layers
        self._context_mod_layers = mnet._context_mod_layers

        self._is_properly_setup(check_has_bias=False)

    @property
    def layer_weight_tensors(self):
        """Getter for read-only attribute
        :attr:`mnets.mnet_interface.MainNetInterface.layer_weight_tensors`.

        Returns:
            A list (e.g., an instance of class :class:`torch.nn.ParameterList`).
        """
        warn('Class "GaussianBNNWrapper" didn\'t modify the attribute ' +
             '"layer_weight_tensors", such that the contained weights only ' +
             'represent mean parameters.')

        #return super(MainNetInterface, self).layer_weight_tensors
        return super().layer_weight_tensors

    @property
    def layer_bias_vectors(self):
        """Getter for read-only attribute 
        :attr:`mnets.mnet_interface.MainNetInterface.layer_bias_vectors`.

        Returns:
            A list (e.g., an instance of class :class:`torch.nn.ParameterList`).
        """
        warn('Class "GaussianBNNWrapper" didn\'t modify the attribute ' +
             '"layer_bias_vectors", such that the contained weights only ' +
             'represent mean parameters.')

        #return super(MainNetInterface, self).layer_bias_vectors
        return super().layer_bias_vectors

    @property
    def batchnorm_layers(self):
        """Getter for read-only attribute 
        :attr:`mnets.mnet_interface.MainNetInterface.batchnorm_layers`.

        Returns:
            (:class:`torch.nn.ModuleList`): A list of
            :class:`utils.batchnorm_layer.BatchNormLayer` instances, if batch
            normalization is used.
        """
        # FIXME 
        #warn('Class "GaussianBNNWrapper" didn\'t modify the attribute ' +
        #     '"batchnorm_layers", such that the contained weights only ' +
        #     'represent mean parameters.')

        #return super(MainNetInterface, self).batchnorm_layers
        return super().batchnorm_layers

    @property
    def context_mod_layers(self):
        """Getter for read-only attribute 
        :attr:`mnets.mnet_interface.MainNetInterface.context_mod_layers`.

        Returns:
            (:class:`torch.nn.ModuleList`): A list of
            :class:`utils.context_mod_layer.ContextModLayer` instances, if these
            layers are in use.
        """
        warn('Class "GaussianBNNWrapper" didn\'t modify the attribute ' +
             '"context_mod_layers", such that the contained weights only ' +
             'represent mean parameters.')

        #return super(MainNetInterface, self).context_mod_layers
        return super().context_mod_layers

    @property
    def mean(self):
        """Getter for read-only attribute :attr:`mean`.

        Returns:
            (torch.nn.ParameterList): List of internally stored mean parameters
            of ``None`` if no parameters are maintained internally.
        """
        return self._mean_params

    @property
    def rho(self):
        r"""Getter for read-only attribute :attr:`rho`.

        .. note::

            This getter does not apply any offset correction (see constructor
            argument ``apply_rho_offset``). Instead, it returns the parameters
            directly (for instance, to pass them to an optimizer).

            Offset correction is applied by method :meth:`extract_mean_and_rho`.

        Returns:
            (torch.nn.ParameterList): List of internally stored encoded std
            parameters (:math:`\rho`) of ``None`` if no parameters are
            maintained internally.
        """
        return self._rho_params

    @property
    def rho_offset_applied(self):
        r"""Getter for read-only attribute :attr:`rho_offset_applied`.

        Returns:
            (bool): Whether an offset for :math:`\rho` is applied.
        """
        return self._apply_rho_offset

    @property
    def orig_param_shapes(self):
        """Getter for read-only attribute :attr:`orig_param_shapes`.

        Returns:
            (list): List of lists of integers.
        """
        return self._mnet.param_shapes

    @property
    def logvar_encoding(self):
        """Getter for read-only attribute :attr:`logvar_encoding`.

        Returns:
            (bool): Whether a log-variance encoding is used.
        """
        return self._logvar_encoding

    def distillation_targets(self):
        """Targets to be distilled after training.

        This method simply calls the corresponding ``distillation_targets``
        method of the ``mnet`` provided to the constructor.
        """
        # FIXME Provide option to disable warning if programmer knows what he
        # is doing.
        warn('Class "GaussianBNNWrapper" doesn\'t modify the method ' +
             '"distillation_targets" of the source main network.')

        self._mnet.distillation_targets()

    def forward(self, x, weights=None, distilled_params=None, condition=None,
                ret_sample=False, mean_only=False, extracted_mean=None,
                extracted_rho=None, sample=None, disable_lrt=False,
                rand_state=None, **kwargs):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        This method uses the current mean (:attr:`mean`) and std (extracted from
        :attr:`rho`) to draw a sample from the correponding Gaussian weight
        distribution. This weight sample is then passed to the underlying
        main network (see constructor argument ``mnet``) in order to make a
        prediction using :math:`x`.

        Note:
            This method automatically treats instances of class
            :class:`probabilistic.gauss_mlp.GaussianMLP` properly, i.e., making
            sure the local reparametrization trick is used. Note, argument
            ``ret_sample`` may not be ``True`` for such instances.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            x: This argument is simply passed to the ``forward`` method of the
                underlying ``mnet``.
            weights (list, optional): A list of parameters. The list must either
                comply with the shapes in attribute :attr:`mnets.\
mnet_interface.MainNetInterface.hyper_shapes_learned`
                or the shapes in attribute
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.

                .. note::

                    Some main networks may allow more complicated ways of
                    passing weights than just via lists (e.g., via
                    dictionaries). For now, we do not support such options.
            distilled_params (optional): This argument is simply passed to the
                ``forward`` method of the underlying ``mnet``.
            condition (optional): This argument is simply passed to the
                ``forward`` method of the underlying ``mnet``.
            ret_sample (bool, optional): Whether the sample that was drawn from
                the weight distribution should be returned as well.
            mean_only (bool, optional): Rather than drawing random sample, the
                mean (:attr:`mean`) would be used.
            extracted_mean (optional): The first return value of method
                :meth:`extract_mean_and_rho`. Can be passed for compuational
                efficiency reasons (if mean- and rho-values where already
                computed). Otherwise, this method will simply call
                :meth:`extract_mean_and_rho`. Passed ``weights`` will be ignored
                if this argument is specified.

                Requires that ``extracted_rho`` is set as well.
            extracted_rho (optional): The second return value of method
                :meth:`extract_mean_and_rho`. See description of
                ``extracted_mean``.
            sample (optional): A weight sample that should be used (and thus
                passed to the underlying ``mnet``).

                For instance, the return value ``sample`` from a previous call
                (see ``ret_sample``) could be passed here to reuse the same
                sample.

                .. note::

                    Specifying this argument means that arguments ``weights``,
                    ``mean_only``, ``extracted_mean``, ``extracted_rho`` and
                    ``rand_state`` are ignored!
            disable_lrt (bool): If ``True``, the local-reparametrization trick
                will be disabled. Instead, a single weight sample is used to
                process the inputs. Option has no effect if the underlying
                network does not support the local-reparametrization trick.
            rand_state (torch.Generator, optional): This generator would be used
                to realize the reparametrization trick (i.e., the weight
                sampling), if specified. This way, the behavior of the forward
                pass is easily reproducible (except if other sources of noise
                are involved in the underlying ``mnet`` such as dropout in train
                mode).
            **kwargs: Additional keyword arguments that are passed to the
                ``forward`` method of the wrapped (underlying) network.

        Returns:
            The output :math:`y` of the network ``mnet`` using a weight sample
            from :math:`\mathcal{N}(\mu, \sigma^2)`.

            If ``ret_sample`` is ``True``, the method returns a tuple
            containing:

            - **y**: The output of the ``forward`` method from the underlying
              ``mnet``.
            - **sample**: The weight sample used to produce ``y``.
        """
        # TODO we may wanna add `**kwargs` to the argument list, in case the
        # underlying main network can take more parameters.
        assert (extracted_mean is None and extracted_rho is None) or \
            (extracted_mean is not None and extracted_rho is not None)

        if sample is not None  and mean_only:
            warn('Argument "mean_only" is ignored since "sample" is provided.')
        if sample is not None  and extracted_mean is not None:
            warn('Argument "extracted_mean" is ignored since "sample" is ' +
                 'provided.')
        if sample is not None  and weights is not None:
            warn('Argument "weights" is ignored since "sample" is provided.')
        if extracted_mean is not None  and weights is not None:
            warn('Argument "weights" is ignored since "extracted_mean" is ' +
                 'provided.')

        # The following warning should be obvious.
        #if rand_state is not None  and (weights is not sample or mean_only):
        #    warn('Argument "rand_state" is ignored since "sample" is ' +
        #         'provided pr "mean_only" was set.')

        if sample is None and extracted_mean is None:
            mean, rho = self.extract_mean_and_rho(weights=weights)
        elif sample is None:
            assert len(extracted_mean) == len(extracted_rho) and \
                len(extracted_mean) == len(self._mnet.param_shapes)
            # We should additionally assert that all shapes comply with
            # `self._mnet.param_shapes`.

            mean = extracted_mean
            rho = extracted_rho

        if isinstance(self._mnet, GaussianMLP) and not disable_lrt:
            if sample is not None:
                raise ValueError('Local reparametrization trick is used. ' +
                                 'Hence, activations are samples and a ' +
                                 'weight "sample" can\'t be utilized.')

            if ret_sample:
                raise ValueError('Argument "ret_sample" is not applicable, ' +
                                 'when using the local reparametrization ' +
                                 'trick.')

            y = self._mnet.forward(x, mean, rho,
                logvar_enc=self._logvar_encoding, mean_only=mean_only,
                rand_state=rand_state, **kwargs)

        else:

            if sample is not None:
                sample = sample # We already have our weight sample.
            elif mean_only:
                sample = mean
            else:
                # TODO might be good to also give the option to the user of
                # having a different weight sample per sample in the minibatch.
                sample = putils.decode_and_sample_diag_gauss(mean, rho,
                    logvar_enc=self._logvar_encoding, generator=rand_state,
                    is_radial=self._is_radial)

            if isinstance(self._mnet, GaussianMLP):
                assert disable_lrt
                y = self._mnet.forward(x, None, None, sample=sample, **kwargs)
            else:
                y = self._mnet.forward(x, weights=sample,
                    distilled_params=distilled_params, condition=condition,
                    **kwargs)

        if ret_sample:
            return y, sample

        return y

    def extract_mean_and_rho(self, weights=None):
        """Extract all means and rho-values from internally maintained and/or
        externally provided weights.

        .. note::

            Note, you should always work with the means and rho-values provided
            by this method. This method knows how to extract these values
            (for instance, from a hypernetwork output) and might apply further
            manipulations, such as mean-corrections (e.g., see constructor
            argument ``apply_rho_offset``).

        Args:
            weights: See docstring of method :meth:`forward`.

        Returns:
            (tuple): A tuple containing:

                - **mean**: All mean values corresponding to ``param_shapes``
                  of the underlying ``mnet``.
                - **rho**: All rho values corresponding to ``param_shapes``
                  of the underlying ``mnet``. Note, a mean-correction might
                  have been performed (see constructor argument
                  ``apply_rho_offset``).
        """
        if weights is None:
            if self.internal_params is None:
                raise ValueError('No internal weights. Parameter "weights" ' +
                                 'may not be None.')
            mean = self._mean_params
            rho = self._rho_params
        else:
            if self.hyper_shapes_learned is not None and \
                    len(self.hyper_shapes_learned) != \
                        len(self.param_shapes) and \
                    len(weights) == len(self.hyper_shapes_learned):
                for i, s in enumerate(self.hyper_shapes_learned):
                    assert np.all(np.equal(s, list(weights[i].shape)))

                #mean = self._mean_params
                #rho = self._rho_params

                # FIXME We have the problem, that we don't know how the
                # underlying `mnet` would reassemble the passed and internal
                # weights. We could use attributes such as
                # `hyper_shapes_learned_ref` to figure it out (if the attribute
                # is specified). But I am not sure whether this is the best
                # option, given that we still can't deal with more complicated
                # passing options (such as passing `weights` as a dict).
                # Therefore, it might be better to provide a private abstract
                # method to the main net interface, that parses passed weights
                # (note, such method must allow the passing of internal weights
                # as optional argument, in case we want to pass our internal
                # `rho` parameters and "half" of the passed `weights`
                # parameter).
                raise NotImplementedError('Partial passing of weights not ' +
                    'yet implemented for this class. You either have to pass ' +
                    'all weights or None.')
            else:
                assert len(weights) == len(self.param_shapes)
                for i, s in enumerate(self.param_shapes):
                    assert np.all(np.equal(s, list(weights[i].shape)))

                mean = weights[:(len(weights) // 2)]
                rho = weights[(len(weights) // 2):]

        if self._apply_rho_offset:
            rho = [r + self._rho_offset for r in rho]

        return mean, rho

    def get_output_weight_mask(self, out_inds=None, device=None):
        """Get masks to select output weights.

        See docstring of overwritten super method
        :meth:`mnets.mnet_interface.MainNetInterface.get_output_weight_mask`.
        """
        ret = self._mnet.get_output_weight_mask(out_inds=out_inds,
                                                device=device)

        assert len(ret) == len(self.param_shapes) // 2
        return ret + ret

    def sample_weights(self, weights=None, mean_only=False, extracted_mean=None,
                       extracted_rho=None, rand_state=None):
        """Sample and return weights from the BNN, e.g. to be reused in a
        call to :meth:`forward`.

        Args:
            weights (list, optional): A list of parameters. The list must either
                comply with the shapes in attribute :attr:`mnets.\
mnet_interface.MainNetInterface.hyper_shapes_learned`
                or the shapes in attribute
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.
            mean_only (bool, optional): Rather than drawing random sample, the
                mean (:attr:`mean`) would be used.
            extracted_mean (optional): The first return value of method
                :meth:`extract_mean_and_rho`. Can be passed for compuational
                efficiency reasons (if mean- and rho-values where already
                computed). Otherwise, this method will simply call
                :meth:`extract_mean_and_rho`. Passed ``weights`` will be ignored
                if this argument is specified.

                Requires that ``extracted_rho`` is set as well.
            extracted_rho (optional): The second return value of method
                :meth:`extract_mean_and_rho`. See description of
                ``extracted_mean``.
            rand_state (torch.Generator, optional): This generator would be used
                to realize the reparametrization trick (i.e., the weight
                sampling), if specified. This way, the behavior of the forward
                pass is easily reproducible (except if other sources of noise
                are involved in the underlying ``mnet`` such as dropout in train
                mode).

        Returns:
            (list): A sample from the factorized weight distribution.
        """
        if extracted_mean is not None and weights is not None:
            warn('Argument "weights" is ignored since "extracted_mean" is ' +
                 'provided.')

        if extracted_mean is None:
            mean, rho = self.extract_mean_and_rho(weights=weights)
        else:
            assert len(extracted_mean) == len(extracted_rho) and \
                   len(extracted_mean) == len(self._mnet.param_shapes)
            # We should additionally assert that all shapes comply with
            # `self._mnet.param_shapes`.

            mean = extracted_mean
            rho = extracted_rho

        if mean_only:
            sample = mean
        else:
            sample = putils.decode_and_sample_diag_gauss(mean, rho,
                                                         logvar_enc=self._logvar_encoding,
                                                         generator=rand_state,
                                                         is_radial=self._is_radial)

        return sample

if __name__ == '__main__':
    pass


