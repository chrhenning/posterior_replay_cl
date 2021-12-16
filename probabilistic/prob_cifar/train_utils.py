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
# @title          :probabilistic/prob_cifar/train_utils.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/30/2020
# @version        :1.0
# @python_version :3.6.9
"""
Training utilities
------------------

A collection of helper functions for training scripts of this subpackage.
"""
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn import functional as F
from warnings import warn

from data.special.permuted_mnist import PermutedMNIST
from hnets.chunked_mlp_hnet import ChunkedHMLP
from hnets.hnet_helpers import init_conditional_embeddings
from hnets.hnet_perturbation_wrapper import HPerturbWrapper
from hnets.mlp_hnet import HMLP
from hnets.structured_hmlp_examples import resnet_chunking, wrn_chunking
from hnets.structured_mlp_hnet import StructuredHMLP
from probabilistic import GaussianBNNWrapper
from probabilistic import prob_utils as putils
from probabilistic.regression import train_utils as rtu
from probabilistic.prob_cifar import hpsearch_config_resnet_avb as hpresnetavb
from probabilistic.prob_cifar import hpsearch_config_resnet_avb_pf as \
    hpresnetavbpf
from probabilistic.prob_cifar import hpsearch_config_zenke_avb as hpzenkeavb
from probabilistic.prob_cifar import hpsearch_config_zenke_avb_pf as \
    hpzenkeavbpf
from probabilistic.prob_cifar import hpsearch_config_zenke_bbb as hpzenkebbb
from probabilistic.prob_cifar import hpsearch_config_resnet_bbb as hpresnetbbb
from probabilistic.prob_cifar import hpsearch_config_resnet_ewc as hpresnetewc
from probabilistic.prob_cifar import hpsearch_config_resnet_mt as hpresnetmt
from probabilistic.prob_cifar import hpsearch_config_resnet_ssge as hpresnetssge
from probabilistic.prob_cifar import hpsearch_config_resnet_ssge_pf as \
    hpresnetssgepf
from probabilistic.prob_gmm import hpsearch_config_gmm_bbb as hpgmmbbb
from probabilistic.prob_gmm import hpsearch_config_gmm_ewc as hpgmmewc
from probabilistic.prob_gmm import hpsearch_config_gmm_avb as hpgmmavb
from probabilistic.prob_gmm import hpsearch_config_gmm_avb_pf as hpgmmavbpf
from probabilistic.prob_gmm import hpsearch_config_gmm_ssge as hpgmmssge
from probabilistic.prob_gmm import hpsearch_config_gmm_ssge_pf as hpgmmssgepf
from probabilistic.prob_mnist import train_utils as pmutils
from probabilistic.prob_mnist import hpsearch_config_split_avb as hpsplitavb
from probabilistic.prob_mnist import hpsearch_config_split_avb_pf as \
    hpsplitavbpf
from probabilistic.prob_mnist import hpsearch_config_perm_avb as hppermavb
from probabilistic.prob_mnist import hpsearch_config_perm_avb_pf as \
    hppermavbpf
from probabilistic.prob_mnist import hpsearch_config_perm_bbb as hppermbbb
from probabilistic.prob_mnist import hpsearch_config_perm_ewc as hppermewc
from probabilistic.prob_mnist import hpsearch_config_perm_mt as hppermmt
from probabilistic.prob_mnist import hpsearch_config_split_bbb as hpsplitbbb
from probabilistic.prob_mnist import hpsearch_config_split_ewc as hpsplitewc
from probabilistic.prob_mnist import hpsearch_config_split_mt as hpsplitmt
from probabilistic.prob_mnist import hpsearch_config_split_ssge as \
    hpsplitssge
from probabilistic.prob_mnist import hpsearch_config_split_ssge_pf as \
    hpsplitssgepf
from utils import gan_helpers as gan
from utils import sim_utils as sutils
from utils import torch_utils as tutils

def generate_networks(config, shared, logger, data_handlers, device,
                      create_mnet=True, create_hnet=True, create_hhnet=True,
                      create_dis=True):
    """Create the networks required for training with implicit distributions.

    This function will create networks based on user configuration.

    This function also takes care of weight initialization.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions.
        logger: Console (and file) logger.
        data_handlers: List of data handlers, one for each task. Needed to
            extract the number of inputs/outputs of the main network. And to
            infer the number of tasks.
        device: Torch device.
        create_mnet (bool, optional): If ``False``, the user can force that no
            main network is generated.
        create_hnet  (bool, optional): If ``False``, the user can force that no
            hypernet ``hnet`` is generated.

            Note:
                Even if ``True``, the ``hnet`` is only generated if the user
                configuration ``config`` requests it.
        create_hhnet  (bool, optional): If ``False``, the user can force that no
            hyper-hypernet ``hhnet`` is generated.

            Note:
                Even if ``True``, the ``hhnet`` is only generated if the user
                configuration ``config`` requests it.
        create_dis  (bool, optional): If ``False``, the user can force that no
            discriminator ``dis`` is generated.

            Note:
                Even if ``True``, the ``dis`` is only generated if the user
                configuration ``config`` requests it.

    Returns:
        (tuple): Tuple containing:

        - **mnet**: Main network instance.
        - **hnet** (optional): Hypernetwork instance. This return value is
          ``None`` if no hypernetwork should be constructed.
          **hhnet** (optional): Hyper-hypernetwork instance. This return value
          is ``None`` if no hyper-hypernetwork should be constructed.
        - **dis** (optional): Discriminator instance. This return value is
          ``None`` if no discriminator should be constructed.
    """
    num_tasks = len(data_handlers)
    if hasattr(config, 'cl_scenario'):
        num_heads = 1 if config.cl_scenario == 2 else num_tasks
    else:
        assert hasattr(config, 'multi_head')
        num_heads = num_tasks if config.multi_head else 1

    # Sanity check!
    for i in range(1, num_tasks):
        assert np.prod(data_handlers[i].in_shape) == \
               np.prod(data_handlers[0].in_shape)
        if data_handlers[0].classification:
            assert data_handlers[i].num_classes == data_handlers[0].num_classes
        else:
            assert np.prod(data_handlers[i].out_shape) == \
                   np.prod(data_handlers[0].out_shape)

    # Parse user "wishes".
    use_hnet = False
    use_hhnet = False
    use_dis = False
    no_mnet_weights = False

    if hasattr(config, 'mnet_only'):
        use_hnet = not config.mnet_only
        use_hhnet = not config.mnet_only and not shared.prior_focused and \
            not config.no_hhnet
        # Note, without the hypernet, there is no weight distribution and therefore
        # no discriminator needed.
        use_dis = use_hnet and not config.no_dis
        no_mnet_weights = not config.mnet_only
        if hasattr(config, 'distill_iter'):
            # Note, if distillation is used, the hnet is first trained independent
            # of a hyper-hypernetwork, which is why it needs its own weights.
            no_hnet_weights = use_hhnet and config.distill_iter == -1
        else:
            no_hnet_weights = use_hhnet

    ####################
    ### Main network ###
    ####################
    if 'gmm' in shared.experiment_type or \
            'regression' in shared.experiment_type:
        mnet_type = 'mlp'
        in_shape = data_handlers[0].in_shape

    elif 'mnist' in shared.experiment_type:
        if hasattr(config, 'net_type'):
            logger.debug('Main network will be of type: %s.' % config.net_type)
            mnet_type = config.net_type
        else:
            logger.debug('Main network will be an MLP.')
            mnet_type = 'mlp'


        assert len(data_handlers[0].in_shape) == 3 # MNIST
        in_shape = data_handlers[0].in_shape
        # Note, that padding is currently only applied when transforming the
        # image to a torch tensor.
        if isinstance(data_handlers[0], PermutedMNIST):
            assert len(data_handlers[0].torch_in_shape) == 3 # MNIST
            in_shape = data_handlers[0].torch_in_shape

    else:
        assert 'cifar' in shared.experiment_type

        in_shape = [32, 32, 3]
        if 'zenke' in shared.experiment_type:
            assert not hasattr(config, 'net_type')
            mnet_type = 'zenke'
        else:
            assert 'resnet' in shared.experiment_type
            mnet_type = config.net_type

    if mnet_type == 'mlp':
        if len(in_shape) > 1:
            n_x = np.prod(in_shape)
            in_shape = [n_x]
    else:
        assert len(in_shape) == 3
        assert mnet_type in ['lenet', 'resnet', 'wrn', 'iresnet', 'zenke']


    if data_handlers[0].classification:
        out_shape = [data_handlers[0].num_classes * num_heads]
    else:
        assert len(data_handlers[0].out_shape) == 1
        out_shape = [data_handlers[0].out_shape[0] * num_heads]

    if not create_mnet:
        # FIXME We would need to allow the passing of old `mnet`s.
        raise NotImplementedError('This function doesn\'t support yet to ' +
                                  'construct networks without constructing ' +
                                  'a main network first.')

    logger.info('Creating main network ...')
    mnet_kwargs = {}
    if mnet_type == 'iresnet':
        mnet_kwargs['cutout_mod'] = True
    mnet =  sutils.get_mnet_model(config, mnet_type, in_shape, out_shape,
                                  device, no_weights=no_mnet_weights,
                                  **mnet_kwargs)

    # Initialize main net weights, if any.
    assert not hasattr(config, 'custom_network_init')
    if hasattr(config, 'normal_init'):
        mnet.custom_init(normal_init=config.normal_init,
                         normal_std=config.std_normal_init, zero_bias=True)
    else:
        mnet.custom_init(zero_bias=True)

    #####################
    ### Discriminator ###
    #####################
    dis = None
    if use_dis and create_dis:
        logger.info('Creating discriminator ...')
        if config.use_batchstats:
            in_shape = [mnet.num_params * 2]
        else:
            in_shape = [mnet.num_params]
        dis = sutils.get_mnet_model(config, config.dis_net_type, in_shape, [1],
                                    device, cprefix='dis_', no_weights=False)
        dis.custom_init(normal_init=config.normal_init,
                        normal_std=config.std_normal_init, zero_bias=True)

    #####################
    ### Hypernetwork ###
    #####################
    def _hyperfan_init(net, mnet, cond_var, uncond_var):
        if isinstance(net, HMLP):
            net.apply_hyperfan_init(method='in', use_xavier=False,
                                    uncond_var=uncond_var, cond_var=cond_var,
                                    mnet=mnet)
        elif isinstance(net, ChunkedHMLP):
            net.apply_chunked_hyperfan_init(method='in', use_xavier=False,
                uncond_var=uncond_var, cond_var=cond_var, mnet=mnet, eps=1e-5,
                cemb_normal_init=False)
        elif isinstance(net, StructuredHMLP):
            # FIXME We should adapt `uncond_var`, as chunk embeddings are
            # additionally inputted as unconditional inputs.
            # FIXME We should provide further instructions on what individual
            # chunks represent (e.g., batchnorm scales and shifts should be
            # initialized differently).
            for int_hnet in net.internal_hnets:
                net.apply_hyperfan_init(method='in', use_xavier=False,
                    uncond_var=uncond_var, cond_var=cond_var, mnet=None)
        else:
            raise NotImplementedError('No hyperfan-init implemented for ' +
                                      'hypernetwork of type %s.' % type(net))

    hnet = None
    if use_hnet and create_hnet:
        logger.info('Creating hypernetwork ...')

        # For now, we either produce all or no weights with the hypernet.
        # Note, it can be that the mnet was produced with internal weights.
        assert mnet.hyper_shapes_learned is None or \
            len(mnet.param_shapes) == len(mnet.hyper_shapes_learned)

        chunk_shapes = None
        num_per_chunk = None
        assembly_fct = None
        if config.imp_hnet_type == 'structured_hmlp':
            if mnet_type == 'resnet':
                chunk_shapes, num_per_chunk, assembly_fct = \
                    resnet_chunking(mnet,
                                    gcd_chunking=config.imp_shmlp_gcd_chunking)
            elif mnet_type == 'wrn':
                chunk_shapes, num_per_chunk, assembly_fct = \
                    wrn_chunking(mnet,
                        gcd_chunking=config.imp_shmlp_gcd_chunking,
                        ignore_bn_weights=False, ignore_out_weights=False)
            else:
                raise NotImplementedError('"structured_hmlp" not implemented ' +
                                          'for network of type %s.' % mnet_type)

        # The hypernet is an implicit distribution, that only receives noise
        # as input, which are unconditional inputs.
        hnet = sutils.get_hypernet(config, device, config.imp_hnet_type,
            mnet.param_shapes, 0, cprefix='imp_',
            no_uncond_weights=no_hnet_weights, no_cond_weights=True,
            uncond_in_size=config.latent_dim, shmlp_chunk_shapes=chunk_shapes,
            shmlp_num_per_chunk=num_per_chunk, shmlp_assembly_fct=assembly_fct)
        #if isinstance(hnet, StructuredHMLP):
        #    print(num_per_chunk)
        #    for ii, int_hnet in enumerate(hnet.internal_hnets):
        #        print('   Internal hnet %d with %d outputs.' % \
        #              (ii, int_hnet.num_outputs))

        ### Initialize hypernetwork.
        if not no_hnet_weights:
            if not config.hyper_fan_init:
                rtu.apply_custom_hnet_init(config, logger, hnet)
            else:
                _hyperfan_init(hnet, mnet, -1, config.latent_std**2)

        ### Apply noise trick if requested by user.
        if config.full_support_perturbation != -1:
            hnet = HPerturbWrapper(hnet, hnet_uncond_in_size=config.latent_dim,
                                   sigma_noise=config.full_support_perturbation)

            shared.noise_dim = hnet.num_outputs
        else:
            shared.noise_dim = config.latent_dim

    ##########################
    ### Hyper-hypernetwork ###
    ##########################
    hhnet = None
    if use_hhnet and create_hhnet:
        if not create_hnet:
            # FIXME We require an existing hnet to do this.
            raise NotImplementedError('This function doesn\'t allow yet the ' +
                                      'creation of a hyper-hypernet without ' +
                                      'first creating a hypernetwork.')
        logger.info('Creating hyper-hypernetwork ...')

        assert hnet is not None
        assert len(hnet.unconditional_param_shapes) == len(hnet.param_shapes)
        hhnet = sutils.get_hypernet(config, device, config.hh_hnet_type,
                                    hnet.unconditional_param_shapes, num_tasks,
                                    cprefix='hh_')

        ### Initialize hypernetwork.
        if not config.hyper_fan_init:
            rtu.apply_custom_hnet_init(config, logger, hhnet)
        else:
            # Note, hyperfan-init doesn't take care of task-embedding
            # intialization.
            init_conditional_embeddings(hhnet,
                                        normal_std=config.std_normal_temb)

            _hyperfan_init(hhnet, hnet, config.std_normal_temb**2, -1)

    return mnet, hnet, hhnet, dis

def setup_summary_dict(config, shared, experiment, mnet, hnet=None,
                       hhnet=None, dis=None):
    """Setup the summary dictionary that is written to the performance
    summary file (in the result folder).

    This method adds the keyword "summary" to ``shared``.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions (summary dict will be added).
        experiment: Type of experiment. See argument `experiment` of method
            :func:`probabilistic.prob_cifar.train_avb.run`.
        mnet: Main network.
        hnet (optional): Implicit Hypernetwork.
        hhnet (optional): Hyper-Hypernetwork.
        dis (optional): Discriminator.
    """
    assert experiment in ['gmm_bbb', 'gmm_avb', 'gmm_avb_pf',
                          'split_bbb', 'perm_bbb',
                          'cifar_zenke_bbb', 'cifar_resnet_bbb',
                          'split_mnist_avb', 'split_mnist_avb_pf',
                          'perm_mnist_avb', 'perm_mnist_avb_pf',
                          'cifar_zenke_avb', 'cifar_zenke_avb_pf',
                          'cifar_resnet_avb', 'cifar_resnet_avb_pf',
                          'gmm_ssge', 'gmm_ssge_pf',
                          'split_mnist_ssge', 'split_mnist_ssge_pf',
                          'perm_mnist_ssge', 'perm_mnist_ssge_pf',
                          'cifar_resnet_ssge', 'cifar_resnet_ssge_pf',
                          'gmm_ewc', 'split_mnist_ewc', 'perm_mnist_ewc',
                          'cifar_resnet_ewc',
                          'gmm_mt', 'split_mnist_mt', 'perm_mnist_mt',
                          'cifar_resnet_mt']

    summary = dict()

    mnum = mnet.num_params
    hnum = -1
    hhnum = -1
    dnum = -1

    hm_ratio = -1
    hhm_ratio = -1
    dm_ratio = -1

    if hnet is not None:
        hnum = hnet.num_params
        hm_ratio = hnum / mnum
    if hhnet is not None:
        hhnum = hhnet.num_params
        hhm_ratio = hhnum / mnum
    if dis is not None:
        dnum = dis.num_params
        dm_ratio = dnum / mnum

    if experiment == 'gmm_bbb':
        summary_keys = hpgmmbbb._SUMMARY_KEYWORDS
    elif experiment == 'split_bbb':
        summary_keys = hpsplitbbb._SUMMARY_KEYWORDS
    elif experiment == 'perm_bbb':
        summary_keys = hppermbbb._SUMMARY_KEYWORDS
    elif experiment == 'cifar_zenke_bbb':
        summary_keys = hpzenkebbb._SUMMARY_KEYWORDS
    elif experiment == 'cifar_resnet_bbb':
        summary_keys = hpresnetbbb._SUMMARY_KEYWORDS
    elif experiment == 'gmm_avb':
        summary_keys = hpgmmavb._SUMMARY_KEYWORDS
    elif experiment == 'gmm_avb_pf':
        summary_keys = hpgmmavbpf._SUMMARY_KEYWORDS
    elif experiment == 'split_mnist_avb':
        summary_keys = hpsplitavb._SUMMARY_KEYWORDS
    elif experiment == 'split_mnist_avb_pf':
        summary_keys = hpsplitavbpf._SUMMARY_KEYWORDS
    elif experiment == 'perm_mnist_avb':
        summary_keys = hppermavb._SUMMARY_KEYWORDS
    elif experiment == 'perm_mnist_avb_pf':
        summary_keys = hppermavbpf._SUMMARY_KEYWORDS
    elif experiment == 'cifar_resnet_avb':
        summary_keys = hpresnetavb._SUMMARY_KEYWORDS
    elif experiment == 'cifar_resnet_avb_pf':
        summary_keys = hpresnetavbpf._SUMMARY_KEYWORDS
    elif experiment == 'cifar_zenke_avb':
        summary_keys = hpzenkeavb._SUMMARY_KEYWORDS
    elif experiment == 'gmm_ssge':
        summary_keys = hpgmmssge._SUMMARY_KEYWORDS
    elif experiment == 'gmm_ssge_pf':
        summary_keys = hpgmmssgepf._SUMMARY_KEYWORDS
    elif experiment == 'split_mnist_ssge':
        summary_keys = hpsplitssge._SUMMARY_KEYWORDS
    elif experiment == 'split_mnist_ssge_pf':
        summary_keys = hpsplitssgepf._SUMMARY_KEYWORDS
    elif experiment == 'perm_mnist_ssge':
        summary_keys = hpsplitssgepf._SUMMARY_KEYWORDS
    elif experiment == 'perm_mnist_ssge_pf':
        summary_keys = hpsplitssgepf._SUMMARY_KEYWORDS
    elif experiment == 'cifar_resnet_ssge':
        summary_keys = hpresnetssge._SUMMARY_KEYWORDS
    elif experiment == 'cifar_resnet_ssge_pf':
        summary_keys = hpresnetssgepf._SUMMARY_KEYWORDS
    elif experiment == 'cifar_zenke_avb_pf':
        summary_keys = hpzenkeavbpf._SUMMARY_KEYWORDS
    elif experiment == 'gmm_ewc':
        summary_keys = hpgmmewc._SUMMARY_KEYWORDS
    elif experiment == 'split_mnist_ewc':
        summary_keys = hpsplitewc._SUMMARY_KEYWORDS
    elif experiment == 'perm_mnist_ewc':
        summary_keys = hppermewc._SUMMARY_KEYWORDS
    elif experiment == 'cifar_resnet_ewc':
        summary_keys = hpresnetewc._SUMMARY_KEYWORDS
    elif experiment == 'gmm_mt':
        summary_keys = hpgmmmt._SUMMARY_KEYWORDS
    elif experiment == 'split_mnist_mt':
        summary_keys = hpsplitmt._SUMMARY_KEYWORDS
    elif experiment == 'perm_mnist_mt':
        summary_keys = hppermmt._SUMMARY_KEYWORDS
    else:
        assert experiment == 'cifar_resnet_mt'
        summary_keys = hpresnetmt._SUMMARY_KEYWORDS

    for k in summary_keys:
        if k == 'acc_task_given' or \
                k == 'acc_task_given_during' or \
                k == 'acc_task_inferred_ent' or \
                k == 'acc_task_inferred_ent_during' or \
                k == 'acc_dis':
            summary[k] = [-1] * config.num_tasks

        elif k == 'acc_avg_final' or \
                k == 'acc_avg_during' or \
                k == 'acc_avg_task_given' or \
                k == 'acc_avg_task_given_during' or \
                k == 'acc_avg_task_inferred_ent' or \
                k == 'acc_avg_task_inferred_ent_during' or \
                k == 'avg_task_inference_acc_ent' or \
                k == 'acc_avg_task_inferred_conf' or \
                k == 'avg_task_inference_acc_conf' or \
                k == 'acc_avg_task_inferred_agree' or \
                k == 'avg_task_inference_acc_agree' or \
                k == 'acc_avg_dis':
            summary[k] = -1

        elif k == 'num_weights_main':
            summary[k] = mnum
        elif k == 'num_weights_hyper':
            summary[k] = hnum
        elif k == 'num_weights_hyper_hyper':
            summary[k] = hhnum
        elif k == 'num_weights_dis':
            summary[k] = dnum
        elif k == 'num_weights_hm_ratio':
            summary[k] = hm_ratio
        elif k == 'num_weights_hhm_ratio':
            summary[k] = hhm_ratio
        elif k == 'num_weights_dm_ratio':
            summary[k] = dm_ratio

        elif k == 'finished':
            summary[k] = 0
        else:
            # Implementation must have changed if this exception is
            # raised.
            raise ValueError('Summary argument %s unknown!' % k)

    shared.summary = summary

def set_train_mode(training, mnet, hnet, hhnet, dis):
    """Set mode of all given networks.

    Note, all networks be passed as ``None`` and only the provided networks
    its mode is set.

    Args:
        training (bool): If ``True``, training mode will be activated.
            Otherwise, evaluation mode is activated.
        (....): The remaining arguments refer to network instances.    
    """
    for net in [mnet, hnet, hhnet, dis]:
        if net is not None:
            if training:
                net.train()
            else:
                net.eval()

def compute_acc(task_id, data, mnet, hnet, hhnet, device, config, shared,
                split_type='test', return_dataset=False, return_entropies=False,
                return_confidence=False, return_agreement=False,
                return_pred_labels=False, return_labels=False,
                return_samples=False, deterministic_sampling=False,
                in_samples=None, out_samples=None, num_w_samples=None,
                w_samples=None):
    """Compute the accuracy over a specified dataset split.

    Note, this function does not explicitly execute the code within a
    ``torch.no_grad()`` context. This needs to be handled from the outside if
    desired.

    Note, this function serves the same purpose as function
    :func:`probabilistic.prob_mnist.train_utils.compute_acc`.

    The ``task_id`` is used only to select the task embedding (if ``hhnet``
    is given) and the correct output units depending on the CL scenario.

    Args:
        (....): See docstring of function
            :func:`probabilistic.prob_mnist.train_utils.compute_acc`.
        return_samples: If ``True``, the attribute ``samples`` will be
            added to the ``return_vals`` Namespace (see return values). This
            field will contain all weight samples that have been drawn from
            the hypernetwork ``hnet``. If ``hnet`` is not provided,
            this field will be ``None``. The field will be filled with a
            numpy array.

    Returns:
        (tuple): Tuple containing:

        - **accuracy**: Overall accuracy on dataset split.
        - **return_vals**: A namespace object that contains several attributes,
          depending on the arguments passed. It will allways contain the
          following attribute, denoting the current weights of the implicit
          distribution.

              - ``theta``: The current output of the ``hhnet`` for ``task_id``.
                If no ``hhnet`` is provided but an ``hnet`` is given,
                then its weights ``theta`` will be provided. It will be
                ``None`` if only a main network ``mnet`` is provided.
    """
    # FIXME The code is almost a perfect copy from the original function.

    assert in_samples is not None or split_type in ['test', 'val', 'train']
    assert out_samples is None or in_samples is not None

    generator = None
    if deterministic_sampling:
        generator = torch.Generator()#device=device)
        # Note, PyTorch recommends using large random seeds:
        # https://tinyurl.com/yx7fwrry
        generator.manual_seed(2147483647)

    return_vals = Namespace()

    allowed_outputs = pmutils.out_units_of_task(config, data, task_id,
                                                shared.num_trained)

    ST = shared.softmax_temp[task_id]
    if not config.calibrate_temp:
        assert ST == 1.

    if in_samples is not None:
        X = in_samples
        T = out_samples
    elif split_type == 'train':
        X = data.get_train_inputs()
        T = data.get_train_outputs()
    elif split_type == 'test' or data.num_val_samples == 0:
        X = data.get_test_inputs()
        T = data.get_test_outputs()
    else:
        X = data.get_val_inputs()
        T = data.get_val_outputs()

    num_samples = X.shape[0]

    if T is not None:
        T = pmutils.fit_targets_to_softmax(config, shared, device, data,
                                           task_id, T)

    if return_dataset:
        return_vals.inputs = X
        return_vals.targets = T

    labels = None
    if T is not None:
        labels = np.argmax(T, axis=1)
    if return_labels:
        return_vals.labels = labels

    X = data.input_to_torch_tensor(X, device)
    #if T is not None:
    #    T = data.output_to_torch_tensor(T, device)

    hnet_theta = None
    return_vals.theta = None
    if hhnet is not None:
        assert hnet is not None
        hnet_theta = hhnet.forward(cond_id=task_id)
        return_vals.theta = hnet_theta
    elif hnet is not None:
        return_vals.theta = hnet.unconditional_params

    # There is no weight sampling without an implicit hypernetwork.
    if w_samples is not None:
        num_w_samples = len(w_samples)
    elif num_w_samples is None:
        num_w_samples = 1 if hnet is None else config.val_sample_size
    else:
        if hnet is None and num_w_samples > 1:
            warn('Cannot draw multiple weight samples for deterministic ' +
                 'network')
            num_w_samples = 1

    if hasattr(config, 'non_growing_sf_cl3') and config.cl_scenario == 3 \
            and config.non_growing_sf_cl3:
        softmax_width = config.num_tasks * data.num_classes
    elif config.cl_scenario == 3 and not config.split_head_cl3:
        softmax_width = len(allowed_outputs)
    else:
        softmax_width = data.num_classes
    softmax_outputs = np.empty((num_w_samples, X.shape[0], softmax_width))

    if return_samples:
        return_vals.samples = None

    # FIXME Note, that a continually learned hypernet (whose weights come from a
    # hyper-hypernet) would in principle also require correct argument passing,
    # e.g., to choose the correct set of batch statistics.
    kwargs = pmutils.mnet_kwargs(config, task_id, mnet)

    for j in range(num_w_samples):
        weights = None
        if w_samples is not None:
            weights = w_samples[j]
        elif hnet is not None:
            z = torch.normal(torch.zeros(1, shared.noise_dim),
                config.latent_std, generator=generator).to(device)
            weights = hnet.forward(uncond_input=z, weights=hnet_theta)

        if weights is not None and return_samples:
            if j == 0:
                return_vals.samples = np.empty((num_w_samples,
                                                hnet.num_outputs))
            return_vals.samples[j, :] = torch.cat([p.detach().flatten() \
                for p in weights]).cpu().numpy()


        curr_bs = config.val_batch_size
        n_processed = 0

        while n_processed < num_samples:
            if n_processed + curr_bs > num_samples:
                curr_bs = num_samples - n_processed
            n_processed += curr_bs

            sind = n_processed - curr_bs
            eind = n_processed

            Y = mnet.forward(X[sind:eind, :], weights=weights, **kwargs)
            if allowed_outputs is not None:
                Y = Y[:, allowed_outputs]

            softmax_outputs[j, sind:eind, :] = F.softmax(Y / ST, dim=1). \
                detach().cpu().numpy()

    # Predictive distribution per sample.
    pred_dists = softmax_outputs.mean(axis=0)

    pred_labels = np.argmax(pred_dists, axis=1)
    # Note, that for CL3 (without split heads) `labels` are already absolute,
    # not relative to the head (see post-processing of targets `T` above).
    if labels is not None:
        accuracy = 100. * np.sum(pred_labels == labels) / num_samples
    else:
        accuracy = None

    if return_pred_labels:
        assert pred_labels.size == X.shape[0]
        return_vals.pred_labels = pred_labels

    if return_entropies:
        # We use the "maximum" trick to improve numerical stability.
        return_vals.entropies = - np.sum(pred_dists * \
                                         np.log(np.maximum(pred_dists, 1e-5)),
                                         axis=1)
        # return_vals.entropies = - np.sum(pred_dists * np.log(pred_dists),
        #                                 axis=1)
        assert return_vals.entropies.size == X.shape[0]

        # Normalize by maximum entropy.
        max_ent = - np.log(1.0 / data.num_classes)
        return_vals.entropies /= max_ent

    if return_confidence:
        return_vals.confidence = np.max(pred_dists, axis=1)
        assert return_vals.confidence.size == X.shape[0]

    if return_agreement:
        return_vals.agreement = softmax_outputs.std(axis=0).mean(axis=1)
        assert return_vals.agreement.size == X.shape[0]

    return accuracy, return_vals

def estimate_implicit_moments(config, shared, task_id, hnet, hhnet, num_samples,
                              device):
    """Estimate the first two moments of an implicit distribution.

    This function takes the implicit distribution represented by ``hnet`` and
    estimates the mean and the variances of its outputs.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions.
        task_id (int): In case ``hhnet`` is provided, this will be used to
            select the task embedding.
        hnet: The hypernetwork.
        hhnet: The hyper-hypernetwork, may be ``None``.
        num_samples: The number of samples that should be drawn from the
            ``hnet`` to estimate the statistics.
        device: The PyTorch device.

    Returns:
        (tuple): Tuple containing:

        - **sample_mean** (torch.Tensor): Estimated mean of the implicit
          distribution.
        - **sample_std** (torch.Tensor): Estimated standard deviation of the
          implicit distribution.
    """
    theta = None
    if hhnet is not None:
        theta = hhnet.forward(cond_id=task_id)

    samples = torch.empty((num_samples, hnet.num_outputs)).to(device)

    for j in range(num_samples):
        z = torch.normal(torch.zeros(1, shared.noise_dim), config.latent_std).\
            to(device)

        weights = hnet.forward(uncond_input=z, weights=theta)

        samples[j, :] = torch.cat([p.detach().flatten() for p in weights])

    sample_mean = samples.mean(dim=0)
    sample_std = samples.std(dim=0)

    return sample_mean, sample_std

def process_dis_batch(config, shared, batch_size, device, dis, hnet, hnet_theta,
                      dist=None):
    """Process a batch of weight samples via the discriminator.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions.
        batch_size (int): How many samples should be fed through the
            discriminator.
        device: PyTorch device.
        dis: Discriminator.
        hnet: The hypernetwork, representing an implicit distribution from
            which to sample weights. Is only used to draw samples if
            ``dist`` is ``None``.
        hnet_theta: The weights passed to ``hnet`` when drawing samples.
        dist (torch.distributions.normal.Normal): A normal distribution,
            from which discriminator inputs can be sampled.

    Returns:
        (tuple): Tuple containing:

        - **dis_out** (torch.Tensor): The discriminator output for the given
          batch of samples.
        - **dis_input** (torch.Tensor): The samples that have been passed
          through the discriminator.
    """

    if dist is not None:
        samples = dist.sample([batch_size])
        if hnet is not None:
            assert np.all(np.equal(samples.shape,
                                   [batch_size, hnet.num_outputs]))
    else:
        assert hnet is not None

        z = torch.normal(torch.zeros(batch_size, shared.noise_dim),
                         config.latent_std).to(device)

        samples = hnet.forward(uncond_input=z, weights=hnet_theta,
                               ret_format='flattened')

    if config.use_batchstats:
        samples = gan.concat_mean_stats(samples)

    return dis.forward(samples), samples

def calc_prior_matching(config, shared, batch_size, device, dis, hnet,
                        theta_current, dist_prior, dist_ac,
                        return_current_samples=False):
    """Calculate the prior-matching term.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions.
        batch_size (int): How many samples should be fed through the
            discriminator.
        device: PyTorch device.
        dis: Discriminator.
        hnet: The hypernetwork, representing an implicit distribution from
            which to sample weights. Is used to draw samples from the current
            implicit distribution ``theta_current`` (which may be ``None`` if
            internal weights should be selected).
        theta_current: The weights passed to ``hnet`` when drawing samples from
            the current implicit distribution that should be matched to the
            prior (can be ``None`` if internally maintaned weights of ``hnet``
            should be used).
        dist_prior (torch.distributions.normal.Normal): A normal distribution,
            that represents an explicit prior. Only used if ``dist_ac`` is
            not ``None``.
        dist_ac (torch.distributions.normal.Normal): A normal distribution,
            that can be passed if the adaptive contrast trick is used. If not
            ``None``, then ``dist_prior`` may not be ``None``.
        return_current_samples (bool): If ``True``, the samples collected from
            the current implicit distribution are returned.

    Returns:
        (tuple): Tuple containing:

        - **loss_pm**: (torch.Tensor): The unscaled loss value for the
          prior-matching term.
        - **curr_samples** (list): List of samples drawn from the implicit
          distribution ``hnet`` (using ``theta_current``).
    """
    assert dist_ac is None or dist_prior is not None

    # The following two terms are only required if AC is used.
    log_prob_ac = 0
    log_prob_prior = 0

    if return_current_samples:
        curr_samples = []
    else:
        curr_samples = None

    # Translate into samples from the current implicit distribution.
    w_samples = torch.empty((batch_size, hnet.num_outputs)).to(device)

    # FIXME Create batch of samples rather than looping.
    for j in range(batch_size):
        z = torch.normal(torch.zeros(1, shared.noise_dim), config.latent_std).\
            to(device)

        weights = hnet.forward(uncond_input=z, weights=theta_current)

        w_samples[j, :] = torch.cat([p.flatten() for p in weights])

        if return_current_samples:
            curr_samples.append(weights)

    if dist_ac is not None:
        log_prob_ac = dist_ac.log_prob(w_samples).sum(dim=1).mean()
        log_prob_prior = dist_prior.log_prob(w_samples).sum(dim=1).mean()

    if config.use_batchstats:
        w_samples = gan.concat_mean_stats(w_samples)

    value_t = dis.forward(w_samples).mean()

    return value_t + log_prob_ac - log_prob_prior, curr_samples

def calc_batch_uncertainty(config, shared, task_id, device, inputs, mnet, hnet,
                           hhnet, data, num_w_samples, hnet_theta=None,
                           allowed_outputs=None):
    """Compute the per-sample uncertainties for a given batch of inputs.

    Note:
        This function is executed inside a ``torch.no_grad()`` context.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared: Miscellaneous data shared among training functions (softmax
            temperature is stored in here).
        task_id (int): In case a hypernet ``hnet`` is given, the ``task_id`` is
            used to load the corresponding main network ``mnet`` weights.
        device: PyTorch device.
        inputs (torch.Tensor): A batch of main network ``mnet`` inputs.
        mnet: The main network.
        hnet (optional): The implicit hypernetwork, can be ``None``.
        hhnet (optional): The hyper-hypernetwork, can be ``None``.
        data: Dataset loader. Needed to determine the number of classes.
        num_w_samples (int): The number of weight samples that should be drawn
            to estimate predictive uncertainty.
        hnet_theta (tuple, optional): To save computation, one can pass
            weights for the implicit hypernetwork ``hnet``, if they have been
            computed prior to calling this methods.
        allowed_outputs (tuple, optional): The indices of the neurons belonging
            to outputs head ``task_id``. Only needs to be specified in a
            multi-head setting.

    Returns:
        (numpy.ndarray): The entropy of the estimated predictive distribution
        per input sample.
    """
    assert data.classification
    assert config.cl_scenario == 2 or allowed_outputs is not None
    assert hhnet is None or hnet is not None

    # FIXME We calibrate the temperature after training on a task. This function
    # is currently only used to track batch uncertainty during training or
    # choose coreset samples that have maximum uncertainty on a single model
    # (note, relative order of uncertain samples doesn't change due to
    # calibration for a single model). Hence, the function is invoked before
    # the temperature is optimized.
    # Therefore, I throw an assertion if we use the function in the future for
    # other purposes, just in case the programmer is unaware.
    assert shared.softmax_temp[task_id] == 1.
    ST = shared.softmax_temp[task_id]

    with torch.no_grad():
        if hnet_theta is None and hhnet is not None:
            hnet_theta = hhnet.forward(cond_id=task_id)

        if allowed_outputs is not None:
            num_outs = len(allowed_outputs)
        else:
            num_outs = data.num_classes
        softmax_outputs = np.empty((num_w_samples, inputs.shape[0], num_outs))

        kwargs = pmutils.mnet_kwargs(config, task_id, mnet)

        for j in range(num_w_samples):
            weights = None

            if hnet is not None:
                z = torch.normal(torch.zeros(1, shared.noise_dim),
                                 config.latent_std).to(device)
                weights = hnet.forward(uncond_input=z, weights=hnet_theta)

            Y = mnet.forward(inputs, weights=weights, **kwargs)
            if allowed_outputs is not None:
                Y = Y[:, allowed_outputs]

            softmax_outputs[j, :, :] = F.softmax(Y / ST, dim=1).detach(). \
                cpu().numpy()

        # Predictive distribution per sample.
        pred_dists = softmax_outputs.mean(axis=0)

        # We use the "maximum" trick to improve numerical stability.
        entropies = - np.sum(pred_dists * np.log(np.maximum(pred_dists, 1e-5)),
                             axis=1)
        assert entropies.size == inputs.shape[0]

        # Normalize by maximum entropy.
        max_ent = - np.log(1.0 / data.num_classes)

        return entropies / max_ent

def visualize_implicit_dist(config, task_id, writer, train_iter, w_samples,
                            figsize=(10, 6)):
    """Visualize an implicit distribution.

    TODO
    """
    assert w_samples.ndim == 2

    num_weights = w_samples.shape[1]
    # Ensure that we always plot the same samples, independent of the simulation
    # its random seed.
    rand = np.random.RandomState(42)
    weight_inds = rand.choice(np.arange(num_weights), min(10, num_weights),
                              replace=False)
    weight_inds = np.sort(weight_inds)

    weight_samples = dict(('Weight %d' %  (weight_inds[i]),
        w_samples[:, weight_inds[i]].detach().cpu().numpy()) \
        for i in range(len(weight_inds)))

    # FIXME Adapt our plotting guidelines.
    df = pd.DataFrame.from_dict(weight_samples)

    # correlation matrix.
    plt.rcParams['figure.figsize'] = figsize
    plt.matshow(df.corr(method='pearson'), vmin=-1, vmax=1)
    plt.xticks(range(len(df.columns)), df.columns)
    plt.xticks(rotation=70)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.colorbar()

    writer.add_figure('eval/task_%d/correlation' % task_id, plt.gcf(),
                      train_iter, close=True)

    n = 0
    for p in weight_inds:
        for q in weight_inds:
            if q >= p:
                break

            # Avoid that plots get corrupted due to mode collapse.
            if np.isclose(weight_samples['Weight %d' % p].std(), 0) or \
                    np.isclose(weight_samples['Weight %d' % q].std(), 0):
                n += 1
                warn('Could not create plot "eval/task_%d/weight_%d_%d" ' \
                     % (task_id, p, q) + 'due to mode collapsed posterior ' +
                     'variance.')
                continue

            try:
                sns.jointplot(x='Weight %d' % (p), y='Weight %d' % (q), data=df,
                              kind="kde")
                writer.add_figure('eval/task_%d/weight_%d_%d' % (task_id, p, q),
                                  plt.gcf(), train_iter, close=True)
            except:
                warn('Could not visualize joint weight density.')
            n += 1

            if n > 9:
                break

        if n > 9:
            break

def calibrate_temperature(task_id, data, mnet, hnet, hhnet, device, config,
                          shared, logger, writer, cal_per_model=False,
                          only_correctly_classified=False,
                          cal_target_entropy=-1):
    """Calibrate softmax temperature for current task.

    When training in a continual learning setting, the loss is a combination
    of task-specific terms and regularizers (which are different for every
    task). These differences in the loss functions used for training will have
    an influence on the (in-distribution) softmax outputs.

    To overcome these differences, we perform a post-hoc calibration step
    using a proper score function (the negative-log likelihood, which is
    identical to the cross-entropy when using 1-hot targets) to learn the
    softmax temperature. Note, high softmax temperatures increase entropy,
    whereas low temperatures increase confidence.

    A proper calibration of each task will ensure that between task comparisons
    become easier.

    Note:
        We calibrate on the training set, as we want our in-distribution 
        predictive distributions to be properly calibrated for each task (note,
        tasks are trained using different loss functions since there are
        different regularizers that kick in over time). The only purpose of this
        function is to correct this behavior.

    Args:
        (....): See docstring of function :func:`train`.
        cal_per_model (bool): By default, we calibrate the predictive
            distriubtion, i.e., the averaged softmax across all models from the
            Bayesian ensemble (depending on ``config.cal_sample_size``). If
            instead we should calibrate individual models from this ensemble,
            this option can be set to ``True`` (note, behavior is the same if
            ``config.cal_sample_size == 1``).
        only_correctly_classified (bool): Only use correctly classified samples
            for the calibration (as determined by the argmax of the predictive
            distribution).
        cal_target_entropy (float): If not ``-1``, then instead of calibrating
            using a proper score function, we learn a temperature such that a
            given target in-distribution entropy is matched (i.e., we compute
            the entropy on a mini-batch and minimize the MSE towards the given
            target entropy). In this way, one can ensure actively that all tasks
            have the same in-distribution entropy.
    """
    logger.info('Temperature calibration for task %d ...' % (task_id+1))

    # FIXME We could also follow the code from
    #   https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    # but they don't consider BNNs. Note, there code is much more efficient
    # since they compute the logits before entering the training loop (which
    # is possible when only having one model). Though, in general, we have
    # multiple models.

    set_train_mode(True, mnet, hnet, hhnet, None)

    gauss_main = False
    if isinstance(mnet, GaussianBNNWrapper):
        gauss_main = True

    # Whether the hypernet represents an implicit distribution (i.e., it's
    # input is a random variable), or whether it has task embeddings as input.
    det_hnet = False
    if hnet is not None:
        if hnet.num_known_conds > 0:
            assert hhnet is None

            det_hnet = True
            # Can currently only be the case if we train a BbB setup with option
            # `mean_only` enabled.
            if not gauss_main:
                assert hasattr(config, 'mean_only') and config.mean_only

    # The single parameter to be tuned by this method.
    temp_param = torch.nn.Parameter(shared.softmax_temp[task_id],
                                    requires_grad=True)
    assert temp_param == 1.

    # Which temperature transfer function to use during training. Note, this
    # can ensure that temperatures don't become negative.
    # ttf = temperature transfer function
    ttf_choice = 'softplus'
    if ttf_choice == 'linear':
        ttf = lambda x : x
        #torch.nn.init.ones_(temp_param.data)
    elif ttf_choice == 'exp':
        ttf = torch.exp
        torch.nn.init.zeros_(temp_param.data)
    else:
        ttf = F.softplus
        temp_param.data = torch.log(torch.exp(torch.ones(1)) - \
                                    torch.ones(1)).to(device)

    allowed_outputs = pmutils.out_units_of_task(config, data, task_id,
                                                config.num_tasks)

    optimizer = tutils.get_optimizer([temp_param], config.lr,
        momentum=config.momentum, weight_decay=config.weight_decay,
        use_adam=config.use_adam, adam_beta1=config.adam_beta1,
        use_rmsprop=config.use_rmsprop, use_adadelta=config.use_adadelta,
        use_adagrad=config.use_adagrad)

    mnet_kwargs = pmutils.mnet_kwargs(config, task_id, mnet)

    num_w_samples = config.train_sample_size if config.cal_sample_size == -1 \
        else config.cal_sample_size

    with torch.no_grad():
        # We don't change any network parameters, so these calls produce
        # constant outputs.
        theta_current = None
        if hhnet is not None:
            theta_current = hhnet.forward(cond_id=task_id)
            theta_current = [p.detach() for p in theta_current]

        if gauss_main:
            assert hhnet is None

            if hnet is not None:
                hnet_out = hnet.forward(cond_id=task_id)
            else:
                hnet_out = None
            w_mean, w_rho = mnet.extract_mean_and_rho(weights=hnet_out)
            w_std = putils.decode_diag_gauss(w_rho,
                                             logvar_enc=mnet.logvar_encoding)

        elif det_hnet:
            w_mean = hnet.forward(cond_id=task_id)

        ### We first compute the logit outputs over all samples for all models,
        ### since they don't change anymore.
        # FIXME Could lead to memory issues for large datasets and might not be
        # inefficient if ``config.cal_temp_iter`` is small, since we won't
        # iterate over the whole dataset.
        inputs = data.get_train_inputs()
        targets = data.get_train_outputs()

        T = data.output_to_torch_tensor(targets, device, mode='train')
        # Modify 1-hot encodings according to CL scenario.
        assert T.shape[1] == data.num_classes
        # In CL1, CL2 and CL3 (with seperate heads) we do not have to modify the
        # targets.
        if config.cl_scenario == 3 and not config.split_head_cl3:
            raise NotImplementedError('Temperature calibration not ' +
                                      'implemented for CL3 without split-head.')

        _, labels = torch.max(T, 1) # Integer labels.
        #labels = labels.detach()

        num_samples = inputs.shape[0]

        logit_outputs = torch.empty((num_w_samples, num_samples, T.shape[1])). \
            to(device)

        for j in range(num_w_samples):
            if gauss_main: # Gaussian weight posterior.
                # In case of the local-reparam trick, we anyway have a different
                # weight per sample. So, the demand of having the same model for
                # all samples in the dataset drops.
                if config.local_reparam_trick:
                    # Note, the sampling will happen inside the forward method.
                    weights = None
                    emean = w_mean
                    erho = w_rho
                else:
                    weights = putils.sample_diag_gauss(w_mean, w_std,
                        is_radial=config.radial_bnn)
                    emean = None
                    erho = None

            elif det_hnet:
                weights = w_mean

            else:
                if hnet is not None: # Implicit hypernetwork.
                    z = torch.normal(torch.zeros(1, shared.noise_dim),
                                     config.latent_std).to(device)
                    weights = hnet.forward(uncond_input=z,
                                           weights=theta_current)
                else: # Main network only training.
                    weights = None

            # I use the validation batch size on purpose, since it is usually
            # bigger and we just want to quickly compute the logits.
            curr_bs = config.val_batch_size
            n_processed = 0

            while n_processed < num_samples:
                if n_processed + curr_bs > num_samples:
                    curr_bs = num_samples - n_processed
                n_processed += curr_bs

                sind = n_processed - curr_bs
                eind = n_processed

                ### Compute negative log-likelihood (NLL).
                X = data.input_to_torch_tensor(inputs[sind:eind, :], device,
                                               mode='train')

                if gauss_main:
                    Y = mnet.forward(X, weights=None, mean_only=False,
                                     extracted_mean=emean, extracted_rho=erho,
                                     sample=weights, **mnet_kwargs)
                else:
                    Y = mnet.forward(X, weights=weights, **mnet_kwargs)

                if allowed_outputs is not None:
                    Y = Y[:, allowed_outputs]

                logit_outputs[j, sind:eind, :] = Y

        # Since we computed all training logits, we might as well compute
        # the training accuracy on the predictive distributions at temperature 1
        # (note, temperature doesn't change predicted labels).
        pred_dists = F.softmax(logit_outputs, dim=2).mean(dim=0)
        assert pred_dists.ndim == 2
        _, pred_labels = torch.max(pred_dists, 1)
        train_acc = 100. * torch.sum(pred_labels == labels) / num_samples
        logger.debug('Task %d -- training accuracy: %.2f%%.' % \
                     (task_id+1, train_acc))

        log_pred_dists = torch.log(torch.clamp(pred_dists, min=1e-5))
        in_entropies = -torch.sum(pred_dists * log_pred_dists, dim=1)

        # Normalize by maximum entropy.
        max_ent = - np.log(1.0 / data.num_classes)
        in_entropies /= max_ent

        in_entropies_mean = in_entropies.mean()
        in_entropies_std = in_entropies.std()
        logger.debug('Task %d -- training in-dist. entropy: %f.' % \
                     (task_id+1, in_entropies_mean))

        if not hasattr(shared, 'train_in_ent_mean'):
            shared.train_in_ent_mean = []
            shared.train_in_ent_std = []
        shared.train_in_ent_mean.append( \
            in_entropies_mean.detach().cpu().numpy())
        shared.train_in_ent_std.append(in_entropies_std.detach().cpu().numpy())

        if only_correctly_classified:
            num_correct = torch.sum(pred_labels == labels)

            logger.info('Task %d -- only using %d/%d correctly classified ' \
                        % (task_id+1, num_correct, num_samples) + \
                        'samples for calibration.')

            logit_outputs = logit_outputs[:, pred_labels == labels, :]
            num_samples = num_correct
            assert logit_outputs.shape[1] == num_correct

            labels = labels[pred_labels == labels]
            assert labels.shape[0] == num_correct

            # Sanity check!
            pred_dists = F.softmax(logit_outputs, dim=2).mean(dim=0)
            _, pred_labels = torch.max(pred_dists, 1)
            assert torch.sum(pred_labels == labels) == num_correct

    logit_outputs = logit_outputs.detach()

    ### Calibrate temperature.
    for i in range(config.cal_temp_iter):
        optimizer.zero_grad()

        batch_inds = np.random.randint(0, num_samples, config.batch_size)

        batch_logits = logit_outputs[:, batch_inds, :]
        batch_labels = labels[batch_inds]
        assert batch_logits.ndim == 3

        # Note, this first option is more numerically stable when calibrating NLL.
        if cal_per_model or num_w_samples == 1:
            loss = 0
            for j in range(num_w_samples):
                if cal_target_entropy != -1:
                    batch_sm = F.softmax(batch_logits[j, :, :] / \
                        ttf(temp_param), dim=1)
                    # For numerical stability.
                    batch_log_sm = torch.log(torch.clamp(batch_sm, min=1e-5))

                    # Mean entropy within the batch.
                    batch_entropy = -torch.sum(batch_sm * batch_log_sm,
                                               dim=1).mean()

                    loss += (batch_entropy - cal_target_entropy)**2
                else: # Compute NLL loss
                    # Note, softmax will be computed inside the `cross_entropy`.
                    loss += F.cross_entropy( \
                        batch_logits[j, :, :] / ttf(temp_param), batch_labels,
                        reduction='mean')
            loss /= num_w_samples

        else:
            batch_pred_dist = F.softmax(batch_logits / ttf(temp_param),
                                        dim=2).mean(dim=0)
            # FIXME nll_loss expects log_softmax as input. To compute the
            # predictive distribution, we have to first average softmax outputs
            # before we can apply the log, which might lead to numerical
            # instabilities.
            #batch_log_pd = batch_pred_dist
            #batch_log_pd[batch_pred_dist < 1e-5] = 1e-5
            batch_log_pd = torch.clamp(batch_pred_dist, min=1e-5)
            batch_log_pd = torch.log(batch_log_pd)
            if cal_target_entropy != -1:
                # Mean entropy within the batch.
                batch_entropy = -torch.sum(batch_pred_dist * batch_log_pd,
                                           dim=1).mean()

                loss += (batch_entropy - cal_target_entropy)**2
            else: # Compute NLL loss
                loss = F.nll_loss(batch_log_pd, batch_labels, reduction='mean')

        loss.backward()
        if config.clip_grad_value != -1:
            torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'],
                                            config.clip_grad_value)
        elif config.clip_grad_norm != -1:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],
                                           config.clip_grad_norm)
        optimizer.step()

        if ttf_choice == 'linear':
            # NOTE In this case, nothing prevents the temperature from going
            # negative (e.g., when starting with a large learning rate).
            # Therefore, we have to actively capture this case.
            temp_param.data = torch.clamp(temp_param, min=1e-5)

        if i % 50 == 0:
            writer.add_scalar('cal/task_%d/loss' % task_id, loss, i)
            writer.add_scalar('cal/task_%d/temp' % task_id,
                              ttf(temp_param), i)

    final_temp = ttf(temp_param).data
    shared.softmax_temp[task_id] = final_temp.data

    logger.info('Calibrated softmax temperature of task %d is: %f.' % \
                (task_id+1, final_temp))

    logger.info('Temperature calibration for task %d ... Done' % (task_id+1))

if __name__ == '__main__':
    pass


