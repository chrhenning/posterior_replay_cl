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
# @title           :probabilistic/regression/train_utils.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :08/25/2019
# @version         :1.0
# @python_version  :3.6.8
"""
Helper functions for training scripts
-------------------------------------

The module :mod:`probabilistic.regression.train_utils` is used to improve
readibility and modularity of the training scripts such as
:mod:`probabilistic.regression.train_bbb`.
"""
from argparse import Namespace
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import os
import sys
from warnings import warn

from data.special.regression1d_data import ToyRegression
from mnets.mnet_interface import MainNetInterface
from hnets import hnet_helpers
from hnets.hnet_perturbation_wrapper import HPerturbWrapper
from hnets.structured_hmlp_examples import resnet_chunking, wrn_chunking
from probabilistic import prob_utils as putils
from probabilistic import GaussianBNNWrapper
from probabilistic.gauss_mlp import GaussianMLP
from probabilistic.gauss_hnet_init import gauss_hyperfan_init
from probabilistic.regression import hpsearch_config_bbb as hpbbb
from probabilistic.regression import hpsearch_config_avb as hpavb
from probabilistic.regression import hpsearch_config_ssge as hpssge
from probabilistic.regression import hpsearch_config_ewc as hpewc
import utils.misc as utils
import utils.sim_utils as sutils

def generate_tasks(config, writer):
    """Generate a set of predefined tasks.

    Args:
        writer: Tensorboard writer, in case plots should be logged.
        config: Command-line arguments.

    Returns:
        data_handlers: A list of data handlers.
        num_tasks: Number of generated tasks.
    """
    return generate_1d_tasks(show_plots=config.show_plots,
                             data_random_seed=config.data_random_seed,
                             writer=writer, task_set=config.used_task_set)

def generate_1d_tasks(show_plots=True, data_random_seed=42, writer=None,
                      task_set=1):
    """Generate a set of tasks for 1D regression.

    Args:
        show_plots: Visualize the generated datasets.
        data_random_seed: Random seed that should be applied to the
            synthetic data generation.
        writer: Tensorboard writer, in case plots should be logged.
        task_set (int): The set of tasks to be used. All sets are hard-coded
            inside this function.

    Returns:
        (tuple): Tuple containing:

        - **data_handlers**: A data handler for each task (instance of class
            :class:`data.special.regression1d_data.ToyRegression`).
        - **num_tasks**: Number of generated tasks.
    """
    if task_set == 0:
        # Here, we define as benchmark a typical dataset used in the
        # uncertainty literature:
        # Regression task y = x**3 + eps, where eps ~ N(0, 9*I).
        # For instance, see here:
        #   https://arxiv.org/pdf/1703.01961.pdf
        
        # How far outside the regime of the training data do we wanna predict
        # samples?
        test_offset = 1.5

        map_funcs = [lambda x : (x**3.)]
        num_tasks = len(map_funcs)
        x_domains = [[-4, 4]]
        std = 3 # 3**2 == 9
        num_train = 20

        # Range of pred. dist. plots.
        test_domains = [[x_domains[0][0] - test_offset,
                        x_domains[0][1] + test_offset]]

    elif task_set == 1:
        #test_offset = 1
        map_funcs = [lambda x : (x+3.),
                     lambda x : 2. * np.power(x, 2) - 1,
                     lambda x : np.power(x-3., 3)]
        num_tasks = len(map_funcs)
        x_domains = [[-4,-2], [-1,1], [2,4]]
        std = .05

        #test_domains = [[-4.1, -0.5], [-2.5,2.5], [.5, 4.1]]
        test_domains = [[-4.1, 4.1], [-4.1, 4.1], [-4.1, 4.1]]

        #m = 32 # magnitude
        #s = -.25 * np.pi
        #map_funcs = [lambda x : m*np.pi * np.sin(x + s),
        #             lambda x : m*np.pi * np.sin(x + s),
        #             lambda x : m*np.pi * np.sin(x + s)]
        #x_domains = [[-2*np.pi, -1*np.pi], [-0.5*np.pi, 0.5*np.pi],
        #             [1*np.pi, 2*np.pi]]
        #x_domains_test = [[-2*np.pi, 2*np.pi], [-2*np.pi, 2*np.pi],
        #                  [-2*np.pi, 2*np.pi]]
        #std = 3

        num_tasks = len(map_funcs)
        num_train = 20

    elif task_set == 2:
        map_funcs = [lambda x : np.power(x+3., 3),
                     lambda x : 2. * np.power(x, 2) - 1,
                     lambda x : -np.power(x-3., 3)]
        num_tasks = len(map_funcs)
        x_domains = [[-4,-2], [-1,1], [2,4]]
        std = .1

        test_domains = [[-4.1, 4.1], [-4.1, 4.1], [-4.1, 4.1]]

        num_tasks = len(map_funcs)
        num_train = 20

    elif task_set == 3:
        # Same as task set 2, but less aleatoric uncertainty.

        map_funcs = [lambda x : np.power(x+3., 3),
                     lambda x : 2. * np.power(x, 2) - 1,
                     lambda x : -np.power(x-3., 3)]
        num_tasks = len(map_funcs)
        x_domains = [[-4,-2], [-1,1], [2,4]]
        std = .05

        test_domains = [[-4.1, 4.1], [-4.1, 4.1], [-4.1, 4.1]]

        num_tasks = len(map_funcs)
        num_train = 20

    else:
        raise NotImplementedError('Set of tasks "%d" unknown!' % task_set)

    dhandlers = []
    for i in range(num_tasks):
        print('Generating %d-th task.' % (i))
        #test_inter = [x_domains[i][0] - test_offset,
        #              x_domains[i][1] + test_offset]
        dhandlers.append(ToyRegression(train_inter=x_domains[i],
            num_train=num_train, test_inter=test_domains[i], num_test=50,
            val_inter=x_domains[i], num_val=50,
            map_function=map_funcs[i], std=std, rseed=data_random_seed))

        if writer is not None:
            dhandlers[-1].plot_dataset(show=False)
            writer.add_figure('task_%d/dataset' % i, plt.gcf(),
                              close=not show_plots)
            if show_plots:
                utils.repair_canvas_and_show_fig(plt.gcf())

        elif show_plots:
            dhandlers[-1].plot_dataset()

    return dhandlers, num_tasks

def generate_gauss_networks(config, logger, data_handlers, device,
                            no_mnet_weights=None, create_hnet=True,
                            in_shape=None, out_shape=None, net_type='mlp',
                            non_gaussian=False):
    """Create main network and potentially the corresponding hypernetwork.

    The function will first create a normal MLP and then convert it into a
    network with Gaussian weight distribution by using the wrapper
    :class:`probabilistic.gauss_mnet_interface.GaussianBNNWrapper`.

    This function also takes care of weight initialization.

    Args:
        config: Command-line arguments.
        logger: Console (and file) logger.
        data_handlers: List of data handlers, one for each task. Needed to
            extract the number of inputs/outputs of the main network. And to
            infer the number of tasks.
        device: Torch device.
        no_mnet_weights (bool, optional): Whether the main network should not
            have trainable weights. If left unspecified, then the main network
            will only have trainable weights if ``create_hnet`` is ``False``.
        create_hnet (bool): Whether a hypernetwork should be constructed.
        in_shape (list, optional): Input shape that is passed to function
            :func:`utils.sim_utils.get_mnet_model` as argument ``in_shape``.
            If not specified, it is set to ``[data_handlers[0].in_shape[0]]``.
        out_shape (list, optional): Output shape that is passed to function
            :func:`utils.sim_utils.get_mnet_model` as argument ``out_shape``.
            If not specified, it is set to ``[data_handlers[0].out_shape[0]]``.
        net_type (str): See argument ``net_type`` of function
            :func:`utils.sim_utils.get_mnet_model`.
        non_gaussian (bool): If ``True``, then the main network will not be
            converted into a Gaussian network. Hence, networks remain
            deterministic.

    Returns:
        (tuple): Tuple containing:

        - **mnet**: Main network instance.
        - **hnet** (optional): Hypernetwork instance. This return value is
          ``None`` if no hypernetwork should be constructed.
    """
    assert not hasattr(config, 'mean_only') or config.mean_only == non_gaussian
    assert not non_gaussian or not config.local_reparam_trick
    assert not non_gaussian or not config.hyper_gauss_init

    num_tasks = len(data_handlers)

    # Should be set, except for regression.
    if in_shape is None or out_shape is None:
        assert in_shape is None and out_shape is None
        assert net_type == 'mlp'
        assert hasattr(config, 'multi_head')

        n_x = data_handlers[0].in_shape[0]
        n_y = data_handlers[0].out_shape[0]
        if config.multi_head:
            n_y = n_y * num_tasks

        in_shape = [n_x]
        out_shape = [n_y]

    ### Main network.
    logger.info('Creating main network ...')

    if no_mnet_weights is None:
        no_mnet_weights = create_hnet

    if config.local_reparam_trick:
        if net_type != 'mlp':
            raise NotImplementedError('The local reparametrization trick is ' +
                                      'only implemented for MLPs so far!')
            assert len(in_shape) == 1 and len(out_shape) == 1

        mlp_arch = utils.str_to_ints(config.mlp_arch)
        net_act = utils.str_to_act(config.net_act)
        mnet = GaussianMLP(n_in=in_shape[0], n_out=out_shape[0],
            hidden_layers=mlp_arch, activation_fn=net_act,
            use_bias=not config.no_bias, no_weights=no_mnet_weights).to(device)
    else:
        mnet_kwargs = {}
        if net_type == 'iresnet':
            mnet_kwargs['cutout_mod'] = True
        mnet =  sutils.get_mnet_model(config, net_type, in_shape, out_shape,
                                      device, no_weights=no_mnet_weights,
                                      **mnet_kwargs)

    # Initiaize main net weights, if any.
    assert(not hasattr(config, 'custom_network_init'))
    mnet.custom_init(normal_init=config.normal_init,
                     normal_std=config.std_normal_init, zero_bias=True)

    # Convert main net into Gaussian BNN.
    orig_mnet = mnet
    if not non_gaussian:
        mnet = GaussianBNNWrapper(mnet, no_mean_reinit=config.keep_orig_init,
            logvar_encoding=config.use_logvar_enc, apply_rho_offset=True,
            is_radial=config.radial_bnn).to(device)
    else:
        logger.debug('Created main network will not be converted into a ' +
                     'Gaussian main network.')

    ### Hypernet.
    hnet = None
    if create_hnet:
        logger.info('Creating hypernetwork ...')

        chunk_shapes, num_per_chunk, assembly_fct = None, None, None
        if config.hnet_type == 'structured_hmlp':
            if net_type == 'resnet':
                chunk_shapes, num_per_chunk, orig_assembly_fct = \
                    resnet_chunking(orig_mnet,
                                    gcd_chunking=config.shmlp_gcd_chunking)
            elif net_type == 'wrn':
                chunk_shapes, num_per_chunk, orig_assembly_fct = \
                    wrn_chunking(orig_mnet,
                        gcd_chunking=config.shmlp_gcd_chunking,
                        ignore_bn_weights=False, ignore_out_weights=False)
            else:
                raise NotImplementedError('"structured_hmlp" not implemented ' +
                                          'for network of type %s.' % net_type)

            if non_gaussian:
                assembly_fct = orig_assembly_fct
            else:
                chunk_shapes = chunk_shapes + chunk_shapes
                num_per_chunk = num_per_chunk + num_per_chunk

                def assembly_fct_gauss(list_of_chunks):
                    n = len(list_of_chunks)
                    mean_chunks = list_of_chunks[:n//2]
                    rho_chunks = list_of_chunks[n//2:]

                    return orig_assembly_fct(mean_chunks) + \
                        orig_assembly_fct(rho_chunks)

                assembly_fct = assembly_fct_gauss

        # For now, we either produce all or no weights with the hypernet.
        # Note, it can be that the mnet was produced with internal weights.
        assert mnet.hyper_shapes_learned is None or \
            len(mnet.param_shapes) == len(mnet.hyper_shapes_learned)

        hnet = sutils.get_hypernet(config, device, config.hnet_type,
            mnet.param_shapes, num_tasks, shmlp_chunk_shapes=chunk_shapes,
            shmlp_num_per_chunk=num_per_chunk, shmlp_assembly_fct=assembly_fct)

        if config.hnet_out_masking != 0:
            logger.info('Generating binary masks to select task-specific ' +
                        'subnetworks from hypernetwork.')
            # Add a wrapper around the hypernpetwork that masks its outputs
            # using a task-specific binary mask layer per layer. Note that
            # output weights are not masked.

            # Ensure that masks are kind of deterministic for a given hyper-
            # param config/task.
            mask_gen = torch.Generator()
            mask_gen = mask_gen.manual_seed(42)

            # Generate a random binary mask per task.
            assert len(mnet.param_shapes) == len(hnet.target_shapes)
            hnet_out_masks = []
            for tid in range(config.num_tasks):
                hnet_out_mask = []
                for layer_shapes, is_output in zip(mnet.param_shapes, \
                        mnet.get_output_weight_mask()):
                    layer_mask = torch.ones(layer_shapes)
                    if is_output is None:
                        # We only mask weights that are not output weights.
                        layer_mask = torch.rand(layer_shapes,
                                                generator=mask_gen)
                        layer_mask[layer_mask > config.hnet_out_masking] = 1
                        layer_mask[layer_mask <= config.hnet_out_masking] = 0
                    hnet_out_mask.append(layer_mask)
                hnet_out_masks.append(hnet_out_mask)

            hnet_out_masks = hnet.convert_out_format(hnet_out_masks,
                'sequential', 'flattened')

            def hnet_out_masking_func(hnet_out_int, uncond_input=None,
                                      cond_input=None, cond_id=None):
                assert isinstance(cond_id, (int, list))
                if isinstance(cond_id, int):
                    cond_id = [cond_id]

                hnet_out_int[hnet_out_masks[cond_id, :]==0] = 0
                return hnet_out_int

            def hnet_inp_handler(uncond_input=None, cond_input=None,
                                 cond_id=None): # Identity
                return uncond_input, cond_input, cond_id

            hnet = HPerturbWrapper(hnet, output_handler=hnet_out_masking_func,
                                   input_handler=hnet_inp_handler)

        #if config.hnet_type == 'structured_hmlp':
        #    print(num_per_chunk)
        #    for ii, int_hnet in enumerate(hnet.internal_hnets):
        #        print('   Internal hnet %d with %d outputs.' % \
        #              (ii, int_hnet.num_outputs))

        ### Initialize hypernetwork.
        if not config.hyper_gauss_init:
            apply_custom_hnet_init(config, logger, hnet)
        else:
            # Initialize task embeddings, if any.
            hnet_helpers.init_conditional_embeddings(hnet,
                normal_std=config.std_normal_temb)

            gauss_hyperfan_init(hnet, mnet=mnet, use_xavier=True,
                                cond_var=config.std_normal_temb**2,
                                keep_hyperfan_mean=config.keep_orig_init)

    return mnet, hnet

def apply_custom_hnet_init(config, logger, hnet):
    """Applying a custom hypernetwork init (based on user configs).

    Note, this method might not be safe to use, as it is not customized to
    network internals.

    Args:
        config: Command-line arguments.
        logger: Console (and file) logger.
        hnet: Hypernetwork object.
    """
    assert not hasattr(config, 'custom_network_init')

    if isinstance(hnet, MainNetInterface):
        hnet.custom_init(normal_init=config.normal_init,
                         normal_std=config.std_normal_init, zero_bias=True)

        # Initialize task embeddings, if any.
        hnet_helpers.init_conditional_embeddings(hnet,
            normal_std=config.std_normal_temb)

        # Initialize chunk embeddings, if any.
        hnet_helpers.init_chunk_embeddings(hnet,
            normal_std=config.std_normal_emb)
    else:
        # Legacy support.
        logger.warning('Applying unsafe custom network initialization. This ' +
                       'init does not take the class internal structure ' +
                       'into account and fails, for instance, when using ' +
                       'batch or spectral norm.')
        init_params = list(hnet.parameters())

        for W in init_params:
            # FIXME not all 1D vectors are bias vectors (e.g., batch norm weights).
            if W.ndimension() == 1: # Bias vector.
                torch.nn.init.constant_(W, 0)
            elif config.normal_init:
                torch.nn.init.normal_(W, mean=0, std=config.std_normal_init)
            else:
                torch.nn.init.xavier_uniform_(W)

        # Note, the embedding vectors from the partitioned hypernet have been
        # considered as bias vectors and thus initialized to zero.
        if hasattr(hnet, 'chunk_embeddings'):
            for emb in hnet.chunk_embeddings:
                torch.nn.init.normal_(emb, mean=0, std=config.std_normal_emb)

        # Also the task embeddings are initialized differently.
        if hnet.has_task_embs:
            for temb in hnet.get_task_embs():
                torch.nn.init.normal_(temb, mean=0., std=config.std_normal_temb)

def plot_predictive_distribution(data, inputs, predictions, show_raw_pred=False,
                                 figsize=(10, 6), show=True):
    """Plot the predictive distribution of a single regression task.

    Args:
        data: The dataset handler (class `ToyRegression`).
        inputs: A 2D numpy array, denoting the inputs used to generate the
            `predictions`.
        predictions: A 2D numpy array with dimensions (batch_size x sample_size)
            where the sample size refers to the number of weight samples that
            have been used to produce an ensemble of predictions.
        show_raw_pred: Whether a second subplot should be shown, in which the
            standard deviations of the predictions are ignored and only the mean
            is shown.
        figsize: A tuple, determining the size of the figure in inches.
        show: Whether the plot should be shown.
    """
    assert(isinstance(data, ToyRegression))
    colors = utils.get_colorbrewer2_colors(family='Dark2')

    num_plots = 2 if show_raw_pred else 1
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=figsize)

    train_x = data.get_train_inputs().squeeze()
    train_y = data.get_train_outputs().squeeze()

    #test_x = data.get_test_inputs().squeeze()
    #test_y = data.get_test_outputs().squeeze()

    sample_x, sample_y = data._get_function_vals()

    for i, ax in enumerate(axes):
        # The default matplotlib setting is usually too high for most plots.
        ax.locator_params(axis='y', nbins=2)
        ax.locator_params(axis='x', nbins=6)
        
        ax.plot(sample_x, sample_y, color='k', label='f(x)',
                 linestyle='dashed', linewidth=.5)
        ax.plot(train_x, train_y, 'o', color='k', label='Train')
        #plt.plot(test_x, test_y, 'o', color=colors[1], label='Test')
    
        inputs = inputs.squeeze()
        mean_pred = predictions.mean(axis=1)
        std_pred = predictions.std(axis=1)
    
        c = colors[2]
        ax.plot(inputs, mean_pred, color=c, label='Pred')
        if i == 0:
            ax.fill_between(inputs, mean_pred + std_pred, mean_pred - std_pred,
                            color=c, alpha=0.3)
            ax.fill_between(inputs, mean_pred + 2.*std_pred,
                            mean_pred - 2.*std_pred, color=c, alpha=0.2)
            ax.fill_between(inputs, mean_pred + 3.*std_pred,
                            mean_pred - 3.*std_pred, color=c, alpha=0.1)

        ax.legend()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        if i == 1:
            ax.set_title('Mean Predictions')
        else:
            ax.set_title('Predictive Distribution')

    if show:
        plt.show()

def plot_mse(config, writer, num_tasks, current_mse, during_mse=None,
             baselines=None, save_fig=True, summary_label='test/mse'):
    """Produce a scatter plot that shows the current and immediate mse values
    (and maybe a set of baselines) of each task. This visualization helps to
    understand the impact of forgetting.

    Args:
        config: Command-line arguments.
        writer: Tensorboard summary writer.
        num_tasks: Number of tasks.
        current_mse: Array of MSE values currently achieved.
        during_mse (optional): Array of MSE values achieved right after
            training on the corresponding task.
        baselines (optional): A dictionary of label names mapping onto arrays.
            Can be used to plot additional baselines.
        save_fig: Whether the figure should be saved in the output folder.
        summary_label: Label used for the figure when writing it to tensorboard.
    """
    x_vals = np.arange(1, num_tasks+1)

    num_plots = 1
    if during_mse is not None:
        num_plots += 1
    if baselines is not None:
        num_plots += len(baselines.keys())

    colors = utils.get_colorbrewer2_colors(family='Dark2')
    if num_plots > len(colors):
        warn('Changing to automatic color scheme as we don\'t have ' +
             'as many manual colors as tasks.')
        colors = cm.rainbow(np.linspace(0, 1, num_plots))

    fig, axes = plt.subplots(figsize=(10, 6))
    plt.title('Current MSE on each task')

    plt.scatter(x_vals, current_mse, color=colors[0], label='Current Val MSE')

    if during_mse is not None:
        plt.scatter(x_vals, during_mse, label='During MSE',
                    color=colors[1], marker='*')

    if baselines is not None:
        for i, (label, vals) in enumerate(baselines.items()):
            plt.scatter(x_vals, vals, label=label, color=colors[2+i],
                        marker='x')

    plt.ylabel('MSE')
    plt.xlabel('Task')
    plt.xticks(x_vals)
    plt.legend()

    if save_fig:
        plt.savefig(os.path.join(config.out_dir, 'mse_%d' % num_tasks),
                    bbox_inches='tight')

    writer.add_figure(summary_label, plt.gcf(), num_tasks,
                      close=not config.show_plots)
    if config.show_plots:
        utils.repair_canvas_and_show_fig(plt.gcf())

def compute_mse(task_id, data, mnet, hnet, device, config, shared, hhnet=None,
                split_type='test', return_dataset=False,
                return_predictions=False, return_samples=False,
                disable_lrt=False, normal_post=None):
    r"""Compute the MSE over a specified dataset split.

    Note, this method does not execute the code within a ``torch.no_grad()``
    context. This needs to be handled from the outside if desired.

    The number of weight samples evaluated by this method is determined by the
    argument ``config.val_sample_size``, even for the training split!

    We expect the networks to be in the correct state (usually the `eval`
    state).

    The ``task_id`` is used only to select the hypernet embedding (and the
    correct output head in case of a multi-head setup).

    **Remarks on the MSE value**

    Ideally, we would like to compute the MSE by sampling form the predictive
    posterior for a test sample :math:`(x, y)` as follows

    .. math::

        \mathbb{E}_{\hat{y} \sim p(y \mid \mathcal{D}; x)} \
            \big[ (\hat{y} - y)^2 \big]

    However, since we don't want to sample from the Gaussian likelihood, we
    look at two simplifications of the MSE that only require the evaluation of
    the mean of the likelihood.

    In the first one, we compute the MSE per sampled model and average over
    these MSEs and in the second, we compute the mean over the likelihood means
    and compute the MSE using these "mean" predictions. The relation between
    these two can be shown to be as follows:

    .. math::
        
         \mathbb{E}_{p(W \mid \mathcal{D})} \big[ (f(x, w) - y)^2 \big] = \
             \text{Var}\big( f(x, w) \big) + \
             \Big( \mathbb{E}_{p(W \mid \mathcal{D})} \big[ f(x, w) \big] - \
             y\Big)^2 

    We prefer the first method as it respects the variance of the predictive
    distribution. If we would like to use the MSE as a measure for
    hyperparameter selection, then the model that leads to lower in-distribution
    uncertainty should be preferred (to be precise, the MSE would then have to
    be computed on the training data).

    Args:
        (....): See docstring of method :func:`train`. Note, ``hnet`` can be
            passed as ``None``.
        hhnet (optional): Hyper-hypernetwork.
        split_type: The name of the dataset split that should be used:

            - ``'test'``: The test set will be used.
            - ``'val'``: The validation set will be used. If not available, the
              test set will be used.
            - ``'train'``: The training set will be used.

        return_dataset: If ``True``, the attributes ``inputs`` and ``targets``
            will be added to the ``return_vals`` Namespace (see return values).
            Both fields will be filled with numpy arrays.

            Note:

                The ``inputs`` and ``targets`` are returned as they are stored
                in the ``data`` dataset handler. I.e., for 1D regression data
                the shapes would be ``[num_samples, 1]``.

        return_predictions: If ``True``, the attribute ``predictions`` will be
            added to the ``return_vals`` Namespace (see return values). These
            fields will correspond to main net outputs for each sample. The
            field will be filled with a numpy array.
        return_samples: If ``True``, the attribute ``samples`` will be added
            to the ``return_vals`` Namespace (see return values). This field
            will contain all weight samples used.
            The field will be filled with a numpy array.
        disable_lrt (bool): Disable the local-reparametrization trick in the
            forward pass of the main network (if it uses it).
        normal_post (tuple, optional): A tuple of lists. The lists have the
            length of the parameter list of the main network. The first list
            represents mean and the second stds of a normal posterior
            distribution. If provided, weights are sampled from this
            distribution.

    Returns:
        (tuple): Tuple containing:

        - ``mse``: The mean over the return value ``mse_vals``.
        - ``return_vals``: A namespace object that contains several attributes,
          depending on the arguments passed. It will always contains:

            - ``w_mean``: ``None`` if the hypernetwork encodes an implicit
              distribution. Otherwise, the current mean values of all synapses
              in the main network.
            - ``w_std``: ``None`` if the hypernetwork encodes an implicit
              distribution. Otherwise, the current standard deviations of all
              synapses in the main network or ``None`` if the main network has
              deterministic weights.
            - ``w_hnet``: The output of the hyper-hypernetwork ``hhnet`` for the
              current task, if applicable. ``None`` otherwise.
            - ``mse_vals``: Numpy array of MSE values per weight sample.
    """
    assert split_type in ['test', 'val', 'train']
    assert np.prod(data.out_shape) == 1 # Method expects 1D regression task.
    assert normal_post is None or hnet is None and hhnet is None

    return_vals = Namespace()

    allowed_outputs = None
    if config.multi_head:
        n_y = data.out_shape[0]
        allowed_outputs = list(range(task_id*n_y, (task_id+1)*n_y))

    if split_type == 'train':
        X = data.get_train_inputs()
        T = data.get_train_outputs()
    elif split_type == 'test' or data.num_val_samples == 0:
        X = data.get_test_inputs()
        T = data.get_test_outputs()
    else:
        X = data.get_val_inputs()
        T = data.get_val_outputs()

    if return_dataset:
        return_vals.inputs = X
        return_vals.targets = T

    X = data.input_to_torch_tensor(X, device)
    T = data.output_to_torch_tensor(T, device)

    gauss_main = False
    if isinstance(mnet, GaussianBNNWrapper):
        assert hhnet is None
        gauss_main = True

    # Whether the input to the hnet is a task embedding or latent sample z?
    # Note, also a Gaussian main net has a deterministic hnet.
    deterministic_hnet = False
    # Note, `conditional_params` would correpond to task embeddings.
    if hnet is not None and hnet.num_known_conds > 0:
        deterministic_hnet = True

    hnet_out = None
    hnet_weights = None
    if hnet is not None:
        if hhnet is not None:
            hnet_weights = hhnet.forward(cond_id=task_id)

        if deterministic_hnet:
            hnet_out = hnet.forward(cond_id=task_id, weights=hnet_weights)

    return_vals.w_mean = None
    return_vals.w_std = None
    return_vals.w_hnet = hnet_weights
    if gauss_main:
        w_mean, w_rho = mnet.extract_mean_and_rho(weights=hnet_out)
        w_std = putils.decode_diag_gauss(w_rho, logvar_enc=mnet.logvar_encoding)

        return_vals.w_mean = w_mean
        return_vals.w_std = w_std
    elif deterministic_hnet:
        w_mean = hnet_out

        return_vals.w_mean = w_mean
        return_vals.w_std = None

    if return_samples:
        num_w = mnet.num_params
        if gauss_main:
            num_w = num_w // 2
        return_vals.samples = np.empty((config.val_sample_size, num_w))

    mse_vals = np.empty(config.val_sample_size)
    if return_predictions:
        predictions = np.empty((X.shape[0], config.val_sample_size))

    for j in range(config.val_sample_size):
        weights = None
        if normal_post is not None: # Sample weights from a Gaussian posterior.
            weights = []
            for ii, pmean in enumerate(normal_post[0]):
                pstd = normal_post[1][ii]
                weights.append(torch.normal(pmean, pstd, generator=None))
            Y = mnet.forward(X, weights=weights)
        elif gauss_main:
            Y = mnet.forward(X, weights=None, mean_only=False,
                             extracted_mean=w_mean, extracted_rho=w_rho,
                             disable_lrt=disable_lrt)
        elif deterministic_hnet:
            # FIXME Wasteful computation - same predictions every iteration
            Y = mnet.forward(X, weights=w_mean)
        else:
            if hnet is not None: # Implicit hypernetwork.
                z = torch.normal(torch.zeros(1, shared.noise_dim),
                                 config.latent_std).to(device)
                weights = hnet.forward(uncond_input=z, weights=hnet_weights)
            else: # Main network only training.
                # FIXME Wasteful computation - same predictions every iteration
                weights = None

            Y = mnet.forward(X, weights=weights)
        if config.multi_head:
            Y = Y[:, allowed_outputs]

        mse_vals[j] = F.mse_loss(Y, T).cpu().numpy()
        if return_predictions:
            predictions[:, j] = Y.cpu().numpy().squeeze()

        if return_samples:
            if weights is None:
                return_vals.samples = None
            else:
                return_vals.samples[j, :] = torch.cat([p.detach().flatten() \
                    for p in weights]).cpu().numpy()

    return_vals.mse_vals = mse_vals

    if return_predictions:
        #return_vals.preds_mean = predictions.mean(axis=1)
        #return_vals.preds_std = predictions.std(axis=1)
        return_vals.predictions = predictions

    return mse_vals.mean(), return_vals

def plot_predictive_distributions(config, writer, data_handlers, inputs,
                                  preds_mean, preds_std, save_fig=True,
                                  publication_style=False):
    """Plot the predictive distribution of several tasks into one plot.

    Args:
        config: Command-line arguments.
        writer: Tensorboard summary writer.
        data_handlers: A set of data loaders.
        inputs: A list of arrays containing the x values per task.
        preds_mean: The mean predictions corresponding to each task in `inputs`.
        preds_std: The std of all predictions in `preds_mean`.
        save_fig: Whether the figure should be saved in the output folder.
        publication_style: whether plots should be made in publication style.
    """
    num_tasks = len(data_handlers)
    assert(len(inputs) == num_tasks)
    assert(len(preds_mean) == num_tasks)
    assert(len(preds_std) == num_tasks)

    colors = utils.get_colorbrewer2_colors(family='Dark2')
    if num_tasks > len(colors):
        warn('Changing to automatic color scheme as we don\'t have ' +
             'as many manual colors as tasks.')
        colors = cm.rainbow(np.linspace(0, 1, num_tasks))

    fig, axes = plt.subplots(figsize=(12, 6))

    if publication_style:
        ts, lw, ms = 60, 15, 10 # text fontsize, line width, marker size
    else:
        ts, lw, ms = 12, 5, 8

    # The default matplotlib setting is usually too high for most plots.
    plt.locator_params(axis='y', nbins=2)
    plt.locator_params(axis='x', nbins=6)

    for i, data in enumerate(data_handlers):
        assert(isinstance(data, ToyRegression))

        # In what range to plot the real function values?
        train_range = data.train_x_range
        range_offset = (train_range[1] - train_range[0]) * 0.05
        sample_x, sample_y = data._get_function_vals( \
            x_range=[train_range[0]-range_offset, train_range[1]+range_offset])
        #sample_x, sample_y = data._get_function_vals()

        #plt.plot(sample_x, sample_y, color='k', label='f(x)',
        #         linestyle='dashed', linewidth=.5)
        plt.plot(sample_x, sample_y, color='k',
                 linestyle='dashed', linewidth=lw/7.)

        train_x = data.get_train_inputs().squeeze()
        train_y = data.get_train_outputs().squeeze()

        if i == 0:
            plt.plot(train_x, train_y, 'o', color='k', label='Training Data', 
                markersize=ms)
        else:
            plt.plot(train_x, train_y, 'o', color='k', markersize=ms)

        plt.plot(inputs[i], preds_mean[i], color=colors[i],
                 label='Task %d' % (i+1), lw=lw/3.)

        plt.fill_between(inputs[i], preds_mean[i] + preds_std[i],
            preds_mean[i] - preds_std[i], color=colors[i], alpha=0.3)
        plt.fill_between(inputs[i], preds_mean[i] + 2.*preds_std[i],
            preds_mean[i] - 2.*preds_std[i], color=colors[i], alpha=0.2)
        plt.fill_between(inputs[i], preds_mean[i] + 3.*preds_std[i],
            preds_mean[i] - 3.*preds_std[i], color=colors[i], alpha=0.1)

    if publication_style:
        axes.grid(False)
        axes.set_facecolor('w')
        axes.set_ylim([-2.5, 3])
        axes.axhline(y=axes.get_ylim()[0], color='k', lw=lw)
        axes.axvline(x=axes.get_xlim()[0], color='k', lw=lw)
        if len(data_handlers)==3:
            plt.yticks([-2, 0, 2], fontsize=ts)
            plt.xticks([-3, 0, 3], fontsize=ts)
        else:
            for tick in axes.yaxis.get_major_ticks():
                tick.label.set_fontsize(ts) 
            for tick in axes.xaxis.get_major_ticks():
                tick.label.set_fontsize(ts) 
        axes.tick_params(axis='both', length=lw, direction='out', width=lw/2.)
        if config.train_from_scratch:
            plt.title('training from scratch', fontsize=ts, pad=ts)
        elif config.beta == 0:
            plt.title('fine-tuning', fontsize=ts, pad=ts)
        else: 
            plt.title('CL with prob. hnet reg.', fontsize=ts, pad=ts)
    else:
        plt.legend()
        plt.title('Predictive distributions', fontsize=ts, pad=ts)

    plt.xlabel('$x$', fontsize=ts)
    plt.ylabel('$y$', fontsize=ts)

    if save_fig:
        plt.savefig(os.path.join(config.out_dir, 'pred_dists_%d' % num_tasks),
                    bbox_inches='tight')

    writer.add_figure('test/pred_dists', plt.gcf(), num_tasks,
                      close=not config.show_plots)
    if config.show_plots:
        utils.repair_canvas_and_show_fig(plt.gcf())

def calc_batch_uncertainty(config, task_id, inputs, mnet, hnet, num_w_samples,
                           mnet_weights=None, allowed_outputs=None):
    """Compute the per-sample uncertainties for a given batch of inputs.

    Note:
        This function currently assumes a Gaussian weight posterior.

    Note:
        This function is executed inside a ``torch.no_grad()`` context.

    Args:
        config (argparse.Namespace): Command-line arguments.
        task_id (int): In case a hypernet ``hnet`` is given, the ``task_id`` is
            used to load the corresponding main network ``mnet`` weights.
        inputs (torch.Tensor): A batch of main network ``mnet`` inputs.
        mnet: The main network.
        hnet (optional): The hypernetwork, can be ``None``.
        num_w_samples (int): The number of weight samples that should be drawn
            to estimate predictive uncertainty.
        mnet_weights (tuple, optional): To save computation, one can pass
            weights for the main network here, if they have been computed
            prior to calling this methods.

            Example:

                If the main network is an instance of class
                :class:`probabilistic.gauss_mnet_interface.GaussianBNNWrapper`,
                then ``mnet_weights`` should be a tuple of means and rhos of
                the Gaussian weight posterior
                (``mnet_weights=(w_mean, w_rho)``).
        allowed_outputs (tuple, optional): The indices of the neurons belonging
            to outputs head ``task_id``. Only needs to be specified in a
            multi-head setting.

    Returns:
        (numpy.ndarray): In case of a regression task, the array will contain
        the std of predictions made by the ``mnet`` (i.e., the uncertainty), one
        value for each input.
    """
    assert not config.multi_head or allowed_outputs is not None
    assert isinstance(mnet, GaussianBNNWrapper)
    if inputs.shape[1] != 1:
        raise ValueError('This method is currently only implemented to deal ' +
                         'with 1D regression tasks.')

    with torch.no_grad():
        if mnet_weights is None:
            if hnet is None:
                hnet_out = None
            else:
                hnet_out = hnet.forward(cond_id=task_id)
            w_mean, w_rho = mnet.extract_mean_and_rho(weights=hnet_out)
        else:
            assert len(mnet_weights) == 2
            w_mean, w_rho = mnet_weights

        predictions = np.empty((inputs.shape[0], num_w_samples))

        for j in range(num_w_samples):
            # Note, sampling happens inside the forward method for Gaussian
            # networks.
            Y = mnet.forward(inputs, weights=None, mean_only=False,
                             extracted_mean=w_mean, extracted_rho=w_rho)
            if config.multi_head:
                Y = Y[:, allowed_outputs]

            predictions[:, j] = Y.cpu().numpy().squeeze()

        uncertainties = predictions.std(axis=1)

        return uncertainties

def setup_summary_dict(config, shared, experiment, num_tasks, mnet, hnet=None,
                       hhnet=None, dis=None):
    """Setup the summary dictionary that is written to the performance
    summary file (in the result folder).

    This method adds the keyword "summary" to `shared`.

    Args:
        config: Command-line arguments.
        shared: Miscellaneous data shared among training functions (summary dict
            will be added to this :class:`argparse.Namespace`).
        experiment (str): Type of experiment. Possible values are:

            - ``'bbb'``: Assuming summary values from
              :mod:`probabilistic.regression.hpsearch_config_bbb`.
            - ``'avb'``: Assuming summary values from
              :mod:`probabilistic.regression.hpsearch_config_avb`.
            - ``'ssge'``: Assuming summary values from
              :mod:`probabilistic.regression.hpsearch_config_ssge`.
            - ``'ewc'``: Assuming summary values from
              :mod:`probabilistic.regression.hpsearch_config_ewc`.
        num_tasks (int): Number of tasks.
        mnet: Main network.
        hnet (optional): Hypernetwork.
        hhnet (optional): Hyper-hypernetwork.
        dnet (optional): Discriminator.
    """
    assert(experiment in ['bbb', 'avb', 'ssge', 'ewc', 'mt'])

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

    # Note, we assume that all configs have the exact same keywords.
    if experiment == 'bbb':
        summary_keys = hpbbb._SUMMARY_KEYWORDS
    elif experiment == 'ewc':
        summary_keys = hpewc._SUMMARY_KEYWORDS
    elif experiment == 'mt':
        summary_keys = hpmt._SUMMARY_KEYWORDS
    elif experiment == 'avb':
        summary_keys = hpavb._SUMMARY_KEYWORDS
    else:
        summary_keys = hpssge._SUMMARY_KEYWORDS
        assert experiment == 'ssge'

    for k in summary_keys:
        if k == 'aa_mse_during' or \
                k == 'aa_mse_final' or \
                k == 'aa_task_inference' or \
                k == 'aa_mse_during_inferred' or \
                k == 'aa_mse_final_inferred' or \
                k == 'aa_acc_dis':
            summary[k] = [-1] * num_tasks

        elif k == 'aa_mse_during_mean' or \
                k == 'aa_mse_final_mean' or \
                k == 'aa_task_inference_mean' or \
                k == 'aa_mse_during_inferred_mean' or \
                k == 'aa_mse_final_inferred_mean' or \
                k == 'aa_acc_avg_dis':
            summary[k] = -1

        elif k == 'aa_num_weights_main':
            summary[k] = mnum
        elif k == 'aa_num_weights_hyper':
            summary[k] = hnum
        elif k == 'aa_num_weights_hyper_hyper':
            summary[k] = hhnum
        elif k == 'aa_num_weights_dis':
            summary[k] = dnum

        elif experiment == 'bbb' and k == 'aa_num_weights_ratio' or \
                experiment in ['avb', 'ssge'] and \
                k == 'aa_num_weights_hm_ratio':
            summary[k] = hm_ratio
        elif k == 'aa_num_weights_hhm_ratio':
            summary[k] = hhm_ratio
        elif k == 'aa_num_weights_dm_ratio':
            summary[k] = dm_ratio


        elif k == 'finished':
            summary[k] = 0
        else:
            # Implementation must have changed if this exception is
            # raised.
            raise ValueError('Summary argument %s unknown!' % k)

    shared.summary = summary

def save_summary_dict(config, shared):
    """Write a text file in the result folder that gives a quick
    overview over the results achieved so far.

    Args:
        (....): See docstring of function :func:`setup_summary_dict`.
    """
    # "setup_summary_dict" must be called first.
    assert(hasattr(shared, 'summary'))

    summary_fn = 'performance_overview.txt'
    #summary_fn = hpbbb._SUMMARY_FILENAME

    with open(os.path.join(config.out_dir, summary_fn), 'w') as f:
        for k, v in shared.summary.items():
            if isinstance(v, list):
                f.write('%s %s\n' % (k, utils.list_to_str(v)))
            elif isinstance(v, float):
                f.write('%s %f\n' % (k, v))
            else:
                f.write('%s %d\n' % (k, v))

def backup_cli_command(config):
    """Write the curret CLI call into a script.

    This will make it very easy to reproduce a run, by just copying the call
    from the script in the output folder. However, this call might be ambiguous
    in case default values have changed. In contrast, all default values are
    backed up in the file ``config.json``.

    Args:
        (....): See docstring of function :func:`setup_summary_dict`.
    """
    script_name = sys.argv[0]
    run_args = sys.argv[1:]
    command = 'python3 ' + script_name
    # FIXME Call reconstruction fails if user passed strings with white spaces.
    for arg in run_args:
        command += ' ' + arg

    fn_script = os.path.join(config.out_dir, 'cli_call.sh')

    with open(fn_script, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('# The user invoked CLI call that caused the creation of\n')
        f.write('# this output folder.\n')
        f.write(command)

if __name__ == '__main__':
    pass


