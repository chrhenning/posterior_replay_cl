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
# @title          :probabilistic/prob_gmm/train_utils.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :03/10/2020
# @version        :1.0
# @python_version :3.6.10
"""
Training utilities
------------------

A collection of helper functions for training scripts in this subpackage.
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.special import gaussian_mixture_data as gauss_mod
from data.special.gaussian_mixture_data import get_gmm_tasks
from data.special.gmm_data import GMMData
from probabilistic import prob_utils as putils
from probabilistic.prob_cifar import train_utils as pcutils
from probabilistic.prob_mnist import train_utils as pmutils
from utils import misc

def generate_datasets(config, logger, writer):
    """Create a data handler per task.

    Note:
        The datasets are hard-coded in this function.

    Args:
        config (argparse.Namespace): Command-line arguments. This function will
            add the key ``num_tasks`` to this namespace if not existing.

            Note, this function will also add the keys ``gmm_grid_size``,
            ``gmm_grid_range_1``, ``gmm_grid_range_2``, which are used for
            plotting.

        logger: Logger object.
        writer (tensorboardX.SummaryWriter): Tensorboard logger.

    Returns:
        (list): A list of data handlers.
    """
    NUM_TRAIN = 10
    NUM_TEST = 100

    config.gmm_grid_size = 250

    dhandlers = []

    TASK_SET = 3

    if TASK_SET == 0:
        config.gmm_grid_range_1 = config.gmm_grid_range_2 = [-1, 1]
        means = [
            [np.array([0, 1]), np.array([0, -1])]
        ]
        variances = [
            [0.05**2 * np.eye(len(mean)) for mean in means[0]]
        ]
    elif TASK_SET == 1:
        config.gmm_grid_range_1 = config.gmm_grid_range_2 = [-6, 6]

        means = [gauss_mod.CHE_MEANS[i:i+2] for i in range(0, 6, 2)]
        variances = [gauss_mod.CHE_VARIANCES[i:i+2] for i in range(0, 6, 2)]
    elif TASK_SET == 2:
        config.gmm_grid_range_1 = config.gmm_grid_range_2 = [-9, 9]

        means = [gauss_mod.CHE_MEANS[i:i+2] for i in range(0, 6, 2)]
        variances = [[1.**2 * np.eye(len(m)) for m in mm] for mm in means]
    elif TASK_SET == 3:
        config.gmm_grid_range_1 = config.gmm_grid_range_2 = [-9, 9]

        means = [gauss_mod.CHE_MEANS[i:i+2] for i in range(0, 6, 2)]
        variances = [[.2**2 * np.eye(len(m)) for m in mm] for mm in means]
    else:
        raise NotImplementedError()

    # Note, this is a synthetic dataset where the number of tasks and the
    # number of classes per tasks is hard-coded inside this function.
    if hasattr(config, 'num_tasks') and config.num_tasks > len(means):
        raise ValueError('Command-line argument "num_tasks" has impossible ' +
                         'value %d (maximum value would be %d).' %
                         (config.num_tasks, len(means)))
    elif not hasattr(config, 'num_tasks'):
        config.num_tasks = len(means)
    else:
        means = means[:config.num_tasks]
        variances = variances[:config.num_tasks]

    if hasattr(config, 'num_classes_per_task'):
        raise ValueError('Command-line argument "num_classes_per_task" ' +
                         'cannot be considered by this function.')

    if hasattr(config, 'val_set_size') and config.val_set_size > 0:
        raise ValueError('GMM Dataset does not support a validation set!')

    show_plots = False
    if hasattr(config, 'show_plots'):
        show_plots = config.show_plots

    # For multiple tasks, generate a combined dataset just to create some plots.
    gauss_bumps_all = get_gmm_tasks(means=list(itertools.chain(*means)),
        covs=list(itertools.chain(*variances)), num_train=NUM_TRAIN,
        num_test=NUM_TEST, map_functions=None,
        rseed=config.data_random_seed)
    if config.num_tasks > 1:
        full_data = GMMData(gauss_bumps_all, classification=True,
                            use_one_hot=True, mixing_coefficients=None)

        input_mesh = full_data.get_input_mesh(x1_range=config.gmm_grid_range_1,
            x2_range=config.gmm_grid_range_2, grid_size=config.gmm_grid_size)

        # Plot data distribution.
        if writer is not None:
            full_data.plot_uncertainty_map(title='All Data',
                     input_mesh=input_mesh, use_generative_uncertainty=True,
                     sketch_components=False, show=False)
            writer.add_figure('all_tasks/data_dist', plt.gcf(),
                              close=not show_plots)
            if show_plots:
                misc.repair_canvas_and_show_fig(plt.gcf())

        # Plot ground-truth conditional uncertainty.
        if writer is not None:
            full_data.plot_uncertainty_map(title='Conditional Uncertainty',
                     input_mesh=input_mesh, sketch_components=True, show=False)
            writer.add_figure('all_tasks/cond_entropy', plt.gcf(),
                              close=not show_plots)
            if show_plots:
                misc.repair_canvas_and_show_fig(plt.gcf())

        # Plot ground-truth class boundaries.
        if writer is not None:
            full_data.plot_optimal_classification(title='Class-Boundaries',
                     input_mesh=input_mesh, sketch_components=True, show=False)
            writer.add_figure('all_tasks/class_boundaries', plt.gcf(),
                              close=not show_plots)
            if show_plots:
                misc.repair_canvas_and_show_fig(plt.gcf())

        # Plot ground-truth class boundaries together with all training data.
        # Note, that might visualize training points the would even be
        # misclassified by the true underlying model (due to the stochastic
        # drawing of samples).
        if writer is not None:
            full_data.plot_optimal_classification(
                title='Class-Boundaries - Training Data',
                input_mesh=input_mesh, sketch_components=True, show=False,
                sample_inputs=full_data.get_train_inputs(),
                sample_modes=np.argmax(full_data.get_train_outputs(), axis=1))
            writer.add_figure('all_tasks/class_boundaries_train', plt.gcf(),
                              close=not show_plots)
            if show_plots:
                misc.repair_canvas_and_show_fig(plt.gcf())

        # Plot ground-truth class boundaries together with all test data.
        if writer is not None:
            full_data.plot_optimal_classification(
                title='Class-Boundaries - Test Data',
                input_mesh=input_mesh, sketch_components=True, show=False,
                sample_inputs=full_data.get_test_inputs(),
                sample_modes=np.argmax(full_data.get_test_outputs(), axis=1))
            writer.add_figure('all_tasks/class_boundaries_test', plt.gcf(),
                              close=not show_plots)
            if show_plots:
                misc.repair_canvas_and_show_fig(plt.gcf())

    # Create individual task datasets.
    ii = 0
    for i in range(len(means)):
        gauss_bumps = gauss_bumps_all[ii:ii+len(means[i])]
        ii += len(means[i])

        dhandlers.append(GMMData(gauss_bumps, classification=True,
                                 use_one_hot=True, mixing_coefficients=None))

        input_mesh = dhandlers[-1].get_input_mesh( \
            x1_range=config.gmm_grid_range_1, x2_range=config.gmm_grid_range_2,
            grid_size=config.gmm_grid_size)

        # Plot training data.
        if writer is not None:
            dhandlers[-1].plot_uncertainty_map(title='Training Data',
                     input_mesh=input_mesh, use_generative_uncertainty=True,
                     sample_inputs=dhandlers[-1].get_train_inputs(),
                     sample_modes=np.argmax(dhandlers[-1].get_train_outputs(), \
                                            axis=1),
                     #sample_label='Training data',
                     sketch_components=True, show=False)
            writer.add_figure('task_%d/train_data' % i, plt.gcf(),
                              close=not show_plots)
            if show_plots:
                misc.repair_canvas_and_show_fig(plt.gcf())

        # Plot test data.
        if writer is not None:
            dhandlers[-1].plot_uncertainty_map(title='Test Data',
                     input_mesh=input_mesh, use_generative_uncertainty=True,
                     sample_inputs=dhandlers[-1].get_test_inputs(),
                     sample_modes=np.argmax(dhandlers[-1].get_test_outputs(), \
                                            axis=1),
                     #sample_label='Training data',
                     sketch_components=True, show=False)
            writer.add_figure('task_%d/test_data' % i, plt.gcf(),
                              close=not show_plots)
            if show_plots:
                misc.repair_canvas_and_show_fig(plt.gcf())

        # Plot ground-truth conditional uncertainty.
        if writer is not None:
            dhandlers[-1].plot_uncertainty_map(title='Conditional Uncertainty',
                     input_mesh=input_mesh, sketch_components=True, show=False)
            writer.add_figure('task_%d/cond_entropy' % i, plt.gcf(),
                              close=not show_plots)
            if show_plots:
                misc.repair_canvas_and_show_fig(plt.gcf())

        # Plot ground-truth class boundaries.
        if writer is not None:
            dhandlers[-1].plot_optimal_classification(title='Class-Boundaries',
                     input_mesh=input_mesh, sketch_components=True, show=False)
            writer.add_figure('task_%d/class_boundaries' % i, plt.gcf(),
                              close=not show_plots)
            if show_plots:
                misc.repair_canvas_and_show_fig(plt.gcf())

    return dhandlers

def plot_gmm_preds(task_id, data, mnet, hnet, hhnet, device, config, shared,
                   logger, writer, tb_step, draw_samples=False,
                   normal_post=None):
    """Visualize the predictive entropy over the whole input space.

    The advantage of the GMM toy example is, that we can visualize quantities
    such as predictions and predictive entropies over an arbitrary large part of
    the 2D input space.

    Here, we use the current model associated with ``task_id``. All plots are
    logged to tensorboard.

    Args:
        (....): See docstring of function
            :func:`probabilistic.prob_cifar.train_avb.test`.
        task_id (int): ID of current task.
        tb_step (int): Tensorboard step for plots to be logged.
        draw_samples (bool): If ``True``, the method will also draw plots for
            single samples (if model is non-deterministic).
        normal_post (tuple, optional): See docstring of function
            :func:`probabilistic.regression.train_utils.compute_mse`
    """
    input_mesh = data.get_input_mesh(x1_range=config.gmm_grid_range_1,
        x2_range=config.gmm_grid_range_2, grid_size=config.gmm_grid_size)

    if 'bbb' in shared.experiment_type or 'ewc' in shared.experiment_type:
        disable_lrt = config.disable_lrt_test if \
            hasattr(config, 'disable_lrt_test') else False
        _, ret_fig = pmutils.compute_acc(task_id, data, mnet, hnet,
            device, config, shared, split_type=None, return_entropies=True,
            return_pred_labels=True, deterministic_sampling=True,
            disable_lrt=disable_lrt, in_samples=input_mesh[2],
            normal_post=normal_post)
    else:
        assert 'avb' in shared.experiment_type or 'ssge' in \
            shared.experiment_type
        _, ret_fig = pcutils.compute_acc(task_id, data, mnet, hnet, hhnet,
            device, config, shared, split_type=None, return_entropies=True,
            return_pred_labels=True, deterministic_sampling=True,
            in_samples=input_mesh[2])

    # The means of other tasks.
    other_means = np.concatenate([dh.means for dh in shared.all_dhandlers],
                                 axis=0)

    # Plot entropies over whole input space (according to `input_mesh`).
    data.plot_uncertainty_map( \
        title='Entropy of predictive distribution',
        input_mesh=input_mesh, uncertainties=ret_fig.entropies.reshape(-1, 1),
        sample_inputs=other_means,
        sketch_components=True, show=False)
    writer.add_figure('task_%d/pred_entropies' % task_id, plt.gcf(),
                      tb_step, close=not config.show_plots)
    if config.show_plots:
        misc.repair_canvas_and_show_fig(plt.gcf())

    # Plot entropies over whole input space (according to `input_mesh`).
    data.plot_optimal_classification(title='Predicted Class-Boundaries',
        input_mesh=input_mesh, mesh_modes=ret_fig.pred_labels.reshape(-1, 1),
        sample_inputs=other_means,
        sketch_components=True, show=False)
    writer.add_figure('task_%d/pred_class_boundaries' % task_id, plt.gcf(),
                      tb_step, close=not config.show_plots)
    if config.show_plots:
        misc.repair_canvas_and_show_fig(plt.gcf())

    # If not deterministic, plot single weight samples.
    # TODO We could also plot them for EWC
    if 'bbb' in shared.experiment_type and not config.mean_only or \
            ('avb' in shared.experiment_type or \
             'ssge' in shared.experiment_type) and hnet is not None and \
            draw_samples:
        for ii in range(10):
            if 'bbb' in shared.experiment_type:
                _, ret_fig = pmutils.compute_acc(task_id, data, mnet, hnet,
                    device, config, shared, split_type=None,
                    return_entropies=True, return_pred_labels=True,
                    deterministic_sampling=False,
                    disable_lrt=config.disable_lrt_test,
                    in_samples=input_mesh[2], num_w_samples=1)
            else:
                assert 'avb' in shared.experiment_type or \
                    'ssge' in shared.experiment_type
                _, ret_fig = pcutils.compute_acc(task_id, data, mnet, hnet,
                    hhnet, device, config, shared, split_type=None,
                    return_entropies=True, return_pred_labels=True,
                    deterministic_sampling=False, in_samples=input_mesh[2],
                    num_w_samples=1)

            # Plot entropies over whole input space (according to `input_mesh`).
            data.plot_uncertainty_map( \
                title='Entropy of predictive distribution',
                input_mesh=input_mesh,
                uncertainties=ret_fig.entropies.reshape(-1, 1),
                sample_inputs=other_means,
                sketch_components=True, show=False)
            writer.add_figure('task_%d/single_samples_pred_entropies' \
                % task_id, plt.gcf(), ii, close=not config.show_plots)
            if config.show_plots:
                misc.repair_canvas_and_show_fig(plt.gcf())

            # Plot entropies over whole input space (according to `input_mesh`).
            data.plot_optimal_classification(title='Predicted Class-Boundaries',
                input_mesh=input_mesh,
                mesh_modes=ret_fig.pred_labels.reshape(-1, 1),
                sample_inputs=other_means,
                sketch_components=True, show=False)
            writer.add_figure(\
                'task_%d/single_samples_pred_class_boundaries' % task_id,
                plt.gcf(), ii, close=not config.show_plots)
            if config.show_plots:
                misc.repair_canvas_and_show_fig(plt.gcf())

def plot_gmm_prior_preds(task_id, data, mnet, hnet, hhnet, device, config,
                         shared, logger, writer, prior_mean, prior_std,
                         prior_theta=None):
    """Visualize the prior predictive entropy over the whole input space.

    Similar to function :func:`plot_gmm_preds`, but rather than sampling from
    the approximate posterior, samples are drawn from a given prior
    distribution.

    Args:
        (....): See docstring of function :func:`plot_gmm_preds`.
        prior_mean (list): A list of tensors that represent the mean of an
            explicit prior. Is expected to be ``None`` if ``prior_theta`` is
            specified.
        prior_std (list): A list of tensors that represent the std of an
            explicit prior. See``prior_mean`` for more details.
        prior_theta (list): The weights passed to ``hnet`` when drawing samples
            from the current implicit distribution, which represents the prior.
    """
    # FIXME Code in this function is almost identical to the one in function
    # `plot_gmm_preds`.
    assert prior_mean is None and prior_std is None or \
           prior_mean is not None and prior_std is not None
    assert (prior_theta is None or prior_mean is None) and \
           (prior_theta is not None or prior_std is not None)

    # Gather prior samples.
    prior_samples = []
    for i in range(config.val_sample_size):
        if prior_theta is not None:
            z = torch.normal(torch.zeros(1, shared.noise_dim),
                             config.latent_std).to(device)
            prior_samples.append(hnet.forward(uncond_input=z,
                                              weights=prior_theta))
        else:
            prior_samples.append(putils.sample_diag_gauss(prior_mean,
                                                          prior_std))


    input_mesh = data.get_input_mesh(x1_range=config.gmm_grid_range_1,
        x2_range=config.gmm_grid_range_2, grid_size=config.gmm_grid_size)

    if 'bbb' in shared.experiment_type:
        _, ret_fig = pmutils.compute_acc(task_id, data, mnet, hnet,
            device, config, shared, split_type=None, return_entropies=True,
            return_pred_labels=True, deterministic_sampling=True,
            disable_lrt=config.disable_lrt_test, in_samples=input_mesh[2],
            w_samples=prior_samples)
    else:
        assert 'avb' in shared.experiment_type or 'ssge' in \
            shared.experiment_type
        _, ret_fig = pcutils.compute_acc(task_id, data, mnet, hnet, hhnet,
            device, config, shared, split_type=None, return_entropies=True,
            return_pred_labels=True, deterministic_sampling=True,
            in_samples=input_mesh[2], w_samples=prior_samples)

    # The means of other tasks.
    other_means = np.concatenate([dh.means for dh in shared.all_dhandlers],
                                 axis=0)

    # Plot entropies over whole input space (according to `input_mesh`).
    data.plot_uncertainty_map( \
        title='Entropy of prior predictive distribution',
        input_mesh=input_mesh, uncertainties=ret_fig.entropies.reshape(-1, 1),
        sample_inputs=other_means,
        sketch_components=True, show=False)
    writer.add_figure('prior/task_%d/pred_entropies' % task_id, plt.gcf(),
                      close=not config.show_plots)
    if config.show_plots:
        misc.repair_canvas_and_show_fig(plt.gcf())

    # Plot entropies over whole input space (according to `input_mesh`).
    data.plot_optimal_classification(title='Prior Predicted Class-Boundaries',
        input_mesh=input_mesh, mesh_modes=ret_fig.pred_labels.reshape(-1, 1),
        sample_inputs=other_means,
        sketch_components=True, show=False)
    writer.add_figure('prior/task_%d/pred_class_boundaries' % task_id,
                      plt.gcf(), close=not config.show_plots)
    if config.show_plots:
        misc.repair_canvas_and_show_fig(plt.gcf())

    # Plots for single samples from the prior.
    assert 'bbb' in shared.experiment_type and not config.mean_only or \
           'avb' in shared.experiment_type and hnet is not None or \
           'ssge' in shared.experiment_type and hnet is not None
    for ii in range(min(10, len(prior_samples))):
        if 'bbb' in shared.experiment_type:
            _, ret_fig = pmutils.compute_acc(task_id, data, mnet, hnet,
                device, config, shared, split_type=None,
                return_entropies=True, return_pred_labels=True,
                deterministic_sampling=False,
                disable_lrt=config.disable_lrt_test,
                in_samples=input_mesh[2], w_samples=[prior_samples[ii]])
        else:
            assert 'avb' in shared.experiment_type or 'ssge' in \
                shared.experiment_type
            _, ret_fig = pcutils.compute_acc(task_id, data, mnet, hnet,
                hhnet, device, config, shared, split_type=None,
                return_entropies=True, return_pred_labels=True,
                deterministic_sampling=False, in_samples=input_mesh[2],
                w_samples=[prior_samples[ii]])

        # Plot entropies over whole input space (according to `input_mesh`).
        data.plot_uncertainty_map( \
            title='Entropy of prior predictive distribution',
            input_mesh=input_mesh,
            uncertainties=ret_fig.entropies.reshape(-1, 1),
            sample_inputs=other_means,
            sketch_components=True, show=False)
        writer.add_figure('prior/task_%d/single_samples_pred_entropies' \
            % task_id, plt.gcf(), ii, close=not config.show_plots)
        if config.show_plots:
            misc.repair_canvas_and_show_fig(plt.gcf())

        # Plot entropies over whole input space (according to `input_mesh`).
        data.plot_optimal_classification( \
            title='Prior Predicted Class-Boundaries',
            input_mesh=input_mesh,
            mesh_modes=ret_fig.pred_labels.reshape(-1, 1),
            sample_inputs=other_means,
            sketch_components=True, show=False)
        writer.add_figure(\
            'prior/task_%d/single_samples_pred_class_boundaries' % task_id,
            plt.gcf(), ii, close=not config.show_plots)
        if config.show_plots:
            misc.repair_canvas_and_show_fig(plt.gcf())

if __name__ == '__main__':
    pass


