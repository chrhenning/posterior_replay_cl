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
# @title           :probabilistic/regression/train_avb.py
# @author          :rtr
# @contact         :henningc@ethz.ch
# @created         :08/25/2019
# @version         :0.1
# @python_version  :3.7
"""
Per-task implicit posterior via AVB
-----------------------------------

In this script, we train a target network via variational inference, where the
variational family is NOT restricted to a set of Gaussian distributions with
diagonal covariance matrix (as in
:mod:`probabilistic.regression.train_bbb`).
For the training we use an implicit method, the training method for this case
is described in

    Mescheder et al., "Adversarial Variational Bayes: Unifying Variational
    Autoencoders and Generative Adversarial Networks", 2018
    https://arxiv.org/abs/1701.04722

Specifically, we use a hypernetwork to output the weights for the target
network of each task in a continual learning setup, where tasks are presented
sequentially and forgetting of previous tasks is prevented by the
regularizer proposed in

    https://arxiv.org/abs/1906.00695
"""
# Do not delete the following import for all executable scripts!
import __init__  # pylint: disable=unused-import

from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
import torch
import torch.distributions

from probabilistic.prob_cifar import train_utils as pcu
from probabilistic.prob_mnist import train_utils as pmu
from probabilistic.regression import train_args
from probabilistic.regression import train_bbb
from probabilistic.regression import train_utils
from probabilistic import train_vi as tvi
import utils.misc as utils
from utils import sim_utils as sutils

def evaluate(task_id, data, mnet, hnet, hhnet, dis, device, config, shared,
             logger, writer, train_iter=None):
    """Evaluate the training progress.

    Evaluate the performance of the network on a single task (that is currently
    being trained) on the validation set.

    Note, if no validation set is available, the test set will be used instead.

    Args:
        (....): See docstring of method
            :func:`probabilistic.prob_cifar.train_avb.evaluate`.
    """
    # FIXME Code below almost identical to
    # `probabilistic.regression.train_bbb.evaluate`.
    if train_iter is None:
        logger.info('# Evaluating training ...')
    else:
        logger.info('# Evaluating network on task %d ' % (task_id+1) +
                    'before running training step %d ...' % train_iter)

    pcu.set_train_mode(False, mnet, hnet, hhnet, dis)

    with torch.no_grad():
        # Note, if no validation set exists, we use the training data to compute
        # the MSE (note, test data may contain out-of-distribution data in our
        # setup).
        split_type = 'train' if data.num_val_samples == 0 else 'val'
        if split_type == 'train':
            logger.debug('Eval - Using training set as no validation set is ' +
                         'available.')

        mse_val, val_struct = train_utils.compute_mse(task_id, data, mnet, hnet,
                device, config, shared, hhnet=hhnet, split_type=split_type)
        ident = 'training' if split_type == 'train' else 'validation'

        logger.info('Eval - Mean MSE on %s set: %f (std: %g).'
                    % (ident, mse_val, val_struct.mse_vals.std()))

        # In contrast, we visualize uncertainty using the test set.
        mse_test, test_struct = train_utils.compute_mse(task_id, data, mnet,
            hnet, device, config, shared, hhnet=hhnet, split_type='test',
            return_dataset=True, return_predictions=True)
        logger.debug('Eval - Mean MSE on test set: %f (std: %g).'
                     % (mse_test.mean(), mse_test.std()))

        if config.show_plots or train_iter is not None:
            train_utils.plot_predictive_distribution(data, test_struct.inputs,
                test_struct.predictions, show_raw_pred=True, figsize=(10, 4),
                show=train_iter is None)
            if train_iter is not None:
                writer.add_figure('task_%d/predictions' % task_id, plt.gcf(),
                                  train_iter, close=not config.show_plots)
                if config.show_plots:
                    utils.repair_canvas_and_show_fig(plt.gcf())

                writer.add_scalar('eval/task_%d/val_mse' % task_id,
                                  mse_val.mean(), train_iter)
                writer.add_scalar('eval/task_%d/test_mse' % task_id,
                                  mse_test.mean(), train_iter)

        # FIXME Code below copied from
        # `probabilistic.prob_cifar.train_avb.evaluate`.
        ### Compute discriminator accuracy.
        if dis is not None and hnet is not None:
            hnet_theta = None
            if hhnet is not None:
                hnet_theta = hhnet.forward(cond_id=task_id)

            # FIXME Is it ok if I only look at how samples from the current
            # implicit distribution are classified?
            dis_out, dis_inputs = pcu.process_dis_batch(config, shared,
                config.val_sample_size, device, dis, hnet, hnet_theta,
                dist=None)
            dis_acc = (dis_out > 0).sum().detach().cpu().numpy() / \
                config.val_sample_size * 100.

            logger.debug('Eval - Discriminator accuracy: %.2f%%.' % (dis_acc))
            writer.add_scalar('eval/task_%d/dis_acc' % task_id, dis_acc,
                              train_iter)

            # FIXME Summary results should be written in the test method after
            # training on a task has finished (note, eval is no guaranteed to be
            # called after or even during training). But I just want to get an
            # overview.
            s = shared.summary
            s['aa_acc_dis'][task_id] = dis_acc
            s['aa_acc_avg_dis'] = np.mean(s['aa_acc_dis'][:(task_id+1)])

            # Visualize weight samples.
            # FIXME A bit hacky.
            w_samples = dis_inputs
            if config.use_batchstats:
                w_samples = dis_inputs[:, (dis_inputs.shape[1]//2):]
            pcu.visualize_implicit_dist(config, task_id, writer, train_iter,
                                        w_samples, figsize=(10, 6))

        logger.info('# Evaluating training ... Done')

def run(method='avb'):
    """Run the script.

    Args:
        method (str, optional): The VI algorithm. Possible values are:

            - ``'avb'``
            - ``'ssge'``

    Returns:
        (tuple): Tuple containing:

        - **final_mse**: Final MSE for each task.
        - **during_mse**: MSE achieved directly after training on each task.
    """
    script_start = time()
    mode = 'regression_' + method
    use_dis = False # whether a discriminator network is used
    if method == 'avb':
        use_dis = True
    config = train_args.parse_cmd_arguments(mode=mode)

    device, writer, logger = sutils.setup_environment(config,
        logger_name=mode)

    train_utils.backup_cli_command(config)

    if config.prior_focused:
        logger.info('Running a prior-focused CL experiment ...')
    else:
        logger.info('Learning task-specific posteriors sequentially ...')

    ### Create tasks.
    dhandlers, num_tasks = train_utils.generate_tasks(config, writer)

    ### Simple struct, that is used to share data among functions.
    shared = Namespace()
    shared.experiment_type = mode
    shared.all_dhandlers = dhandlers
    shared.prior_focused = config.prior_focused

    ### Generate networks and environment
    mnet, hnet, hhnet, dnet = pcu.generate_networks(config, shared, logger,
                                                    dhandlers, device,
                                                    create_dis=use_dis)

    # Mean and variance of prior that is used for variational inference.
    # For a prior-focused training, this prior will only be used for the
    # first task.
    pstd = np.sqrt(config.prior_variance)
    shared.prior_mean = [torch.zeros(*s).to(device) \
                         for s in mnet.param_shapes]
    shared.prior_std = [pstd * torch.ones(*s).to(device) \
                        for s in mnet.param_shapes]

    # Note, all MSE values are measured on a validation set if given, otherwise
    # on the training set. All samples in the validation set are expected to
    # lay inside the training range. Test samples may lay outside the training
    # range.
    # The MSE value achieved right after training on the corresponding task.
    shared.during_mse = np.ones(num_tasks) * -1.
    # The weights of the main network right after training on that task
    # (can be used to assess how close the final weights are to the original
    # ones). Note, weights refer to mean and variances (e.g., the output of the
    # hypernetwork).
    shared.during_weights = [-1] * num_tasks
    # MSE achieved after most recent call of test method.
    shared.current_mse = np.ones(num_tasks) * -1.

    # Where to save network checkpoints?
    shared.ckpt_dir = os.path.join(config.out_dir, 'checkpoints')
    # Note, some networks have stuff to store such as batch statistics for
    # batch norm. So it is wise to always checkpoint all networks, even if they
    # where constructed without weights.
    shared.ckpt_mnet_fn = os.path.join(shared.ckpt_dir, 'mnet_task_%d')
    shared.ckpt_hnet_fn = os.path.join(shared.ckpt_dir, 'hnet_task_%d')
    shared.ckpt_hhnet_fn = os.path.join(shared.ckpt_dir, 'hhnet_task_%d')
    #shared.ckpt_dis_fn = os.path.join(shared.ckpt_dir, 'dis_task_%d')

    ### Initialize the performance measures, that should be tracked during
    ### training.
    train_utils.setup_summary_dict(config, shared, method, num_tasks, mnet,
                                   hnet=hnet, hhnet=hhnet, dis=dnet)
    logger.info('Ratio num hnet weights / num mnet weights: %f.'
                % shared.summary['aa_num_weights_hm_ratio'])
    if hhnet is not None:
        logger.info('Ratio num hyper-hnet weights / num mnet weights: %f.'
                    % shared.summary['aa_num_weights_hhm_ratio'])
    if mode == 'regression_avb' and dnet is not None:
        # A discriminator only exists for AVB.
        logger.info('Ratio num dis weights / num mnet weights: %f.'
                    % shared.summary['aa_num_weights_dm_ratio'])

    # Add hparams to tensorboard, such that the identification of runs is
    # easier.
    hparams_extra_dict = {
        'num_weights_hm_ratio': shared.summary['aa_num_weights_hm_ratio'],
        'num_weights_hhm_ratio': shared.summary['aa_num_weights_hhm_ratio']
    }
    if mode == 'regression_avb':
        hparams_extra_dict['num_weights_dm_ratio'] = \
            shared.summary['aa_num_weights_dm_ratio']
    writer.add_hparams(hparam_dict={**vars(config), **hparams_extra_dict},
                       metric_dict={})

    ### Train on tasks sequentially.
    for i in range(num_tasks):
        logger.info('### Training on task %d ###' % (i + 1))
        data = dhandlers[i]

        # Train the network.
        tvi.train(i, data, mnet, hnet, hhnet, dnet, device, config, shared,
                    logger, writer, method=method)

        # Test networks.
        train_bbb.test(dhandlers[:(i + 1)], mnet, hnet, device, config, shared,
                       logger, writer, hhnet=hhnet)

        if config.train_from_scratch and i < num_tasks - 1:
            # We have to checkpoint the networks, such that we can reload them
            # for task inference later during testing.
            # Note, we only need the discriminator as helper for training,
            # so we don't checkpoint it.
            pmu.checkpoint_nets(config, shared, i, mnet, hnet, hhnet=hhnet,
                                    dis=None)

            mnet, hnet, hhnet, dnet = pcu.generate_networks(config, shared,
                logger, dhandlers, device)

        elif config.store_during_models:
            logger.info('Checkpointing current model ...')
            pmu.checkpoint_nets(config, shared, i, mnet, hnet, hhnet=hhnet,
                                dis=None)

    if config.store_final_model:
        logger.info('Checkpointing final model ...')
        pmu.checkpoint_nets(config, shared, num_tasks-1, mnet, hnet,
                            hhnet=hhnet, dis=None)

    logger.info('During MSE values after training each task: %s' % \
                np.array2string(shared.during_mse, precision=5, separator=','))
    logger.info('Final MSE values after training on all tasks: %s' % \
                np.array2string(shared.current_mse, precision=5, separator=','))
    logger.info('Final MSE mean %.4f (std %.4f).' % (shared.current_mse.mean(),
                                                     shared.current_mse.std()))

    ### Write final summary.
    shared.summary['finished'] = 1
    train_utils.save_summary_dict(config, shared)

    writer.close()

    logger.info('Program finished successfully in %f sec.'
                % (time() - script_start))

    return shared.current_mse, shared.during_mse


if __name__ == '__main__':
    _, _ = run()
