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
# @title           :probabilistic/regression/train_bbb.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :08/25/2019
# @version         :1.0
# @python_version  :3.6.8
"""
Per-task posterior via Bayes-by-Backprop
----------------------------------------

In the script :mod:`probabilistic.regression.train_bbb`, we obtain an
approximate weight posterior of the target network via variational inference,
where the variational family is specified through the set of Gaussian
dristributions with diagonal covariance matrix. The training method for this
case is described in

    Blundell et al., "Weight Uncertainty in Neural Networks", 2015.
    https://arxiv.org/abs/1505.05424

Specifically, we use a hypernetwork to output the mean and variance (where the
variance is usually encoded in a real number) of each task in a continual
learning setup, where tasks are presented sequentially and forgetting of
previous tasks is prevented by the regularizer proposed in

    von Oswald et al., "Continual learning with hypernetworks", ICLR, 2020.
    https://arxiv.org/abs/1906.00695
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
import torch
import torch.nn.functional as F
from warnings import warn

from probabilistic import ewc_utils as ewcutil
from probabilistic.gauss_mnet_interface import GaussianBNNWrapper
from probabilistic import prob_utils as putils
from probabilistic.prob_cifar import train_utils as pcutils
from probabilistic.prob_mnist import train_utils as pmutils
from probabilistic.regression import train_args
from probabilistic.regression import train_utils
import utils.hnet_regularizer as hreg
import utils.misc as utils
import utils.sim_utils as sutils
import utils.torch_utils as tutils

def test(data_handlers, mnet, hnet, device, config, shared, logger, writer,
         hhnet=None, save_fig=True):
    """Test the performance of all tasks.

    Tasks are assumed to be regression tasks.

    Args:
        (....): See docstring of method
            :func:`probabilistic.train_vi.train`.
        data_handlers: A list of data handlers, each representing a task.
        save_fig: Whether the figures should be saved in the output folder.
    """
    logger.info('### Testing all trained tasks ... ###')

    if hasattr(config, 'mean_only') and config.mean_only:
         warn('Task inference calculated in test method doesn\'t make any ' +
             'sense, as the deterministic main network has no notion of ' +
             'uncertainty.')

    pcutils.set_train_mode(False, mnet, hnet, hhnet, None)

    n = len(data_handlers)

    disable_lrt_test = config.disable_lrt_test if \
        hasattr(config, 'disable_lrt_test') else None

    if not hasattr(shared, 'current_mse'):
        shared.current_mse = np.ones(n) * -1.
    elif shared.current_mse.size < n:
        tmp = shared.current_mse
        shared.current_mse = np.ones(n) * -1.
        shared.current_mse[:tmp.size] = tmp

    # Current MSE value on test set.
    test_mse = np.ones(n) * -1.
    # Current MSE value using the mean prediction of the inferred embedding.
    inferred_val_mse = np.ones(n) * -1.
    # Task inference accuracies.
    task_infer_val_accs = np.ones(n) * -1.

    with torch.no_grad():
        # We need to keep data for plotting results on all tasks later on.
        val_inputs = []
        val_targets = [] # Needed to compute MSE values.
        val_preds_mean = []
        val_preds_std = []

        # Which uncertainties have been measured per sample and task. The argmax
        # over all tasks gives the predicted task.
        val_task_preds = []

        test_inputs = []
        test_preds_mean = []
        test_preds_std = []

        normal_post = None
        if 'ewc' in shared.experiment_type:
            assert hnet is None
            normal_post = ewcutil.build_ewc_posterior(data_handlers, mnet,
                device, config, shared, logger, writer, n, task_id=n-1)

        if config.train_from_scratch and n > 1:
            # We need to iterate over different networks when we want to
            # measure the uncertainty of dataset i on task j.
            # Note, we will always load the corresponding checkpoint of task j
            # before using these networks.
            if 'avb' in shared.experiment_type \
                    or 'ssge' in shared.experiment_type\
                    or 'ewc' in shared.experiment_type:
                mnet_other, hnet_other, hhnet_other, _ = \
                    pcutils.generate_networks(config, shared, logger,
                        shared.all_dhandlers, device, create_dis=False)
            else:
                assert hhnet is None
                hhnet_other = None
                non_gaussian = config.mean_only \
                    if hasattr(config, 'mean_only') else True
                mnet_other, hnet_other = train_utils.generate_gauss_networks( \
                    config, logger, shared.all_dhandlers, device,
                    create_hnet=hnet is not None, non_gaussian=non_gaussian)

            pcutils.set_train_mode(False, mnet_other, hnet_other, hhnet_other,
                                   None)

        task_n_mnet = mnet
        task_n_hnet = hnet
        task_n_hhnet = hhnet
        task_n_normal_post = normal_post

        # This renaming is just a protection against myself, that I don't use
        # any of those networks (`mnet`, `hnet`, `hhnet`) in the future
        # inside the loop when training from scratch.
        if config.train_from_scratch:
            mnet = None
            hnet = None
            hhnet = None
            normal_post = None

        ### For each data set (i.e., for each task).
        for i in range(n):
            data = data_handlers[i]

            ### We want to measure MSE values within the training range only!
            split_type = 'val'
            num_val_samples = data.num_val_samples
            if num_val_samples == 0:
                split_type = 'train'
                num_val_samples = data.num_train_samples

                logger.debug('Test: Task %d - Using training set as no ' % i +
                             'validation set is available.')

            ### Task inference.
            # We need to iterate over each task embedding and measure the
            # predictive uncertainty in order to decide which embedding to use.
            data_preds = np.empty((num_val_samples, config.val_sample_size, n))
            data_preds_mean = np.empty((num_val_samples, n))
            data_preds_std = np.empty((num_val_samples, n))

            for j in range(n):
                ckpt_score_j = None
                if config.train_from_scratch and j == (n-1):
                    # Note, the networks trained on dataset (n-1) haven't been
                    # checkpointed yet.
                    mnet_j = task_n_mnet
                    hnet_j = task_n_hnet
                    hhnet_j = task_n_hhnet
                    normal_post_j = task_n_normal_post
                elif config.train_from_scratch:
                    ckpt_score_j = pmutils.load_networks(shared, j, device,
                        logger, mnet_other, hnet_other, hhnet=hhnet_other,
                        dis=None)
                    mnet_j = mnet_other
                    hnet_j = hnet_other
                    hhnet_j = hhnet_other
                    normal_post_j = None
                    if 'ewc' in shared.experiment_type:
                        normal_post_j = ewcutil.build_ewc_posterior( \
                            data_handlers, mnet_j, device, config, shared,
                            logger, writer, n, task_id=j)
                else:
                    mnet_j = mnet
                    hnet_j = hnet
                    hhnet_j = hhnet
                    normal_post_j = normal_post

                mse_val, val_struct = train_utils.compute_mse(j, data, mnet_j,
                    hnet_j, device, config, shared, hhnet=hhnet_j,
                    split_type=split_type, return_dataset=i==j,
                    return_predictions=True, disable_lrt=disable_lrt_test,
                    normal_post=normal_post_j)

                if i == j: # I.e., we used the correct embedding.
                    # This sanity check is likely to fail as we don't
                    # deterministically sample the models.
                    #if ckpt_score_j is not None:
                    #    assert np.allclose(-mse_val, ckpt_score_j)

                    val_inputs.append(val_struct.inputs)
                    val_targets.append(val_struct.targets)
                    val_preds_mean.append(val_struct.predictions.mean(axis=1))
                    val_preds_std.append(val_struct.predictions.std(axis=1))

                    shared.current_mse[i] = mse_val

                    logger.debug('Test: Task %d - Mean MSE on %s set: %f '
                                 % (i, split_type, mse_val)
                                 + '(std: %g).' % (val_struct.mse_vals.std()))
                    writer.add_scalar('test/task_%d/val_mse' % i,
                                      shared.current_mse[i], n)

                    # The test set spans into the OOD range and can be used to
                    # visualize how uncertainty behaves outside the
                    # in-distribution range.
                    mse_test, test_struct = train_utils.compute_mse(i, data,
                        mnet_j, hnet_j, device, config, shared, hhnet=hhnet_j,
                        split_type='test', return_dataset=True,
                        return_predictions=True, disable_lrt=disable_lrt_test,
                        normal_post=normal_post_j)

                data_preds[:, :, j] = val_struct.predictions
                data_preds_mean[:, j] = val_struct.predictions.mean(axis=1)
                ### We interpret this value as the certainty of the prediction.
                # I.e., how certain is our system that each of the samples
                # belong to task j?
                data_preds_std[:, j] = val_struct.predictions.std(axis=1)

            val_task_preds.append(data_preds_std)

            ### Compute task inference accuracy.
            inferred_task_ids = data_preds_std.argmin(axis=1)
            num_correct = np.sum(inferred_task_ids == i)
            accuracy = 100. * num_correct / num_val_samples
            task_infer_val_accs[i] = accuracy

            logger.debug('Test: Task %d - Accuracy of task inference ' % i +
                         'on %s set: %.2f%%.'
                         % (split_type, accuracy))
            writer.add_scalar('test/task_%d/accuracy' % i, accuracy, n)

            ### Compute MSE based on inferred embedding.

            # Note, this (commented) way of computing the mean does not take
            # into account the variance of the predictive distribution, which is
            # why we don't use it (see docstring of `compute_mse`).
            #means_of_inferred_preds = data_preds_mean[np.arange( \
            #    data_preds_mean.shape[0]), inferred_task_ids]
            #inferred_val_mse[i] = np.power(means_of_inferred_preds -
            #                               val_targets[-1].squeeze(), 2).mean()

            inferred_preds = data_preds[np.arange(data_preds.shape[0]), :,
                                        inferred_task_ids]
            inferred_val_mse[i] = np.power(inferred_preds - \
                val_targets[-1].squeeze()[:, np.newaxis], 2).mean()

            logger.debug('Test: Task %d - Mean MSE on %s set using inferred '\
                         % (i, split_type) + 'embeddings: %f.'
                         % (inferred_val_mse[i]))
            writer.add_scalar('test/task_%d/inferred_val_mse' % i,
                              inferred_val_mse[i], n)

            ### We are interested in the predictive uncertainty across the
            ### whole test range!
            test_mse[i] = mse_test
            writer.add_scalar('test/task_%d/test_mse' % i, test_mse[i], n)

            test_inputs.append(test_struct.inputs.squeeze())
            test_preds_mean.append(test_struct.predictions.mean(axis=1). \
                                   squeeze())
            test_preds_std.append(test_struct.predictions.std(axis=1).squeeze())

            if hasattr(shared, 'during_mse') and \
                    shared.during_mse[i] == -1:
                shared.during_mse[i] = shared.current_mse[i]

            if test_struct.w_hnet is not None or test_struct.w_mean is not None:
                assert hasattr(shared, 'during_weights')
                if test_struct.w_hnet is not None:
                    # We have a hyper-hypernetwork. In this case, the CL
                    # regularizer is applied to its output and therefore, these
                    # are the during weights whose Euclidean distance we want to
                    # track.
                    assert task_n_hhnet is not None
                    w_all = test_struct.w_hnet
                else:
                    assert test_struct.w_mean is not None
                    # We will be here whenever the hnet is deterministic (i.e.,
                    # doesn't represent an implicit distribution).
                    w_all = list(test_struct.w_mean)
                    if test_struct.w_std is not None:
                        w_all += list(test_struct.w_std)

                W_curr = torch.cat([d.clone().view(-1) for d in w_all])
                if type(shared.during_weights[i]) == int:
                    assert(shared.during_weights[i] == -1)
                    shared.during_weights[i] = W_curr
                else:
                    W_during = shared.during_weights[i]
                    W_dis = torch.norm(W_curr - W_during, 2)
                    logger.info('Euclidean distance between hypernet output ' +
                                'for task %d: %g' % (i, W_dis))

    ### Compute overall task inference accuracy.
    num_correct = 0
    num_samples = 0
    for i, uncertainties in enumerate(val_task_preds):
        pred_task_ids = uncertainties.argmin(axis=1)
        num_correct += np.sum(pred_task_ids == i)
        num_samples += pred_task_ids.size

    accuracy = 100. * num_correct / num_samples
    logger.info('Task inference accuracy: %.2f%%.' % accuracy)

    # TODO Compute overall MSE on all tasks using inferred embeddings.

    ### Plot the mean predictions on all tasks.
    # (Using the validation set and the correct embedding per dataset)
    plot_x_ranges = []
    for i in range(n):
        plot_x_ranges.append(data_handlers[i].train_x_range)

    fig_fn = None
    if save_fig:
        fig_fn = os.path.join(config.out_dir, 'val_predictions_%d' % n)

    data_inputs = val_inputs
    mean_preds = val_preds_mean
    data_handlers[0].plot_datasets(data_handlers, data_inputs,
        mean_preds, fun_xranges=plot_x_ranges, filename=fig_fn,
        show=False, publication_style=config.publication_style)
    writer.add_figure('test/val_predictions', plt.gcf(), n,
                      close=not config.show_plots)
    if config.show_plots:
        utils.repair_canvas_and_show_fig(plt.gcf())

    ### Scatter plot showing MSE per task (original + current one).
    during_mse = None
    if hasattr(shared, 'during_mse'):
        during_mse = shared.during_mse[:n]
    train_utils.plot_mse(config, writer, n, shared.current_mse[:n],
                         during_mse, save_fig=save_fig)
    additional_plots = {
        'Current Inferred Val MSE': inferred_val_mse,
        #'Current Test MSE': test_mse
    }
    train_utils.plot_mse(config, writer, n, shared.current_mse[:n],
        during_mse, baselines=additional_plots, save_fig=False,
        summary_label='test/mse_detailed')

    ### Plot predictive distributions over test range for all tasks.
    data_inputs = test_inputs
    mean_preds = test_preds_mean
    std_preds = test_preds_std

    train_utils.plot_predictive_distributions(config, writer, data_handlers,
        data_inputs, mean_preds, std_preds, save_fig=save_fig,
        publication_style=config.publication_style)

    logger.info('Mean task MSE: %f (std: %d)' % (shared.current_mse[:n].mean(),
                                                 shared.current_mse[:n].std()))

    ### Update performance summary.
    s = shared.summary
    s['aa_mse_during'][:n] = shared.during_mse[:n].tolist()
    s['aa_mse_during_mean'] = shared.during_mse[:n].mean()
    s['aa_mse_final'][:n] = shared.current_mse[:n].tolist()
    s['aa_mse_final_mean'] = shared.current_mse[:n].mean()

    s['aa_task_inference'][:n] = task_infer_val_accs.tolist()
    s['aa_task_inference_mean'] = task_infer_val_accs.mean()

    s['aa_mse_during_inferred'][n-1] = inferred_val_mse[n-1]
    s['aa_mse_during_inferred_mean'] = np.mean(s['aa_mse_during_inferred'][:n])
    s['aa_mse_final_inferred'] = inferred_val_mse[:n].tolist()
    s['aa_mse_final_inferred_mean'] = inferred_val_mse[:n].mean()

    train_utils.save_summary_dict(config, shared)

    logger.info('### Testing all trained tasks ... Done ###')

def evaluate(task_id, data, mnet, hnet, device, config, shared, logger, writer,
             train_iter=None):
    """Evaluate the training progress.

    Evaluate the performance of the network on a single task (that is currently
    being trained) on the validation set.

    Note, if no validation set is available, the test set will be used instead.

    Args:
        (....): See docstring of method :func:`train`. Note, `hnet` can be
            passed as :code:`None`. In this case, no weights are passed to the
            `forward` method of the main network.
        train_iter: The current training iteration. If not given, the `writer`
            will not be used.
    """
    if train_iter is None:
        logger.info('# Evaluating training ...')
    else:
        logger.info('# Evaluating network on task %d ' % (task_id+1) +
                    'before running training step %d ...' % (train_iter))

    # TODO: write histograms of weight samples to tensorboard.

    mnet.eval()
    if hnet is not None:
        hnet.eval()

    with torch.no_grad():
        # Note, if no validation set exists, we use the training data to compute
        # the MSE (note, test data may contain out-of-distribution data in our
        # setup).
        split_type = 'train' if data.num_val_samples == 0 else 'val'
        if split_type == 'train':
            logger.debug('Eval - Using training set as no validation set is ' +
                         'available.')

        mse_val, val_struct = train_utils.compute_mse(task_id, data, mnet,
            hnet, device, config, shared, split_type=split_type)
        ident = 'training' if split_type == 'train' else 'validation'

        logger.info('Eval - Mean MSE on %s set: %f (std: %g).'
                    % (ident, mse_val, val_struct.mse_vals.std()))

        # In contrast, we visualize uncertainty using the test set.
        mse_test, test_struct = train_utils.compute_mse(task_id, data, mnet,
            hnet, device, config, shared, split_type='test', return_dataset=True,
            return_predictions=True)
        logger.debug('Eval - Mean MSE on test set: %f (std: %g).'
                     % (mse_test, test_struct.mse_vals.std()))

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
                                  mse_val, train_iter)
                writer.add_scalar('eval/task_%d/test_mse' % task_id,
                                  mse_test, train_iter)

        logger.info('# Evaluating training ... Done')

def train(task_id, data, mnet, hnet, device, config, shared, logger, writer):
    r"""Train the network using the task-specific loss plus a regularizer that
    should weaken catastrophic forgetting.

    .. math::

        \text{loss} = \text{task\_loss} + \beta * \text{regularizer}

    The task specific loss aims to learn the mean and variances of the main net
    weights such that the posterior parameter distribution is approximated.

    Args:
        task_id: The index of the task on which we train.
        data: The dataset handler.
        mnet: The model of the main network.
        hnet: The model of the hyoer network. May be ``None``.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        shared: Miscellaneous data shared among training functions.
        logger: Command-line logger.
        writer: The tensorboard summary writer.
    """
    assert isinstance(mnet, GaussianBNNWrapper) or config.mean_only

    logger.info('Training network on task %d ...' % (task_id+1))

    mnet.train()
    if hnet is not None:
        hnet.train()

    # Not all output units have to be regularized for every task in a multi-
    # head setup.
    regged_outputs = None
    if config.multi_head:
        # FIXME We currently only mask the variances correctly, but the means
        # are not masked at all. See function "flatten_and_remove_out_heads".
        warn('Note, method "calc_fix_target_reg" doesn\'t know that our hnet ' +
             'outputs means and variances, so it can\'t correctly mask ' +
             'unused output heads.')
        n_y = data.out_shape[0]
        out_head_inds = [list(range(i*n_y, (i+1)*n_y)) for i in
                         range(task_id+1)]
        # Outputs to be regularized.
        regged_outputs = out_head_inds[:-1]
    allowed_outputs = out_head_inds[task_id] if config.multi_head else None

    # Whether the regularizer will be computed during training?
    calc_reg = hnet is not None  and task_id > 0 and config.beta > 0 and \
         not config.train_from_scratch

    # Regularizer targets.
    # Store distributions for each task before training on the current task.
    if calc_reg:
        targets, w_mean_pre, w_logvar_pre = pmutils.calc_reg_target(config,
            task_id, hnet, mnet=mnet)

    ### Define Prior
    # Whether prior-matching should even be performed?

    # What prior to use for BbB training?
    standard_prior = False
    if config.use_prev_post_as_prior and task_id > 0:
        assert isinstance(mnet, GaussianBNNWrapper)

        if config.train_from_scratch:
            raise NotImplementedError()
        if config.radial_bnn:
            # TODO Prior is not a Gaussian anymore.
            raise NotImplementedError()

        logger.debug('Choosing posterior of previous task as prior.')
        if hnet is None:
            hnet_out = None
        else:
            hnet_out = hnet.forward(cond_id=task_id-1)
        w_mean_prev, w_rho_prev = mnet.extract_mean_and_rho(weights=hnet_out)
        w_std_prev, w_logvar_prev = putils.decode_diag_gauss(w_rho_prev, \
            logvar_enc=mnet.logvar_encoding, return_logvar=True)

        prior_mean = [p.detach().clone() for p in w_mean_prev]
        prior_logvar = [p.detach().clone() for p in w_logvar_prev]
        prior_std = [p.detach().clone() for p in w_std_prev]

        # Note task-specific head weights of this task and future tasks should
        # be pulled to the prior, as they haven't been learned yet.
        # Note, for radial BNNs that would be difficult, as a mixture of radial
        # and Gaussian prior would need to be applied.
        # Note, in principle this step is not necessary, as those task-specific
        # weights have been only pulled to the prior when learning the prior
        # tasks.
        if config.multi_head: # FIXME A bit hacky :D
            # Output head weight masks for all previous tasks
            out_masks = [mnet._mnet.get_output_weight_mask( \
                         out_inds=regged_outputs[i], device=device) \
                         for i in range(task_id)]
            for ii, mask in enumerate(out_masks[0]):
                if mask is None: # Shared parameter.
                    continue
                else: # Output weight tensor.
                    tmp_mean = prior_mean[ii]
                    tmp_logvar = prior_logvar[ii]
                    tmp_std = prior_std[ii]

                    prior_mean[ii] = shared.prior_mean[ii].clone()
                    prior_logvar[ii] = shared.prior_logvar[ii].clone()
                    prior_std[ii] = shared.prior_std[ii].clone()

                    for jj, t_mask in enumerate(out_masks):
                        m = t_mask[ii]
                        prior_mean[ii][m] = tmp_mean[m]
                        prior_logvar[ii][m] = tmp_logvar[m]
                        prior_std[ii][m] = tmp_std[m]
    else:
        prior_mean = shared.prior_mean
        prior_logvar = shared.prior_logvar
        prior_std = shared.prior_std
        if config.prior_variance == 1:
            # Use standard Gaussian prior with 0 mean and unit variance.
            standard_prior = True

    if hnet is None:
        params = mnet.parameters()
    else:
        params = hnet.parameters()
    optimizer = tutils.get_optimizer(params, config.lr,
        momentum=None, weight_decay=config.weight_decay,
        use_adam=True, adam_beta1=config.adam_beta1)

    assert config.ll_dist_std > 0
    ll_scale = 1. / config.ll_dist_std**2

    for i in range(config.n_iter):
        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            evaluate(task_id, data, mnet, hnet, device, config, shared, logger,
                     writer, i)
            mnet.train()
            if hnet is not None:
                hnet.train()

        if i % 100 == 0:
            logger.debug('Training iteration: %d.' % i)

        ### Train theta and task embedding.
        optimizer.zero_grad()

        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        T = data.output_to_torch_tensor(batch[1], device, mode='train')

        if hnet is None:
            hnet_out = None
        else:
            hnet_out = hnet.forward(cond_id=task_id)
        if config.mean_only:
            if hnet_out is None:
                w_mean = mnet.weights
            else:
                w_mean = hnet_out
            w_std = None
        else:
            w_mean, w_rho = mnet.extract_mean_and_rho(weights=hnet_out)
            w_std, w_logvar = putils.decode_diag_gauss(w_rho, \
                logvar_enc=mnet.logvar_encoding, return_logvar=True)

        ### Prior-matching loss.
        if config.mean_only:
            loss_kl = 0
        elif not config.radial_bnn:
            if standard_prior:
                # Gaussian prior with zero mean and unit variance.
                loss_kl = putils.kl_diag_gauss_with_standard_gauss(w_mean,
                    w_logvar)
            else:
                loss_kl = putils.kl_diag_gaussians(w_mean, w_logvar,
                    prior_mean, prior_logvar)
        else:
            # When using radial BNNs the weight distribution is not gaussian.
            loss_kl = putils.kl_radial_bnn_with_diag_gauss(w_mean, w_std,
                prior_mean, prior_std, ce_sample_size=config.num_kl_samples)

        ### Compute negative log-likelihood (NLL).
        loss_nll = 0
        for j in range(config.train_sample_size):
            if config.mean_only:
                Y = mnet.forward(X, weights=w_mean)
            else:
                # Note, the sampling will happen inside the forward method.
                Y = mnet.forward(X, weights=None, mean_only=False,
                                 extracted_mean=w_mean, extracted_rho=w_rho)
            if config.multi_head:
                Y = Y[:, allowed_outputs]

            # Task-specific loss.
            # We use the reduction method 'mean' on purpose and scale with
            # the number of training samples below.
            loss_nll += F.mse_loss(Y, T, reduction='mean')

        loss_nll *= 0.5 * ll_scale * \
            data.num_train_samples / config.train_sample_size

        ### Compute CL regularizer.
        loss_reg = 0
        if calc_reg:
            if config.regularizer == 'mse':
                # Compute the regularizer as given in von Oswald et al. 2019
                loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
                    targets=targets, mnet=mnet,
                    inds_of_out_heads=regged_outputs)
            else:
                # Compute the regularizer based on a distance metric between
                # the posterior distributions of all previous tasks before and
                # while learning the current task.
                for t in range(task_id):
                    hnet_out = hnet.forward(cond_id=t)

                    w_mean_t, w_rho_t = mnet.extract_mean_and_rho( \
                        weights=hnet_out)
                    _, w_logvar_t = putils.decode_diag_gauss(w_rho_t, \
                        logvar_enc=mnet.logvar_encoding, return_logvar=True)

                    if config.regularizer == 'fkl':
                        # Use the forward KL divergence
                        loss_reg += putils.kl_diag_gaussians(w_mean_pre[t],
                                w_logvar_pre[t], w_mean_t, w_logvar_t)
                    elif config.regularizer == 'rkl':
                        # Use the reverse KL divergence
                        loss_reg += putils.kl_diag_gaussians(w_mean_t,
                                w_logvar_t, w_mean_pre[t], w_logvar_pre[t])
                    elif config.regularizer == 'w2':
                        # Use the Wasserstein-2 metric
                        loss_reg += putils.square_wasserstein_2(w_mean_pre[t],
                                w_logvar_pre[t], w_mean_t, w_logvar_t)

                loss_reg /= task_id

        loss = loss_kl + loss_nll + config.beta * loss_reg

        loss.backward()
        if config.clip_grad_value != -1:
            torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'],
                                            config.clip_grad_value)
        elif config.clip_grad_norm != -1:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],
                                            config.clip_grad_norm)
        optimizer.step()

        if i % 50 == 0:
            writer.add_scalar('train/task_%d/loss_kl' % task_id, loss_kl, i)
            writer.add_scalar('train/task_%d/loss_nll' % task_id, loss_nll, i)
            writer.add_scalar('train/task_%d/regularizer' % task_id, loss_reg,
                              i)
            writer.add_scalar('train/task_%d/loss' % task_id, loss, i)

            # Plot distribution of mean and log-variance values.
            mean_outputs = torch.cat([d.clone().view(-1) for d in w_mean])
            writer.add_histogram('train/task_%d/predicted_means' % task_id,
                                 mean_outputs, i)
            if w_std is not None:
                rho_outputs = torch.cat([d.clone().view(-1) for d in w_rho])
                std_outputs = torch.cat([d.clone().view(-1) for d in w_std])
                writer.add_histogram('train/task_%d/predicted_rhos' % task_id,
                                     rho_outputs, i)
                writer.add_histogram('train/task_%d/predicted_stds' % task_id,
                                     std_outputs, i)

    logger.info('Training network on task %d ... Done' % (task_id+1))

def run():
    """Run the script.

    Returns:
        (tuple): Tuple containing:

        - **final_mse**: Final MSE for each task.
        - **during_mse**: MSE achieved directly after training on each task.
    """
    script_start = time()

    mode = 'regression_bbb'
    config = train_args.parse_cmd_arguments(mode=mode)

    device, writer, logger = sutils.setup_environment(config,
        logger_name=mode)

    train_utils.backup_cli_command(config)

    ### Create tasks.
    dhandlers, num_tasks = train_utils.generate_tasks(config, writer)

    ### Generate networks.
    use_hnet = not config.mnet_only
    mnet, hnet = train_utils.generate_gauss_networks(config, logger, dhandlers,
        device, create_hnet=use_hnet, non_gaussian=config.mean_only)

    ### Simple struct, that is used to share data among functions.
    shared = Namespace()
    shared.experiment_type = mode
    shared.all_dhandlers = dhandlers
    # Mean and variance of prior that is used for variational inference.
    if config.mean_only: # No prior-matching can be performed.
        shared.prior_mean = None
        shared.prior_logvar = None
        shared.prior_std = None
    else:
        plogvar = np.log(config.prior_variance)
        pstd = np.sqrt(config.prior_variance)
        shared.prior_mean = [torch.zeros(*s).to(device) \
                             for s in mnet.orig_param_shapes]
        shared.prior_logvar = [plogvar * torch.ones(*s).to(device) \
                               for s in mnet.orig_param_shapes]
        shared.prior_std = [pstd * torch.ones(*s).to(device) \
                            for s in mnet.orig_param_shapes]

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
    # Note, some main networks have stuff to store such as batch statistics for
    # batch norm. So it is wise to always checkpoint mnets as well!
    shared.ckpt_mnet_fn = os.path.join(shared.ckpt_dir, 'mnet_task_%d')
    shared.ckpt_hnet_fn = os.path.join(shared.ckpt_dir, 'hnet_task_%d')

    ### Initialize the performance measures, that should be tracked during
    ### training.
    train_utils.setup_summary_dict(config, shared, 'bbb', num_tasks, mnet,
                                   hnet=hnet)

    # Add hparams to tensorboard, such that the identification of runs is
    # easier.
    writer.add_hparams(hparam_dict={**vars(config), **{
        'num_weights_main': shared.summary['aa_num_weights_main'],
        'num_weights_hyper': shared.summary['aa_num_weights_hyper'],
        'num_weights_ratio': shared.summary['aa_num_weights_ratio'],
    }}, metric_dict={})

    ### Train on tasks sequentially.
    for i in range(num_tasks):
        logger.info('### Training on task %d ###' % (i+1))
        data = dhandlers[i]
        # Train the network.
        train(i, data, mnet, hnet, device, config, shared, logger, writer)

        ### Test networks.
        test(dhandlers[:(i+1)], mnet, hnet, device, config, shared, logger,
             writer)

        if config.train_from_scratch and i < num_tasks-1:
            # We have to checkpoint the networks, such that we can reload them
            # for task inference later during testing.
            pmutils.checkpoint_nets(config, shared, i, mnet, hnet)

            mnet, hnet = train_utils.generate_gauss_networks(config, logger,
                dhandlers, device, create_hnet=use_hnet,
                non_gaussian=config.mean_only)

    if config.store_final_model:
        logger.info('Checkpointing final model ...')
        pmutils.checkpoint_nets(config, shared, num_tasks-1, mnet, hnet)

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
                % (time()-script_start))

    return shared.current_mse, shared.during_mse

if __name__ == '__main__':
    _, _ = run()


