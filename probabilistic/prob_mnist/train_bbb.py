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
# title          :probabilistic/prob_mnist/train_bbb.py
# author         :ch
# contact        :henningc@ethz.ch
# created        :09/06/2019
# version        :1.0
# python_version :3.6.8
"""
Per-task posterior via Bayes-by-Backprop
----------------------------------------

This module is meant for training Bayesian Neural Networks on a sequence of
classification problems (i.e., in a continual learning framework).
We adopt the algorithm named Bayes-by-Backprop (BbB):

    Blundell et al., "Weight Uncertainty in Neural Networks", 2015
    https://arxiv.org/abs/1505.05424

Using BbB, we will learn a weight posterior per task. To manage all these
posteriors we entangle them all into a single hypernetwork, such that each of
them can be retrieved using a task-embedding as input to the hypernetwork.

In case the tasks have distinct input domains, the task identity (and thus the
right task embedding) can be selected based on the entropy of the predictive
distribution. Hence, we choose the task embedding where the system is most
certain.
"""
from argparse import Namespace
import numpy as np
import os
from time import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from warnings import warn

from mnets.classifier_interface import Classifier
from hnets.hnet_helpers import get_conditional_parameters
from probabilistic import prob_utils as putils
from probabilistic import GaussianBNNWrapper
from probabilistic import train_vi
from probabilistic.prob_cifar import train_utils as pcutils
from probabilistic.prob_gmm import train_utils as pgutils
from probabilistic.prob_mnist import train_utils
from probabilistic.regression import train_utils as rutils
from utils import hnet_regularizer as hreg
from utils import torch_utils as tutils
from utils import sim_utils as sutils

def evaluate(task_id, data, mnet, hnet, device, config, shared, logger, writer,
             train_iter):
    """Evaluate the network. Evaluate the performance of the network on a
    single task on the validation set.

    Note, if no validation set is available, the test set will be used instead.

    Args:
         (....): See docstring of function :func:`train`. Note, ``hnet`` can be
            passed as ``None``. In this case, no weights are passed to the
            ``forward`` method of the main network.
        train_iter: The current training iteration.
    """
    logger.info('# Evaluating network on task %d ' % (task_id+1) +
                'before running training step %d ...' % (train_iter))

    # TODO: write histograms of weight samples to tensorboard.

    mnet.eval()
    if hnet is not None:
        hnet.eval()

    # Note, this function is called during training, where the temperature is
    # not changed.
    assert shared.softmax_temp[task_id] == 1.

    with torch.no_grad():
        split_name = 'test' if data.num_val_samples == 0 else 'validation'
        if split_name == 'test':
            logger.debug('Eval - Using test set as no validation set is ' +
                         'available.')

        # In contrast, we visualize uncertainty using the test set.
        acc, ret_vals = train_utils.compute_acc(task_id, data, mnet, hnet,
            device, config, shared, split_type='val', return_dataset=False,
            return_entropies=True, return_pred_labels=False)

        logger.info('Eval - Accuracy on %s set: %f%%.' % (split_name, acc))
        writer.add_scalar('eval/task_%d/accuracy' % task_id, acc,
                          train_iter)
        
        if not np.isnan(ret_vals.entropies.mean()):
            logger.debug('Eval - Entropy on %s set: %f (std: %f).'
                % (split_name, ret_vals.entropies.mean(),
                   ret_vals.entropies.std()))
            writer.add_scalar('eval/task_%d/mean_entropy' % task_id,
                              ret_vals.entropies.mean(), train_iter)
            writer.add_histogram('eval/task_%d/entropies' % task_id,
                                 ret_vals.entropies, train_iter)
        else:
            logger.warning('NaN entropy has been detected during evaluation!')

        logger.info('# Evaluating training ... Done')

def distill_net(task_id, data, mnet, hnet, hhnet, device, config, shared,
                logger, writer):
    """Distill the current weights of the main network into the hypernet.

    If the no hyper-hypernetwork ``hhnet`` is given, then this function takes
    the trained main network and distills its weights into the hypernet using
    the embedding corresponding to ``task_id``.

    Otherwise, the the hypernetwork ``hnet`` will be distilled into the
    hyper-hypernetwork ``hhnet`` using the embedding corresponding to
    ``task_id``.

    Args:
         (....): See docstring of function :func:`train`.
         hhnet: The hyper-hypernetwork. Only needs to be provided if the
             ``hnet`` instead of the ``mnet`` should be distilled.
    """
    pcutils.set_train_mode(True, mnet, hnet, hhnet, None)

    assert mnet is not None
    assert hnet is not None

    # FIXME Function might not work as expected in this case if
    # `config.regularizer` is not 'mse'.
    assert not isinstance(mnet, GaussianBNNWrapper) or hhnet is None

    if hhnet is None:
        distill_mnet = True
        assert mnet.weights is not None
        logger.info('Distilling main network into hypernet for task %d ...' \
                    % (task_id+1))
    else:
        distill_mnet = False
        assert hnet.unconditional_params is not None
        logger.info('Distilling hypernet into hyper-hypernet for task %d ...' \
                    % (task_id+1))

    #################################################
    ### Current main net accuracy as a reference ####
    #################################################
    if distill_mnet:
        mnet.eval()
        with torch.no_grad():
            mnet_acc, _ = train_utils.compute_acc(task_id, data, mnet, None,
                device, config, shared, split_type='val',
                deterministic_sampling=True,
                disable_lrt=config.disable_lrt_test)
        mnet.train()
    else:
        pcutils.set_train_mode(False, mnet, hnet, None, None)
        with torch.no_grad():
            mnet_acc, _ = pcutils.compute_acc(task_id, data, mnet, hnet, None,
                device, config, shared, split_type='val',
                deterministic_sampling=True)
        pcutils.set_train_mode(True, mnet, hnet, None, None)

    ############################
    ### Compute new target ####
    ############################
    if distill_mnet:
        # Sanity check.
        for i, p in enumerate(mnet.weights):
            assert np.all(np.equal(list(p.shape), mnet.param_shapes[i]))

        # The desired hypernet output for ``task_id``.
        mnet_w = torch.cat([p.detach().clone().flatten() \
                            for p in mnet.weights])
        mnet_w_mean = None
        mnet_w_logvar = None

        if isinstance(mnet, GaussianBNNWrapper) and \
                config.regularizer != 'mse':
            mnet_w_mean, mnet_w_rho = mnet.extract_mean_and_rho(weights=None)
            _, mnet_w_logvar = putils.decode_diag_gauss(mnet_w_rho, \
                logvar_enc=mnet.logvar_encoding, return_logvar=True)

            mnet_w_mean = [p.detach().clone() for p in mnet_w_mean]
            mnet_w_logvar = [p.detach().clone() for p in mnet_w_logvar]

        new_target = mnet_w

    else:
        # The desired hyper-hypernet output for ``task_id``.
        hnet_w = torch.cat([p.detach().clone().flatten() \
                            for p in hnet.unconditional_params])

        new_target = hnet_w

    ############################
    ### Setup CL regularizer ###
    ############################
    # I.e., setup the protection of old targets.
    if distill_mnet:
        regged_outputs = train_utils.calc_regged_out_inds(config, task_id, data)
    else:
        # FIXME We don't have any masking for a hypernetwork, that produces
        # mnet-head-specific weights, implemented.
        regged_outputs = None

    # Whether the regularizer will be computed during training?
    calc_reg = task_id > 0 and config.beta > 0 and \
        not config.train_from_scratch

    # Regularizer targets.
    if calc_reg:
        targets = None
        prev_hnet_theta = None
        prev_hhnet_theta = None
        prev_task_embs = None

        if distill_mnet:
            if config.calc_hnet_reg_targets_online:
                prev_hnet_theta = [p.detach().clone() \
                                   for p in hnet.unconditional_params]
                prev_task_embs = [p.detach().clone() \
                                  for p in hnet.conditional_params]
            else:
                targets, target_means, target_logvars = \
                    train_utils.calc_reg_target(config, task_id, hnet,
                                                mnet=mnet)
        else:
            if config.calc_hnet_reg_targets_online:
                prev_hhnet_theta = [p.detach().clone() \
                                    for p in hhnet.unconditional_params]
                prev_task_embs = [p.detach().clone() \
                                  for p in hhnet.conditional_params]
            else:
                targets = hreg.get_current_targets(task_id, hhnet)

    ########################
    ### Create optimizer ###
    ########################
    if distill_mnet:
        params = hnet.parameters()
    else:
        params = hhnet.parameters()
    optimizer = tutils.get_optimizer(params, config.lr,
        momentum=config.momentum, weight_decay=config.weight_decay,
        use_adam=config.use_adam, adam_beta1=config.adam_beta1,
        use_rmsprop=config.use_rmsprop, use_adadelta=config.use_adadelta,
        use_adagrad=config.use_adagrad)

    ##########################
    ### Start Distillation ###
    ##########################
    for i in range(config.distill_iter):
        ### Evaluate main net performance, to see how distillation is
        ### progressing.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            pcutils.set_train_mode(False, mnet, hnet, hhnet, None)
            with torch.no_grad():
                # Current accuracy using weights coming from (hyper-)hypernet.
                if distill_mnet:
                    hnet_acc, _ = train_utils.compute_acc(task_id, data, mnet,
                        hnet, device, config, shared, split_type='val',
                        deterministic_sampling=True,
                        disable_lrt=config.disable_lrt_test)
                else:
                    hnet_acc, _ = pcutils.compute_acc(task_id, data, mnet, hnet,
                        hhnet, device, config, shared, split_type='val',
                        deterministic_sampling=True)

            logger.info('Distill - Original main network accuracy: %f%%.' \
                        % (mnet_acc))
            logger.info('Distill - Current accuracy using ' + \
                        'hypernetwork: %f%%.' % (hnet_acc))
            writer.add_scalar('eval/task_%d/distill_acc_diff' % task_id,
                              mnet_acc - hnet_acc, i)
            pcutils.set_train_mode(True, mnet, hnet, hhnet, None)

        if i % 100 == 0:
            logger.debug('Distillation iteration: %d.' % i)

        optimizer.zero_grad()

        if distill_mnet:
            hnet_out = hnet.forward(cond_id=task_id)
            w_mean = None
            w_logvar = None

            if mnet_w_mean is not None:
                w_mean, w_rho = mnet.extract_mean_and_rho(weights=hnet_out)
                w_std, w_logvar = putils.decode_diag_gauss(w_rho, \
                    logvar_enc=mnet.logvar_encoding, return_logvar=True)
        else:
            hnet_out = hhnet.forward(cond_id=task_id)

        ### Compute distillation loss.
        # FIXME We could apply masking in the multi-head case.
        if not isinstance(mnet, GaussianBNNWrapper)  or \
                config.regularizer == 'mse':
            distilled_w = torch.cat([w.flatten() for w in hnet_out])
            dis_loss = (new_target - distilled_w).pow(2).sum()
        else:
            dis_loss = train_utils.calc_gauss_reg(config, task_id, mnet, hnet,
                target_mean=mnet_w_mean, target_logvar=mnet_w_logvar,
                current_mean=w_mean, current_logvar=w_logvar)

        ### Compute CL regularizer.
        loss_reg = 0
        if calc_reg:
            if config.regularizer == 'mse':
                if distill_mnet:
                    loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
                        targets=targets, mnet=mnet,
                        inds_of_out_heads=regged_outputs,
                        prev_theta=prev_hnet_theta,
                        prev_task_embs=prev_task_embs,
                        batch_size=config.hnet_reg_batch_size)
                else:
                    loss_reg = hreg.calc_fix_target_reg(hhnet, task_id,
                        targets=targets, prev_theta=prev_hhnet_theta,
                        prev_task_embs=prev_task_embs,
                        batch_size=config.hnet_reg_batch_size)

            else:
                task_target_means = None
                task_target_logvars = None
                if prev_hnet_theta is None:
                    task_target_means = target_means
                    task_target_logvars = target_logvars
                loss_reg =  train_utils.calc_gauss_reg_all_tasks(config,
                    task_id, mnet, hnet, target_means=task_target_means,
                    target_logvars=task_target_logvars,
                    prev_theta=prev_hnet_theta, prev_task_embs=prev_task_embs,
                    batch_size=config.hnet_reg_batch_size)

        loss = dis_loss + config.beta * loss_reg

        loss.backward()
        if config.clip_grad_value != -1:
            torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'],
                                            config.clip_grad_value)
        elif config.clip_grad_norm != -1:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],
                                            config.clip_grad_norm)
        optimizer.step()

        ###########################
        ### Tensorboard summary ###
        ###########################
        if i % 50 == 0:
            writer.add_scalar('train/task_%d/loss_distill' % task_id, dis_loss,
                              i)
            writer.add_scalar('train/task_%d/regularizer' % task_id, loss_reg,
                              i)
            writer.add_scalar('train/task_%d/loss_distill_total' % task_id,
                              loss, i)

    logger.info('Network distillation for task %d ... Done' % (task_id+1))

def train(task_id, data, mnet, hnet, device, config, shared, logger, writer):
    r"""Train the network using the task-specific loss plus a regularizer that
    should weaken catastrophic forgetting.

    .. math::
        \text{loss} = \text{task\_loss} + \beta * \text{regularizer}

    In case coresets are used to regularize towards high entropy on previous
    tasks, the loss becomes

    .. math::
        \text{loss} = \text{task\_loss} + \beta * \text{regularizer} + \
            \gamma * \text{coreset\_regularizer} 

    The task specific loss aims to learn the mean and variances of the main net
    weights such that the posterior parameter distribution is approximated.

    Args:
        task_id: The index of the task on which we train.
        data: The dataset handler.
        mnet: The model of the main network.
        hnet: The model of the hyper network (may be ``None``).
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        shared: Miscellaneous data shared among training functions.
        logger: Command-line logger.
        writer: The tensorboard summary writer.
    """
    logger.info('Training network on task %d ...' % (task_id+1))

    mnet.train()
    if hnet is not None:
        hnet.train()

    # Note, during training we start with a temperature of 1 (i.e., we can
    # ignore it when computing the network output (e.g., to evaluate the
    # cross-entropy loss)).
    # Only after training, the temperature may be changed.
    assert shared.softmax_temp[task_id] == 1.

    # Which outputs should we consider from the main network for the current
    # task.
    allowed_outputs = train_utils.out_units_of_task(config, data, task_id,
                                                    task_id+1)

    prev_mnet_params = None
    if config.coreset_size != -1 and \
            hasattr(config, 'coresets_for_experience_replay') and \
            config.coresets_for_experience_replay and task_id > 0:
        prev_mnet_params = [p.detach().clone() \
                               for p in mnet.internal_params]

    # It might be that tasks are very similar and we can transfer knowledge
    # from the previous solution.
    if hnet is not None and config.init_with_prev_emb and task_id > 0:
        # All conditional parameters (usually just task embeddings) are task-
        # specific and used for reinitialization.
        last_emb = get_conditional_parameters(hnet, task_id-1)
        for ii, cemb in enumerate(get_conditional_parameters(hnet, task_id)):
            cemb.data = last_emb[ii].data

    ####################
    ### Define Prior ###
    ####################
    # Whether prior-matching should even be performed?
    perform_pm = config.kl_scale != 0 or config.kl_schedule != 0

    # What prior to use for BbB training?
    standard_prior = False
    if config.use_prev_post_as_prior and task_id > 0:
        assert isinstance(mnet, GaussianBNNWrapper)

        if config.train_from_scratch:
            raise NotImplementedError()
            #warn('Loading network from previous task into new network ' +
            #     'instance. I.e., the training progress will be like ' +
            #     'fine-tuning, while the testing is done with separate ' +
            #     'networks.')
            #train_utils.load_networks(shared, task_id-1, device, logger, mnet,
            #                          hnet, hhnet=None, dis=None)

        if config.radial_bnn:
            # TODO Prior is not a Gaussian anymore.
            raise NotImplementedError()

        logger.debug('Choosing posterior of previous task as prior.')
        if allowed_outputs is not None:
            # FIXME Head-specific weights should be set to the Gaussian prior,
            # as we haven't learned a posterior for them when learning the
            # previous task.
            warn('Prior for head-specific weights not set correctly when ' +
                 'using "use_prev_post_as_prior".')
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
    else:
        prior_mean = shared.prior_mean
        prior_logvar = shared.prior_logvar
        prior_std = shared.prior_std
        if config.prior_variance == 1:
            # Use standard Gaussian prior with 0 mean and unit variance.
            standard_prior = True

    # Plot prior prior predictive distribution.
    if perform_pm and shared.experiment_type.startswith('gmm'):
        pgutils.plot_gmm_prior_preds(task_id, data, mnet, hnet, None, device,
            config, shared, logger, writer, prior_mean, prior_std)

    ############################
    ### Setup CL regularizer ###
    ############################
    # Whether the regularizer will be computed during training?
    calc_reg = hnet is not None  and task_id > 0 and config.beta > 0 and \
        not config.train_from_scratch

    # Usually, our regularizer acts on all weights of the main network.
    # Though, some weights might be connected to unused output neurons, which
    # is why we can ignore them.
    regged_outputs = None
    if calc_reg:
        regged_outputs = train_utils.calc_regged_out_inds(config, task_id, data)

    # Regularizer targets.
    if calc_reg:
        if config.calc_hnet_reg_targets_online:
            # Compute targets for the regularizer whenever they are needed.
            # -> Computationally expensive.
            targets = None
            prev_hnet_theta = [p.detach().clone() \
                               for p in hnet.unconditional_params]
            prev_task_embs = [p.detach().clone() \
                              for p in hnet.conditional_params]
        else:
            targets, target_means, target_logvars = \
                train_utils.calc_reg_target(config, task_id, hnet, mnet=mnet)
            prev_hnet_theta = None
            prev_task_embs = None

    ########################
    ### Create optimizer ###
    ########################
    if hnet is None:
        params = mnet.parameters()
    else:
        params = hnet.parameters()
    optimizer = tutils.get_optimizer(params, config.lr,
        momentum=config.momentum, weight_decay=config.weight_decay,
        use_adam=config.use_adam, adam_beta1=config.adam_beta1,
        use_rmsprop=config.use_rmsprop, use_adadelta=config.use_adadelta,
        use_adagrad=config.use_adagrad)

    ################################
    ### Learning rate schedulers ###
    ################################
    plateau_scheduler = None
    lambda_scheduler = None
    if config.plateau_lr_scheduler:
        assert config.epochs != -1
        # The scheduler config has been taken from here:
        # https://keras.io/examples/cifar10_resnet/
        # Note, we use 'max' instead of 'min' as we look at accuracy rather
        # than validation loss!
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau( \
            optimizer, 'max', factor=np.sqrt(0.1), patience=5,
            min_lr=0.5e-6, cooldown=0)

    if config.lambda_lr_scheduler:
        assert config.epochs != -1

        lambda_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
            tutils.lambda_lr_schedule)

    ######################
    ### Start training ###
    ######################
    mnet_kwargs = train_utils.mnet_kwargs(config, task_id, mnet)

    num_train_iter, iter_per_epoch = sutils.calc_train_iter( \
        data.num_train_samples, config.batch_size, num_iter=config.n_iter,
        epochs=config.epochs)

    for i in range(num_train_iter):
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

        # Modify 1-hot encodings according to CL scenario.
        assert(T.shape[1] == data.num_classes)
        # In CL1, CL2 and CL3 (with seperate heads) we do not have to modify the
        # targets.
        if config.cl_scenario == 3 and not config.split_head_cl3 and \
                task_id > 0:
            # We preprend zeros to the 1-hot vector according to the number of
            # output units belonging to previous tasks.
            T = torch.cat((torch.zeros((config.batch_size,
                                        task_id * data.num_classes)).to(device),
                           T), dim=1)

        _, labels = torch.max(T, 1) # Integer labels.
        labels = labels.detach()

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
        if perform_pm:
            if not config.radial_bnn:
                if standard_prior:
                    # Gaussian prior with zero mean and unit variance.
                    loss_kl = putils.kl_diag_gauss_with_standard_gauss(w_mean,
                                                                       w_logvar)
                else:
                    loss_kl = putils.kl_diag_gaussians(w_mean, w_logvar,
                                                       prior_mean, prior_logvar)
            else:
                # When using radial BNNs the weight distribution is not gaussian
                loss_kl = putils.kl_radial_bnn_with_diag_gauss(w_mean, w_std,
                    prior_mean, prior_std, ce_sample_size=config.num_kl_samples)
        else:
            loss_kl = 0

        ### Compute negative log-likelihood (NLL).
        loss_nll = 0
        mean_train_acc = 0
        for j in range(config.train_sample_size):
            if config.mean_only:
                Y = mnet.forward(X, weights=w_mean, **mnet_kwargs)
            else:
                # Note, the sampling will happen inside the forward method.
                Y = mnet.forward(X, weights=None, mean_only=False,
                    extracted_mean=w_mean, extracted_rho=w_rho, **mnet_kwargs)
            if allowed_outputs is not None:
                Y = Y[:, allowed_outputs]

            # Task-specific loss.
            # We use the reduction method 'mean' on purpose and scale with
            # the number of training samples below.
            # Note, that `cross_entropy` also computes the softmax for us.
            loss_nll += F.cross_entropy(Y, labels, reduction='mean')

            # Compute accuracy on batch.
            # Note, softmax wouldn't change the argmax.
            _, pred_labels = torch.max(Y, 1)
            mean_train_acc += 100. * torch.sum(pred_labels == labels) / \
                config.batch_size

        loss_nll *= data.num_train_samples / config.train_sample_size
        mean_train_acc /= config.train_sample_size

        ### Compute CL regularizer.
        loss_reg = 0
        if calc_reg:
            if config.regularizer == 'mse':
                loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
                    targets=targets, mnet=mnet, prev_theta=prev_hnet_theta,
                    prev_task_embs=prev_task_embs,
                    inds_of_out_heads=regged_outputs,
                    batch_size=config.hnet_reg_batch_size)

            else:
                task_target_means = None
                task_target_logvars = None
                if prev_hnet_theta is None:
                    task_target_means = target_means
                    task_target_logvars = target_logvars
                loss_reg =  train_utils.calc_gauss_reg_all_tasks(config,
                    task_id, mnet, hnet, target_means=task_target_means,
                    target_logvars=task_target_logvars,
                    prev_theta=prev_hnet_theta, prev_task_embs=prev_task_embs,
                    batch_size=config.hnet_reg_batch_size)

                loss_reg /= task_id

        ### Compute coreset regularizer.
        cs_reg = 0
        if config.coreset_size != -1 and not config.final_coresets_finetune \
                and (task_id > 0 or config.past_and_future_coresets):
            assert hasattr(shared, 'coreset')
            if config.past_and_future_coresets:
                cs_all = shared.coreset[shared.task_ident != task_id]
                batch_inds = np.random.randint(0, cs_all.shape[0],
                                               config.coreset_batch_size)
                coreset = cs_all[batch_inds]
            else:
                batch_inds = np.random.randint(0, shared.coreset.shape[0],
                                               config.coreset_batch_size)
                coreset = shared.coreset[batch_inds]
            #cs_task_ids = shared.task_ident[batch_inds]
            #cs_tasks = np.unique(cs_task_ids)

            # Construct maximum entropy targets for coreset samples.
            target_size = len(allowed_outputs) if config.cl_scenario != 2 \
                    else data.num_classes
            cs_targets = torch.ones(config.coreset_batch_size, target_size). \
                to(device) / target_size

            for j in range(config.train_sample_size):
                if config.mean_only:
                    cs_preds = mnet.forward(coreset, weights=w_mean,
                                            **mnet_kwargs)
                else:
                    # Note, the sampling will happen inside the forward method.
                    cs_preds = mnet.forward(coreset, weights=None,
                        mean_only=False, extracted_mean=w_mean,
                        extracted_rho=w_rho, **mnet_kwargs)
                if allowed_outputs is not None:
                    cs_preds = cs_preds[:, allowed_outputs]

                cs_reg += Classifier.softmax_and_cross_entropy(cs_preds,
                                                               cs_targets)
                #cs_reg += Classifier.knowledge_distillation_loss(cs_preds,
                #                                                 cs_targets)

            cs_reg /= config.train_sample_size

        ### Compute experience replay regularizer.
        er_reg = 0
        if config.coreset_size != -1 and \
                hasattr(config, 'coresets_for_experience_replay') and \
                config.coresets_for_experience_replay and task_id > 0:
            assert hasattr(shared, 'coreset')

            # If the overall coreset size (summer over all tasks) should stay
            # constant, we need to divide the coreset batch size by the
            # number of tasks.
            if hasattr(config, 'fix_coreset_size') and config.fix_coreset_size:
                sample_size = [config.coreset_batch_size // task_id] * task_id
                sample_size = [s + (1 if i < config.coreset_batch_size %task_id\
                               else 0) for i, s in enumerate(sample_size)]
                coreset_samples_per_task = np.random.permutation(sample_size)

            # Iterate over coresets of all past tasks.
            for prev_task_id in range(task_id):
                coreset_batch_size = config.coreset_batch_size
                if hasattr(config, 'fix_coreset_size') and \
                        config.fix_coreset_size:
                    coreset_batch_size = coreset_samples_per_task[prev_task_id]
                cs_curr_task = shared.coreset[shared.task_ident == prev_task_id]
                batch_inds = np.random.randint(0, cs_curr_task.shape[0],
                                               coreset_batch_size)
                coreset = cs_curr_task[batch_inds]

                # Get the targets from the previous model.
                with torch.no_grad():
                    mnet_kwargs_prev = train_utils.mnet_kwargs(config, \
                            prev_task_id, mnet)
                    er_targets = mnet.forward(coreset,
                        weights=prev_mnet_params, **mnet_kwargs_prev).to(device)

                for j in range(config.train_sample_size):
                    if config.mean_only:
                        er_preds = mnet.forward(coreset, weights=w_mean,
                                                **mnet_kwargs)
                    else:
                        raise NotImplementedError()

                    target_mapping = None
                    if allowed_outputs is not None:
                        # Select allowed outputs for targets.
                        allowed_outputs_prev = train_utils.out_units_of_task(\
                                                                config, data,
                                                                prev_task_id,
                                                                task_id)
                        er_targets = er_targets[:, allowed_outputs_prev]

                        # if growing head
                        if len(allowed_outputs_prev) != len(allowed_outputs):
                            target_mapping = list(allowed_outputs_prev)

                        # Select allowed outputs for predictions.
                        er_preds = er_preds[:, allowed_outputs]

                    er_reg += Classifier.knowledge_distillation_loss(er_preds,
                                     er_targets, target_mapping=target_mapping,
                                     device=device)

            if hasattr(config, 'fix_coreset_size') and config.fix_coreset_size:
                er_reg /= (config.train_sample_size)
            else:
                er_reg /= (config.train_sample_size * task_id)

        kl_scale = train_utils.calc_kl_scale(config, num_train_iter, i, logger)
        assert perform_pm or kl_scale == 0

        loss = kl_scale * loss_kl + loss_nll + config.beta * loss_reg + \
            config.coreset_reg * cs_reg + config.coreset_reg * er_reg

        loss.backward()
        if config.clip_grad_value != -1:
            torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'],
                                            config.clip_grad_value)
        elif config.clip_grad_norm != -1:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],
                                           config.clip_grad_norm)
        optimizer.step()

        ###############################
        ### Learning rate scheduler ###
        ###############################
        train_utils.apply_lr_schedulers(config, shared, logger, task_id, data,
            mnet, hnet, device, i, iter_per_epoch, plateau_scheduler,
            lambda_scheduler)

        ###########################
        ### Tensorboard summary ###
        ###########################
        if i % 50 == 0:
            writer.add_scalar('train/task_%d/loss_kl' % task_id, loss_kl, i)
            writer.add_scalar('train/task_%d/loss_nll' % task_id, loss_nll, i)
            writer.add_scalar('train/task_%d/regularizer' % task_id, loss_reg,
                              i)
            writer.add_scalar('train/task_%d/coreset_reg' % task_id, cs_reg, i)
            writer.add_scalar('train/task_%d/er_reg' % task_id, er_reg, i)
            writer.add_scalar('train/task_%d/loss' % task_id, loss, i)
            writer.add_scalar('train/task_%d/accuracy' % task_id,
                              mean_train_acc, i)

            # Plot distribution of mean and log-variance values.
            mean_outputs = torch.cat([d.clone().view(-1) for d in w_mean])
            writer.add_histogram('train/task_%d/predicted_means' % task_id,
                                 mean_outputs, i)
            if w_std is not None:
                std_outputs = torch.cat([d.clone().view(-1) for d in w_std])
                writer.add_histogram('train/task_%d/predicted_stds' % task_id,
                                     std_outputs, i)

    train_utils.checkpoint_bn_stats(config, task_id, mnet)

    ### Update or setup the coresets (if requested by user).
    if config.coreset_size != -1 and not (config.past_and_future_coresets or \
            config.final_coresets_finetune or \
            (hasattr(config, 'coresets_for_experience_replay') and \
             config.coresets_for_experience_replay)):
        train_utils.update_coreset(config, shared, task_id, data, mnet, hnet,
                                   device, logger, allowed_outputs)

    logger.info('Training network on task %d ... Done' % (task_id+1))

def train_multitask_coresets(dhandlers, mnet, hnet, device, config, shared, 
                             logger, writer):
    r"""Fine-tune the network on the coresets after training on all tasks.

    This function performs a multitask fine-tuning stage after the models have
    been sequentially trained on all tasks. For this stage, a set of samples
    stored in task-specific coresets which were not used during the sequential
    training are used.

    The mathematical derivation of this final update is given by the following.
    We consider a set of :math:`T` tasks, each described by a dataset 
    :math:`\mathcal{D}_t` which can be split into a sequential training part and
    a coreset part:

    .. math::

        \mathcal{D}_t = \mathcal{D}_t \setminus \mathcal{C}_t \cup \mathcal{C}_t

    Then learning the posterior of a set of parameters
    :math:`\theta` given the dataset can be written as follows:

    .. math::

        p(\theta \mid \mathcal{D}_t) &= \
        p(\theta \mid \mathcal{D}_t \setminus \mathcal{C}_t, \mathcal{C}_t) = \
        \frac{p(\theta) p(\mathcal{D}_t \mid \theta) }{ \
            p(\mathcal{D}_t \setminus \mathcal{C}_t, \mathcal{C}_t) } \
        = \frac{p(\theta) p(\mathcal{D}_t \setminus \mathcal{C}_t \mid \theta) \
        p(\mathcal{C}_t \mid \theta) }{ p(\mathcal{D}_t \setminus \
        \mathcal{C}_t, \mathcal{C}_t) } \\
        & \propto \frac{p(\theta) p(\mathcal{D}_t \setminus \mathcal{C}_t \mid \
        \theta) p(\mathcal{C}_t \mid \theta) }{ p(\mathcal{D}_t \setminus \
        \mathcal{C}_t) } = p(\theta \mid \mathcal{D}_t \setminus \mathcal{C}_t)\
        p(\mathcal{C}_t \mid \theta)

    Hence, when doing VI, we can utilize the following relation

    .. math::

        & \arg\min_{\xi_t} KL \big( q_{\xi_t}(\theta) \mid\mid \
        p(\theta \mid \mathcal{D}_t) \big) \\
        &\iff  \arg\min_{\xi_t} KL \big( q_{\xi_t}(\theta) \mid\mid \
        \frac{1}{Z} p(\theta \mid \mathcal{D}_t \setminus \mathcal{C}_t) \
        p(\mathcal{C}_t \mid \theta) \big) \\
        &\iff  \arg\min_{\xi_t} KL \big( q_{\xi_t}(\theta) \mid\mid \
        p(\theta \mid \mathcal{D}_t \setminus \mathcal{C}_t) \big)
        - \mathbb{E}_{q_{\xi_t}(\theta)} [\log p(\mathcal{C}_t \mid \theta)] \\
        &\iff  \arg\min_{\xi_t} KL \big( q_{\xi_t}(\theta) \mid\mid \
        \tilde{q}_{\psi_t}(\theta) \big)
        - \mathbb{E}_{q_{\xi_t}(\theta)} [\log p(\mathcal{C}_t \mid \theta)]

    where :math:`Z` is an appropriate normalization constant and
    :math:`\tilde{q}_{\psi_t}(\theta)` are the approximate per-task posteriors
    that we obtain by learning sequentially on all
    :math:`\mathcal{D}_t \setminus \mathcal{C}_t`
    (i.e. doing our standard CL training), and thus
    :math:`\tilde{q}_{\psi_t}(\theta) \approx  p(\theta \mid \mathcal{D}_t \setminus \mathcal{C}_t)`.

    This objective can be therefore realised by minimising the negative
    log-likelihood on the coresets, while adding a prior-matching term that
    makes the final approximation close to the posteriors found after the
    sequential training.

    Importantly, since at the end of the sequential training we have access to
    all coresets, these can be used to improve uncertainty estimates of out-of-
    distribution data, i.e. one can train the task-specific heads to
    produce high uncertainty estimates for the input data of other tasks. This
    can be achieved by either having maximum entropy labels or random labels in
    OOD data.

    Args: 
        (....): See docstring of function :func:`train`.
        dhandlers: The dataset handlers of all tasks.
    """
    logger.info('### Multitask fine-tuning using coresets. ###')

    mnet.train()
    if hnet is not None:
        hnet.train()

    #############################################
    ### Get the posteriors before fine-tuning ###
    #############################################
    if config.radial_bnn:
        raise NotImplementedError('Unclear how to do prior-focused learning ' +
                                  'with radial posteriors.')
    if config.mean_only:
        raise ValueError('Fine-tuning cannot be implemented for ' +
                         'deterministic networks, as fine-tuning involves a ' +
                         'prior-focused update!')
    if config.train_from_scratch:
        # TODO reload old models to get their posteriors.
        raise NotImplementedError()

    priors_mean = []
    priors_logvar = []
    priors_std = []
    priors_rho = []
    for task_id in range(config.num_tasks):
        ### Get the mean and variances for the checkpointed model.
        if hnet is None:
            hnet_out = None
        else:
            hnet_out = hnet.forward(cond_id=task_id)

        prior_mean, prior_rho = mnet.extract_mean_and_rho(weights=hnet_out)
        prior_std, prior_logvar = putils.decode_diag_gauss(prior_rho,
            logvar_enc=mnet.logvar_encoding, return_logvar=True)

        prior_mean = [p.detach().clone() for p in prior_mean]
        prior_std = [p.detach().clone() for p in prior_std]
        prior_logvar = [p.detach().clone() for p in prior_logvar]
        prior_rho = [p.detach().clone() for p in prior_rho]

        priors_mean.append(prior_mean)
        priors_logvar.append(prior_logvar)
        priors_std.append(prior_std)
        priors_rho.append(prior_rho)

    ########################
    ### Create optimizer ###
    ########################
    if hnet is None:
        params = mnet.parameters()
    else:
        params = hnet.parameters()
    optimizer = tutils.get_optimizer(params, config.lr,
        momentum=config.momentum, weight_decay=config.weight_decay,
        use_adam=config.use_adam, adam_beta1=config.adam_beta1,
        use_rmsprop=config.use_rmsprop, use_adadelta=config.use_adadelta,
        use_adagrad=config.use_adagrad)

    ######################
    ### Start training ###
    ######################
    n_iter = config.n_iter
    if config.final_coresets_n_iter != -1:
        n_iter = config.final_coresets_n_iter
    epochs = config.epochs
    if config.final_coresets_epochs != -1:
        epochs = config.final_coresets_epochs
    num_train_samples = shared.coreset.shape[0]
    num_train_iter, iter_per_epoch = sutils.calc_train_iter( \
        num_train_samples, config.coreset_batch_size, num_iter=n_iter,
        epochs=epochs)

    for i in range(num_train_iter):
        if i % 100 == 0:
            logger.debug('Fine-tuning iteration: %d.' % i)

        kl_scale = train_utils.calc_kl_scale(config, num_train_iter, i, logger,
                                             final_finetune=True)

        ### Train theta and task embedding.
        optimizer.zero_grad()

        loss_nll = 0
        loss_kl = 0
        # Iterate over tasks to accumulate nll and prior-matching losses.
        for task_id, data in enumerate(dhandlers):
            mnet_kwargs = train_utils.mnet_kwargs(config, task_id, mnet)

            # Which outputs should we consider from the main network for the
            # current task.
            allowed_outputs = train_utils.out_units_of_task(config, data,
                                                            task_id, task_id+1)

            ### Get the mean and variances with the current model.
            if hnet is None:
                hnet_out = None
            else:
                hnet_out = hnet.forward(cond_id=task_id)

            w_mean, w_rho = mnet.extract_mean_and_rho(weights=hnet_out)
            w_std, w_logvar = putils.decode_diag_gauss(w_rho, \
                logvar_enc=mnet.logvar_encoding, return_logvar=True)

            loss_kl_curr_task = 0
            if i > 0: # Note, the KL is zero in the first iteration.
                ### Prior-matching loss.
                if not config.radial_bnn:
                    loss_kl_curr_task = putils.kl_diag_gaussians(w_mean,
                        w_logvar, priors_mean[task_id], priors_logvar[task_id])
                    loss_kl += loss_kl_curr_task
                else:
                    # We would need to be able to estimate the KL between two
                    # Radial distributions.
                    raise NotImplementedError()

            ### Get the coreset for the current task and its targets.
            if config.final_coresets_single_task:
                #num_ft_samples = shared.task_ident.size // config.num_tasks
                num_ft_samples = config.coreset_size

                curr_task_inds = np.where(shared.task_ident == task_id)[0]
                curr_task_inds_tmp = curr_task_inds.copy()
                while curr_task_inds.size < config.coreset_batch_size:
                    curr_task_inds = np.concatenate((curr_task_inds,
                                                     curr_task_inds_tmp))
                np.random.shuffle(curr_task_inds)
                batch_inds = curr_task_inds[:config.coreset_batch_size]
            else:
                num_ft_samples = shared.task_ident.size

                if config.final_coresets_balance == -1:
                    batch_inds = np.random.randint(0, shared.coreset.shape[0],
                                                   config.coreset_batch_size)
                else:
                    p = config.final_coresets_balance
                    n_curr = int(np.ceil(p * config.coreset_batch_size))
                    n_other = config.coreset_batch_size - n_curr

                    curr_task_inds = np.where(shared.task_ident == task_id)[0]
                    # Note, we assume that there is an equal amount of samples
                    # of each task in the complete coreset.
                    curr_task_inds = np.tile(curr_task_inds, config.num_tasks-1)
                    other_task_inds = np.where(shared.task_ident != task_id)[0]

                    curr_task_inds_tmp = curr_task_inds.copy()
                    while curr_task_inds.size < n_curr:
                        curr_task_inds = np.concatenate((curr_task_inds,
                                                         curr_task_inds_tmp))
                    other_task_inds_tmp = other_task_inds.copy()
                    while other_task_inds.size < n_other:
                        other_task_inds = np.concatenate((other_task_inds,
                                                          other_task_inds_tmp))

                    np.random.shuffle(curr_task_inds)
                    np.random.shuffle(other_task_inds)

                    batch_inds = np.concatenate((
                        curr_task_inds[:n_curr],
                        other_task_inds[:n_other]
                    ))

            cs_inps = shared.coreset[batch_inds]
            cs_trgts = shared.coreset_targets[batch_inds]

            # Modify 1-hot encodings according to CL scenario.
            assert(cs_trgts.shape[1] == data.num_classes)
            # Modify the targets, if softmax spans multiple heads.
            cs_trgts = train_utils.fit_targets_to_softmax(config, shared,
                device, data, task_id, cs_trgts)

            if not config.final_coresets_single_task:
                other_cs_inds = np.where(shared.task_ident != task_id)[0]
                other_batch_inds_mask = np.isin(batch_inds, other_cs_inds)
                other_batch_inds = batch_inds[other_batch_inds_mask]

                num_other = other_batch_inds.size
                target_size = cs_trgts.shape[1]
                if config.final_coresets_use_random_labels:
                    other_rnd_labels = torch.randint(0, target_size, \
                        (num_other,))
                    other_targets = torch.zeros(num_other,
                                                target_size).to(device)
                    other_targets[range(num_other), other_rnd_labels] = 1
                else:
                    # Construct maximum entropy targets for coreset samples.
                    other_targets = torch.ones(num_other,
                        target_size).to(device) / target_size

                cs_trgts[other_batch_inds_mask, :] = other_targets


            ### Get NLL on the constructed coreset batch.
            loss_nll_curr_task = 0
            for j in range(config.train_sample_size):
                # Note, the sampling will happen inside the forward method.
                cs_preds = mnet.forward(cs_inps, weights=None,
                    mean_only=False, extracted_mean=w_mean,
                    extracted_rho=w_rho, **mnet_kwargs)
                if allowed_outputs is not None:
                    cs_preds = cs_preds[:, allowed_outputs]
                loss_nll_curr_task += Classifier.softmax_and_cross_entropy( \
                    cs_preds, cs_trgts)
            loss_nll_curr_task *= num_ft_samples / config.train_sample_size
            loss_nll += loss_nll_curr_task

            loss_curr_task = kl_scale * loss_kl_curr_task + loss_nll_curr_task
            loss_curr_task.backward()

        loss = kl_scale * loss_kl + loss_nll
        # Note, we call `backward` above inside the loop, and accumulate
        # gradients, to avoid memory issues.
        #loss.backward()
        if config.clip_grad_value != -1:
            torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'],
                                            config.clip_grad_value)
        elif config.clip_grad_norm != -1:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],
                                           config.clip_grad_norm)
        optimizer.step()

        ###########################
        ### Tensorboard summary ###
        ###########################
        if i % 50 == 0:
            writer.add_scalar('fine_tuning/loss_kl', loss_kl, i)
            writer.add_scalar('fine_tuning/loss_nll', loss_nll, i)
            writer.add_scalar('fine_tuning/loss', loss, i)

    logger.info('Multitask fine-tuning using coresets ... Done')

def run(config, experiment='split_bbb'):
    """Run the training.

    Args:
        config: Command-line arguments.
        experiment: Which kind of experiment should be performed?

            - "gmm_bbb": Synthetic Gaussian Mixture Model Dataset
            - "split_bbb": Split MNIST
            - "perm_bbb": Permuted MNIST
            - "cifar_zenke_bbb": CIFAR-10/100 using a ZenkeNet
            - "cifar_resnet_bbb": CIFAR-10/100 using a Resnet
    """
    script_start = time()

    device, writer, logger = sutils.setup_environment(config,
        logger_name=experiment + 'logger')

    rutils.backup_cli_command(config)

    ### Create tasks.
    dhandlers = train_utils.load_datasets(config, logger, experiment, writer)

    ### Generate networks.
    use_hnet = not config.mnet_only
    # If there is a hnet, then we normally don't require mnet weights, except if
    # we use distillation.
    no_mnet_weights = use_hnet and config.distill_iter == -1
    mnet, hnet = train_utils.generate_gauss_networks(config, logger, dhandlers,
        device, experiment, no_mnet_weights=no_mnet_weights,
        create_hnet=use_hnet)

    ### Simple struct, that is used to share data among functions.
    shared = Namespace()
    shared.experiment_type = experiment
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

    # The weights of the main network right after training on that task
    # (can be used to assess how close the final weights are to the original
    # ones). Note, weights refer to mean and variances (e.g., the output of the
    # hypernetwork).
    shared.during_weights = [-1] * config.num_tasks if hnet is not None \
        else None

    # Where to save network checkpoints?
    shared.ckpt_dir = os.path.join(config.out_dir, 'checkpoints')
    # Note, some main networks have stuff to store such as batch statistics for
    # batch norm. So it is wise to always checkpoint mnets as well!
    shared.ckpt_mnet_fn = os.path.join(shared.ckpt_dir, 'mnet_task_%d')
    shared.ckpt_hnet_fn = os.path.join(shared.ckpt_dir, 'hnet_task_%d')

    # Initialize the softmax temperature per-task with one. Might be changed
    # later on to calibrate the temperature.
    shared.softmax_temp = [torch.ones(1).to(device) \
                           for _ in range(config.num_tasks)]
    shared.num_trained = 0

    # Setup coresets iff regularization on all tasks is allowed.
    if config.coreset_size != -1 and (config.past_and_future_coresets or \
            config.final_coresets_finetune or \
            (hasattr(config, 'coresets_for_experience_replay') and \
             config.coresets_for_experience_replay)):
        for i in range(config.num_tasks):
            train_utils.update_coreset(config, shared, i, dhandlers[i], None,
                                       None, device, logger, None)

    # If the coresets will be used after training on all tasks for fine-
    # tuning, we remove those samples from the training datahandlers.
    if config.coreset_size != -1 and config.final_coresets_finetune:
        train_utils.remove_coreset_from_training_data(config, logger,
                                                      dhandlers, shared)

    ### Initialize summary.
    pcutils.setup_summary_dict(config, shared, experiment, mnet, hnet=hnet,
                               hhnet=None, dis=None)
    logger.info('Ratio num hnet weights / num mnet weights: %f.'
                % shared.summary['num_weights_hm_ratio'])

    # Add hparams to tensorboard, such that the identification of runs is
    # easier.
    writer.add_hparams(hparam_dict={**vars(config), **{
        'num_weights_main': shared.summary['num_weights_main'],
        'num_weights_hyper': shared.summary['num_weights_hyper'],
        'num_weights_hm_ratio': shared.summary['num_weights_hm_ratio'],
    }}, metric_dict={})

    during_acc_criterion = train_utils.parse_performance_criterion(config,
        shared, logger)

    ### Train on tasks sequentially.
    for i in range(config.num_tasks):
        logger.info('### Training on task %d ###' % (i+1))
        data = dhandlers[i]

        # Train the network.
        shared.num_trained += 1
        if config.distill_iter == -1:
            train(i, data, mnet, hnet, device, config, shared, logger, writer)
        else:
            assert hnet is not None
            # Train main network only.
            train(i, data, mnet, None, device, config, shared, logger, writer)
            # Distill main network into hypernet.
            distill_net(i, data, mnet, hnet, None, device, config, shared,
                        logger, writer)

            # Create a new main network before training the next task.
            mnet, _ = train_utils.generate_gauss_networks(config, logger,
                dhandlers, device, experiment, no_mnet_weights=False,
                create_hnet=False)

        ### Temperature Calibration.
        if config.calibrate_temp:
            pcutils.calibrate_temperature(i, data, mnet, hnet, None, device,
                                          config, shared, logger, writer)

        ### Test networks.
        test_ids = None
        if config.full_test_interval != -1:
            if i == config.num_tasks-1 or \
                    (i > 0 and i % config.full_test_interval == 0):
                test_ids = None # Test on all tasks.
            else:
                test_ids = [i] # Only test on current task.
        train_vi.test(dhandlers[:(i+1)], mnet, hnet, None, device, config,
                       shared, logger, writer, test_ids=test_ids, method='bbb')

        ### Check if last task got "acceptable" accuracy ###
        curr_dur_acc = shared.summary['acc_task_given_during'][i]
        if i < config.num_tasks-1 and during_acc_criterion[i] != -1 \
                and during_acc_criterion[i] > curr_dur_acc:
            logger.error('During accuracy of task %d too small (%f < %f).' % \
                         (i+1, curr_dur_acc, during_acc_criterion[i]))
            logger.error('Training of future tasks will be skipped')
            writer.close()
            exit(1)

        if config.train_from_scratch and i < config.num_tasks-1:
            # We have to checkpoint the networks, such that we can reload them
            # for task inference later during testing.
            train_utils.checkpoint_nets(config, shared, i, mnet, hnet)

            mnet, hnet = train_utils.generate_gauss_networks(config, logger,
                dhandlers, device, experiment, no_mnet_weights=no_mnet_weights,
                create_hnet=use_hnet)

    ### Multitask fine-tuning using all the coresets.
    if config.final_coresets_finetune:
        train_multitask_coresets(dhandlers, mnet, hnet, device, config, shared,
            logger, writer)

        train_vi.test(dhandlers, mnet, hnet, None, device, config, shared,
                      logger, writer, method='bbb')

    if config.store_final_model:
        logger.info('Checkpointing final model ...')
        train_utils.checkpoint_nets(config, shared, config.num_tasks-1, mnet,
                                    hnet)

    logger.info('During accuracies (task identity given): %s (avg: %.2f%%).' % \
        (np.array2string(np.array(shared.summary['acc_task_given_during']),
                         precision=2, separator=','),
         shared.summary['acc_avg_task_given_during']))
    logger.info('Final accuracies (task identity given): %s (avg: %.2f%%).' % \
        (np.array2string(np.array(shared.summary['acc_task_given']),
                         precision=2, separator=','),
         shared.summary['acc_avg_task_given']))

    logger.info('During accuracies (task identity inferred): ' +
                '%s (avg: %.2f%%).' % \
        (np.array2string(np.array( \
            shared.summary['acc_task_inferred_ent_during']),
                         precision=2, separator=','),
         shared.summary['acc_avg_task_inferred_ent_during']))
    logger.info('Final accuracies (task identity inferred): ' +
                '%s (avg: %.2f%%).' % \
        (np.array2string(np.array(shared.summary['acc_task_inferred_ent']),
                         precision=2, separator=','),
         shared.summary['acc_avg_task_inferred_ent']))

    logger.info('### Avg. during accuracy (CL scenario %d): %.4f.'
                % (config.cl_scenario, shared.summary['acc_avg_during']))
    logger.info('### Avg. final accuracy (CL scenario %d): %.4f.'
                % (config.cl_scenario, shared.summary['acc_avg_final']))

    ### Write final summary.
    shared.summary['finished'] = 1
    train_utils.save_summary_dict(config, shared, experiment)

    writer.close()

    logger.info('Program finished successfully in %f sec.'
                % (time() - script_start))

if __name__ == '__main__':
    # Consult README file!
    raise Exception('Script is not executable!')


