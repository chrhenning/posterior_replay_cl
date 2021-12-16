#!/usr/bin/env python3
# Copyright 2021 Christian Henning
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
# @title          :probabilistic/ewc_utils.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/10/2021
# @version        :1.0
# @python_version :3.8.5
"""
Continual Learning with EWC
---------------------------

This module contains all utilities necessary to train CL experiments with
Elastic Weight Consolidation (EWC), first introduced in

    Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks",
    PNAS, 2017.

However, we consider the mathematical rigorous method here, called `Online EWC`
(see `<Ferenc Huszár https://arxiv.org/abs/1712.03847>`_and
`<Schwarz et al. https://arxiv.org/pdf/1805.06370.pdf>`_for details).

We ensure that the training can be performed such that all mathematical details
are mirrored in the implementation. In addition, we consider multiple inference
scenarios, most notably the `multi-head` scenario.

In the `multi-head` case, each head has its own (task-specific) posterior
approximation, represented by its Fisher matrix. I.e., having a slightly
different Laplace approximation for each task, we can build task-specific
posteriors and use them for task inference.
"""
from argparse import Namespace
import math
import numpy as np
import os
from time import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from warnings import warn

from probabilistic import train_vi
from probabilistic.prob_cifar import train_utils as pcutils
from probabilistic.prob_mnist import train_bbb as class_bbb
from probabilistic.prob_mnist import train_utils as pmutils
from probabilistic.regression import train_bbb as reg_bbb
from probabilistic.regression import train_utils as rutils
import utils.ewc_regularizer as ewc
from utils import torch_utils as tutils
from utils import sim_utils as sutils

def build_ewc_posterior(data_handlers, mnet, device, config, shared, logger,
                        writer, num_trained, task_id=None):
    """Build a normal posterior after having trained using EWC.

    The posterior is constructed as described in function :func:`test`.

    Args:
        (....): See docstring of function :func:`probailistic.train_vi.test`.
        num_trained (int): The number of output heads that already have been
            trained.
        task_id (int, optional): If training from scratch, only a specific head
            has been trained, that has to be specified via this argument.
            
            Note:
                If training from scratch, it is assumed that the correct
                ``mnet`` (corresponding to ``task_id``) has already been loaded
                to memory. This function will not load any checkpoints!
    """
    n = num_trained

    # Build posterior from Fisher approximations.
    is_regression = 'regression' in shared.experiment_type
    is_multihead = None
    if is_regression:
        is_multihead = config.multi_head
    else:
        is_multihead = config.cl_scenario == 1 or \
                       config.cl_scenario == 3 and config.split_head_cl3

    if is_multihead:
        post_means = [None] * len(mnet.internal_params)
        post_stds = [None] * len(mnet.internal_params)

        out_inds = [pmutils.out_units_of_task(config, data_handlers[i], i,
                                              None) for i in range(n)]

        out_masks = [mnet.get_output_weight_mask(out_inds=out_inds[i], \
            device=device) for i in range(n)]

        for ii, mask in enumerate(out_masks[0]):
            pind = mnet.param_shapes_meta[ii]['index']
            buff_w_name, buff_f_name = ewc._ewc_buffer_names(None, pind, True)

            if mask is None: # Shared parameters.
                post_means[pind] = getattr(mnet, buff_w_name)
                # The hessian that is approximated in EWC is corresponds to the
                # inverse variance.
                post_stds[pind] = getattr(mnet, buff_f_name).pow(-.5)
            else:
                # Initialize head weights to prior.
                curr_m = torch.zeros_like(getattr(mnet, buff_w_name)).to(device)
                curr_s = torch.ones_like(getattr(mnet, buff_w_name)).\
                    to(device) * math.sqrt(config.prior_variance)

                # Update head weights for trained output heads.
                for jj, t_mask in enumerate(out_masks):
                    # Note, if we train from scratch, then also all previous
                    # output heads are not trained, thus we let those weights
                    # follow the prior.
                    if not config.train_from_scratch or jj == task_id:
                        m = t_mask[ii]
                        curr_m[m] = getattr(mnet, buff_w_name)[m]
                        curr_s[m] = getattr(mnet, buff_f_name)[m].pow(-.5)

                post_means[pind] = curr_m
                post_stds[pind] = curr_s

        # Quick and dirty solution. Note, that a Pytorch `Normal` object with
        # zero std will just return the mean.
        if hasattr(config, 'det_multi_head') and config.det_multi_head:
            post_stds = [torch.zeros_like(t) for t in post_stds]

        return post_means, post_stds

    return None

def test(data_handlers, mnet, device, config, shared, logger, writer,
         test_ids=None):
    r"""Test the performance on all tasks trained via EWC.

    See docstring of function :func:`probailistic.train_vi.test`.

    We consider the following posterior

    .. math::

        \log p(\theta, \psi_A, \cdots, \psi_T \mid \mathcal{D}_A, \cdots \
            \mathcal{D}_T) = \mathcal{N}\bigg( \phi_T^*, \bigg[ \
            \frac{1}{\sigma_{prior}^2} I + \sum_{t \in {A \cdots T}}  N_t \
            \mathcal{F}_{emp \: t} \bigg]^{-1} \bigg)

    The posterior distribution of task-specific weights of future tasks is
    set to the prior. If training from scratch, then all heads other than the
    trained task are set to the prior.

    Args:
        (....): See docstring of function :func:`probailistic.train_vi.test`.
    """
    is_regression = 'regression' in shared.experiment_type
    is_multihead = None
    if is_regression:
        is_multihead = config.multi_head
    else:
        # FIXME In CL1 er might not want to construct a posterior
        # distrubtion.
        is_multihead = config.cl_scenario == 1 or \
                       config.cl_scenario == 3 and config.split_head_cl3

    if not is_multihead:
        warn('Task inference calculated in test method doesn\'t make any ' +
             'sense, as there is only one model with a shared output!')

    # FIXME If we are not training from scratch (but `is_multihead` is True),
    # then the uncertainty of each task can be determined by drawing samples
    # from the same posterior. However, the test methods use below iterate over
    # each task, draw weights from the posterior and determine the uncertainty
    # only of this current task. This is quite wasteful computation. But as
    # long as the sampling from the posterior is deterministic or sufficient
    # samples are drawn, it will lead to the same uncertainty estimates.
    if is_regression:
        assert test_ids is None
        reg_bbb.test(data_handlers, mnet, None, device, config, shared, logger,
                 writer)
    else:
        train_vi.test(data_handlers, mnet, None, None, device, config, shared,
                      logger, writer, test_ids=test_ids, method='ewc')

def train(task_id, data, mnet, device, config, shared, logger, writer):
    r"""Train a network continually using EWC.

    In general, we consider networks with task shared weights :math:`\theta` and
    task-specific weights (usually the output head weights) :math:`\psi_t`. The
    EWC loss function then arises from the following identity.

    .. math::

        \log p(\theta, \psi_A, \cdots, \psi_T \mid \mathcal{D}_A, \cdots \
             \mathcal{D}_T) &= \log p(\mathcal{D}_T \mid \theta, \psi_T) + \
            \log p(\psi_T) + \sum_{t < T} \bigg[ \log p(\mathcal{D}_t \mid \
            \theta, \psi_t)  + \log p(\psi_t) \bigg] + \log p(\theta) + const \
            \\  &= \log p(\mathcal{D}_T \mid \theta, \psi_T) + \log p(\psi_T) \
            + \log p(\theta, \psi_A \cdots \psi_S \mid \mathcal{D}_A \cdots \
            \mathcal{D}_S) + const

    If there is a single head (or combined head/softmax) such that there are no
    task-specific weights, the :math:`\psi_t`'s can be dropped from the
    equation.

    The (online) EWC loss function can then be derived to be

    .. math::

        \log p(\theta, \psi_A, \cdots, \psi_T \mid \mathcal{D}_A, \cdots \
            \mathcal{D}_T) &\approx  const + \log p(\mathcal{D}_T \mid \theta, \
            \psi_T) + \log p(\psi_T) \\ \
            & \hspace{1cm} - \frac{1}{2} \sum_{i \in \mid \phi \mid} \bigg( \
            \frac{1}{\sigma_{prior}^2} + \sum_{t \in {A \cdots S}}  N_t \
            \mathcal{F}_{emp \: t, i}  \bigg) (\phi_i - \phi_{S, i}^*)^2

    where :math:`\phi` refers to all task-shared weights as well as all
    task-specific weights of previously seen tasks.

    Hence, each weight has its own regularization factor computed as a sum from
    a constant offset (assuming an isotropic prior) and a weighted accumulation
    of Fisher values from all previous tasks. Note, Fisher values of
    task-specific weights are only non-zero when computed on the corresponding
    task.

    As only task-shared and the current output head are being learned, the
    regularizer is trivially zero for all other task-specific weights.

    When learning the first task, we need to find a MAP solution by finding the
    argmax of:

    .. math::

        \log p(\theta, \psi_A \mid \mathcal{D}_A) =  const + \
            \log p(\mathcal{D}_A \mid \theta, \psi_A) + \log p(\theta) +\
            \log p(\psi_A)

    We assume isotropic Gaussian posteriors and therefore can transform the
    prior terms into simple L2 regularization (or weight decay) expressions:

    .. math::

        \log p(\theta) = -\frac{1}{2 \sigma_{prior}^2} \lVert \theta \rVert_2^2

    Args:
        task_id: The index of the task on which we train.
        data: The dataset handler.
        mnet: The model of the main network.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        shared: Miscellaneous data shared among training functions.
        logger: Command-line logger.
        writer: The tensorboard summary writer.
    """
    logger.info('Training network on task %d ...' % (task_id+1))

    mnet.train()

    # Whether we train a classification or regression task?
    is_regression = 'regression' in shared.experiment_type
    # If we have a multihead setting, then we need to distinguish between
    # task-specific and task-shared weights.
    is_multihead = None
    if is_regression:
        assert config.ll_dist_std > 0
        eval_func = reg_bbb.evaluate
        ll_scale = 1. / config.ll_dist_std**2
        is_multihead = config.multi_head
    else:
        assert shared.softmax_temp[task_id] == 1.
        eval_func = class_bbb.evaluate
        is_multihead = config.cl_scenario == 1 or \
            config.cl_scenario == 3 and config.split_head_cl3

    # Which outputs should we consider from the main network for the current
    # task.
    allowed_outputs = pmutils.out_units_of_task(config, data, task_id,
                                                task_id+1)

    #############################################################
    ### Figure out which are task-specific and shared weights ###
    #############################################################

    if is_multihead:
        # Note, that output weights of all output heads share always the same
        # parameter tensors, which is the case at the time of implementation
        # for all mnets.
        out_masks = mnet.get_output_weight_mask(out_inds=allowed_outputs,
                                                device=device)

        shared_params = []
        specific_params = []
        # Within an output weight tensor, we only want to apply the L2 reg to
        # the corresponding output weights.
        specific_mask = []

        for ii, mask in enumerate(out_masks):
            pind = mnet.param_shapes_meta[ii]['index']
            assert pind != -1
            if mask is None: # Shared parameter.
                shared_params.append(mnet.internal_params[pind])
            else: # Output weight tensor.
                specific_params.append(mnet.internal_params[pind])
                specific_mask.append(mask)
    else: # All weights are task-shared.
        shared_params = mnet.internal_params
        specific_params = None

    ###########################
    ### Create optimizer(s) ###
    ###########################
    # For the non-multihead case, we could invoke the L2 reg via the
    # weight-decay parameter here. But for the multihead case, we need to apply
    # an extra mask to the parameter tensor.
    optimizer = tutils.get_optimizer(mnet.internal_params, config.lr,
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
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau( \
            optimizer, 'min' if is_regression else 'max', factor=np.sqrt(0.1),
            patience=5, min_lr=0.5e-6, cooldown=0)

    if config.lambda_lr_scheduler:
        assert config.epochs != -1

        lambda_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
            tutils.lambda_lr_schedule)

    ######################
    ### Start training ###
    ######################
    mnet_kwargs = pmutils.mnet_kwargs(config, task_id, mnet)

    num_train_iter, iter_per_epoch = sutils.calc_train_iter( \
        data.num_train_samples, config.batch_size, num_iter=config.n_iter,
        epochs=config.epochs)

    for i in range(num_train_iter):
        #########################
        ### Evaluate networks ###
        #########################
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            eval_func(task_id, data, mnet, None, device, config, shared, logger,
                      writer, i)
            mnet.train()

        if i % 100 == 0:
            logger.debug('Training iteration: %d.' % i)

        ##########################
        ### Train Current Task ###
        ##########################
        optimizer.zero_grad()

        ### Compute negative log-likelihood (NLL).
        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        T = data.output_to_torch_tensor(batch[1], device, mode='train')

        if not is_regression:
            # Modify 1-hot encodings according to CL scenario.
            assert(T.shape[1] == data.num_classes)
            # Modify the targets, if softmax spans multiple heads.
            T = pmutils.fit_targets_to_softmax(config, shared, device, data,
                                               task_id, T)

            _, labels = torch.max(T, 1) # Integer labels.
            labels = labels.detach()

        Y = mnet.forward(X, **mnet_kwargs)
        if allowed_outputs is not None:
            Y = Y[:, allowed_outputs]

        # Task-specific loss.
        # We use the reduction method 'mean' on purpose and scale with
        # the number of training samples below.
        if is_regression:
            loss_nll = 0.5 * ll_scale * F.mse_loss(Y, T, reduction='mean')
        else:
            # Note, that `cross_entropy` also computed the softmax for us.
            loss_nll = F.cross_entropy(Y, labels, reduction='mean')

            # Compute accuracy on batch.
            # Note, softmax wouldn't change the argmax.
            _, pred_labels = torch.max(Y, 1)
            mean_train_acc = 100. * torch.sum(pred_labels == labels) / \
                config.batch_size

        loss_nll *= data.num_train_samples

        ### Compute L2 reg.
        loss_l2 = 0
        if task_id == 0 or config.train_from_scratch:
            for pp in shared_params:
                loss_l2 += pp.pow(2).sum()
        if specific_params is not None:
            for ii, pp in enumerate(specific_params):
                loss_l2 += (pp * specific_mask[ii]).pow(2).sum()
        loss_l2 *= 1. / (2. * config.prior_variance)

        ### Compute EWC reg.
        loss_ewc = 0
        if task_id > 0 and config.ewc_lambda > 0:
            assert not config.train_from_scratch
            loss_ewc += ewc.ewc_regularizer(task_id, mnet.internal_params,
                mnet, online=True, gamma=config.ewc_gamma)

        loss = loss_nll + loss_l2 + config.ewc_lambda * loss_ewc

        loss.backward()
        if config.clip_grad_value != -1:
            torch.nn.utils.clip_grad_value_( \
                optimizer.param_groups[0]['params'], config.clip_grad_value)
        elif config.clip_grad_norm != -1:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],
                                           config.clip_grad_norm)
        optimizer.step()

        ###############################
        ### Learning rate scheduler ###
        ###############################
        # We can invoke the same function to compute test accuracy as we do for
        # BbB.
        pmutils.apply_lr_schedulers(config, shared, logger, task_id, data, mnet,
            None, device, i, iter_per_epoch, plateau_scheduler,
            lambda_scheduler, hhnet=None, method='bbb')

        ###########################
        ### Tensorboard summary ###
        ###########################
        if i % 50 == 0:
            writer.add_scalar('train/task_%d/loss_nll' % task_id, loss_nll, i)
            writer.add_scalar('train/task_%d/loss_l2' % task_id, loss_l2, i)
            writer.add_scalar('train/task_%d/loss_ewc' % task_id, loss_ewc, i)
            writer.add_scalar('train/task_%d/loss' % task_id, loss, i)
            if not is_regression:
                writer.add_scalar('train/task_%d/accuracy' % task_id,
                                  mean_train_acc, i)

    pmutils.checkpoint_bn_stats(config, task_id, mnet)

    #############################
    ### Compute Fisher matrix ###
    #############################
    # Note, we compute the Fisher after all tasks (even the last task) if we
    # have a multihead setup, since we use those Fisher values to build
    # approximate posterior distributions.
    if is_multihead or task_id < config.num_tasks - 1:
        logger.debug('Computing diagonal Fisher elements ...')

        fisher_params = mnet.internal_params

        # When training from scratch, new networks are generated every round
        # such that the old Fisher matrices as expected by EWC are not existing
        # yet.
        # On the other hand, if the hypernetwork is used, then we learn task-
        # specific models and we have to explicitly avoid that Fisher matrices
        # are accumulated.
        if task_id > 0 and config.train_from_scratch:
            for i, p in enumerate(fisher_params):
                buff_w_name, buff_f_name = ewc._ewc_buffer_names(task_id, i,
                                                                 True)
                mnet.register_buffer(buff_w_name, torch.zeros_like(p))
                mnet.register_buffer(buff_f_name, torch.zeros_like(p))

        # Compute prior-offset of Fisher values.
        if is_multihead:
            out_masks = mnet.get_output_weight_mask(out_inds=allowed_outputs,
                                                    device=device)

            prior_offset = [torch.zeros_like(p) for p in mnet.internal_params]

            for ii, mask in enumerate(out_masks):
                pind = mnet.param_shapes_meta[ii]['index']

                if mask is None: # Shared parameter.
                    if task_id == 0 or config.train_from_scratch:
                        prior_offset[pind][:] = 1. / config.prior_variance
                else: # Current output head.
                    # Note, why don't I apply the offset from the beginning to
                    # all heads?
                    # -> If I would, then Fisher values of output heads of
                    # the current and future tasks would be non-zero and
                    # therefore the corresponding weights would be regularized
                    # by the EWC regularizer. For future tasks this doesn't
                    # matter, as the weights don't change during training and
                    # the reg is still 0. But for the current task this does
                    # matter and therefore the reg would pull the weights
                    # towards the random initialization.
                    prior_offset[pind][mask] = 1. / config.prior_variance

        else:
            prior_offset = 0
            if task_id == 0 or config.train_from_scratch:
                prior_offset = 1. / config.prior_variance

        target_manipulator = None
        if not is_regression:
            target_manipulator = lambda T: pmutils.fit_targets_to_softmax( \
                config, shared, device, data, task_id, T)

        ewc.compute_fisher(task_id, data, fisher_params, device, mnet,
            empirical_fisher=True, online=True, gamma=config.ewc_gamma,
            n_max=config.n_fisher, regression=is_regression,
            allowed_outputs=allowed_outputs, custom_forward=None,
            time_series=False, custom_nll=None, pass_ids=False,
            proper_scaling=True, prior_strength=prior_offset,
            regression_lvar=config.ll_dist_std**2 if is_regression else 1.,
            target_manipulator=target_manipulator)

        ### Log histogram of diagonal Fisher elements.
        diag_fisher = []

        out_masks = mnet.get_output_weight_mask(out_inds=allowed_outputs,
                                                device=device)
        for ii, mask in enumerate(out_masks):
            pind = mnet.param_shapes_meta[ii]['index']
            _, buff_f_name = ewc._ewc_buffer_names(None, pind, True)
            curr_F = getattr(mnet, buff_f_name)
            if mask is not None:
                curr_F = curr_F[mask]
            diag_fisher.append(curr_F)

        diag_fisher = torch.cat([p.detach().flatten().cpu() for p in \
                                 diag_fisher])

        writer.add_scalar('ewc/min_fisher', torch.min(diag_fisher), task_id)
        writer.add_scalar('ewc/max_fisher', torch.max(diag_fisher), task_id)
        writer.add_histogram('ewc/fisher', diag_fisher, task_id)
        try:
            writer.add_histogram('ewc/log_fisher', torch.log(diag_fisher),
                                 task_id)
        except:
            # Should not happen, since diagonal elements should be positive.
            logger.warn('Could not write histogram of diagonal fisher ' +
                        'elements.')

    logger.info('Training network on task %d ... Done' % (task_id+1))

def run(config, experiment='regression_ewc'):
    """Run the EWC training for the given experiment.

    Args:
        config (argparse.Namespace): Command-line arguments.
        experiment (str): Which kind of experiment should be performed?

            - ``'regression_ewc'``: Regression tasks with EWC
            - ``'gmm_ewc'``: GMM Data with EWC
            - ``'split_mnist_ewc'``: SplitMNIST with EWC
            - ``'perm_mnist_ewc'``: PermutedMNIST with EWC
            - ``'cifar_resnet_ewc'``: CIFAR-10/100 with EWC
    """
    assert experiment in ['regression_ewc', 'gmm_ewc', 'split_mnist_ewc',
                          'perm_mnist_ewc', 'cifar_resnet_ewc']

    script_start = time()

    device, writer, logger = sutils.setup_environment(config,
        logger_name=experiment)

    rutils.backup_cli_command(config)

    is_classification = True
    if 'regression' in experiment:
        is_classification = False

    ### Create tasks.
    if is_classification:
        dhandlers = pmutils.load_datasets(config, logger, experiment, writer)
    else:
        dhandlers, num_tasks = rutils.generate_tasks(config, writer)
        config.num_tasks = num_tasks

    ### Simple struct, that is used to share data among functions.
    shared = Namespace()
    shared.experiment_type = experiment
    shared.all_dhandlers = dhandlers
    shared.num_trained = 0

    ### Generate network.
    mnet, _, _, _ = pcutils.generate_networks(config, shared, logger, dhandlers,
        device, create_mnet=True, create_hnet=False, create_hhnet=dhandlers,
        create_dis=False)

    if not is_classification:
        shared.during_mse = np.ones(config.num_tasks) * -1.
        # MSE achieved after most recent call of test method.
        shared.current_mse = np.ones(config.num_tasks) * -1.
    # The weights of the main network right after training on that task
    # (can be used to assess how close the final weights are to the original
    # ones).
    shared.during_weights = [-1] * config.num_tasks

    # Where to save network checkpoints?
    shared.ckpt_dir = os.path.join(config.out_dir, 'checkpoints')
    shared.ckpt_mnet_fn = os.path.join(shared.ckpt_dir, 'mnet_task_%d')

    # Initialize the softmax temperature per-task with one. Might be changed
    # later on to calibrate the temperature.
    if is_classification:
        shared.softmax_temp = [torch.ones(1).to(device) \
                               for _ in range(config.num_tasks)]

    ### Initialize summary.
    if is_classification:
        pcutils.setup_summary_dict(config, shared, experiment, mnet)
    else:
        rutils.setup_summary_dict(config, shared, 'ewc', config.num_tasks, mnet)

    # Add hparams to tensorboard, such that the identification of runs is
    # easier.
    writer.add_hparams(hparam_dict={**vars(config), **{
        'num_weights_main': shared.summary['num_weights_main'] \
            if is_classification else shared.summary['aa_num_weights_main']
    }}, metric_dict={})

    if is_classification:
        during_acc_criterion = pmutils.parse_performance_criterion(config,
            shared, logger)

    ### Train on tasks sequentially.
    for i in range(config.num_tasks):
        logger.info('### Training on task %d ###' % (i+1))
        data = dhandlers[i]

        # Train the network.
        shared.num_trained += 1
        train(i, data, mnet, device, config, shared, logger, writer)

        ### Test networks.
        test_ids = None
        if hasattr(config, 'full_test_interval') and \
                config.full_test_interval != -1:
            if i == config.num_tasks-1 or \
                    (i > 0 and i % config.full_test_interval == 0):
                test_ids = None # Test on all tasks.
            else:
                test_ids = [i] # Only test on current task.
        test(dhandlers[:(i+1)], mnet, device, config, shared, logger, writer,
             test_ids=test_ids)

        ### Check if last task got "acceptable" accuracy ###
        if is_classification:
            curr_dur_acc = shared.summary['acc_task_given_during'][i]
        if is_classification and i < config.num_tasks-1 \
                and during_acc_criterion[i] != -1 \
                and during_acc_criterion[i] > curr_dur_acc:
            logger.error('During accuracy of task %d too small (%f < %f).' % \
                         (i+1, curr_dur_acc, during_acc_criterion[i]))
            logger.error('Training of future tasks will be skipped')
            writer.close()
            exit(1)

        if config.train_from_scratch and i < config.num_tasks-1:
            # We have to checkpoint the networks, such that we can reload them
            # for task inference later during testing.
            pmutils.checkpoint_nets(config, shared, i, mnet, None)

            mnet, _, _, _ = pcutils.generate_networks(config, shared, logger,
                dhandlers, device, create_mnet=True, create_hnet=False,
                create_hhnet=dhandlers, create_dis=False)

    if config.store_final_model:
        logger.info('Checkpointing final model ...')
        pmutils.checkpoint_nets(config, shared, config.num_tasks-1, mnet, None)

    ### Plot final classification scores.
    if is_classification:
        logger.info('During accuracies (task identity given): ' + \
                    '%s (avg: %.2f%%).' % \
            (np.array2string(np.array(shared.summary['acc_task_given_during']),
                             precision=2, separator=','),
             shared.summary['acc_avg_task_given_during']))
        logger.info('Final accuracies (task identity given): ' + \
                    '%s (avg: %.2f%%).' % \
            (np.array2string(np.array(shared.summary['acc_task_given']),
                             precision=2, separator=','),
             shared.summary['acc_avg_task_given']))

    if is_classification and config.cl_scenario == 3 and config.split_head_cl3:
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

    ### Plot final regression scores.
    if not is_classification:
        logger.info('During MSE values after training each task: %s' % \
            np.array2string(np.array(shared.summary['aa_mse_during']),
                            precision=5, separator=','))
        logger.info('Final MSE values after training on all tasks: %s' % \
            np.array2string(np.array(shared.summary['aa_mse_final']),
                            precision=5, separator=','))
        logger.info('Final MSE mean %.4f.' % \
                    (shared.summary['aa_mse_during_mean']))

    ### Write final summary.
    shared.summary['finished'] = 1
    if is_classification:
        pmutils.save_summary_dict(config, shared, experiment)
    else:
        rutils.save_summary_dict(config, shared)

    writer.close()

    logger.info('Program finished successfully in %f sec.'
                % (time() - script_start))

if __name__ == '__main__':
    pass


