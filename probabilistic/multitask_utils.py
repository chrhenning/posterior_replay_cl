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
# @title          :probabilistic/multitask_utils.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :09/03/2021
# @version        :1.0
# @python_version :3.8.10
"""
Multitask baseline
------------------

Continual learning algorithms are most commonly designed to train on a sequence
of discrete tasks, each represented by its own dataset.

An upper baseline for those algorithms is multitask training, where training is
performed on all tasks in parallel. Thus, each mini-batch contains data randomly
drawn from all different tasks.

However, there are still different versions of this baseline. For instance,
a well suited-baseline for task-incremental learning would be a multi-head
network trained in a multitask fashion.
"""
from argparse import Namespace
import numpy as np
import os
from time import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from probabilistic import train_vi
from probabilistic.prob_cifar import train_utils as pcutils
from probabilistic.prob_mnist import train_bbb as class_bbb
from probabilistic.prob_mnist import train_utils as pmutils
from probabilistic.regression import train_bbb as reg_bbb
from probabilistic.regression import train_utils as rutils
from utils import torch_utils as tutils
from utils import sim_utils as sutils

def test(data_handlers, mnet, device, config, shared, logger, writer,
         test_ids=None):
    r"""Test the performance on all tasks.

    See docstring of function :func:`probailistic.train_vi.test`.

    Args:
        (....): See docstring of function :func:`probailistic.train_vi.test`.
    """
    is_regression = 'regression' in shared.experiment_type

    if is_regression:
        assert test_ids is None
        reg_bbb.test(data_handlers, mnet, None, device, config, shared, logger,
                     writer)
    else:
        train_vi.test(data_handlers, mnet, None, None, device, config, shared,
                      logger, writer, test_ids=test_ids, method='mt')

def train(data_handlers, mnet, device, config, shared, logger, writer):
    r"""Train a network in a multitask fashion.

    Args:
        data_handlers: List of dataset handlers.
        mnet: The model of the main network.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        shared: Miscellaneous data shared among training functions.
        logger: Command-line logger.
        writer: The tensorboard summary writer.
    """
    logger.info('Training network on all tasks.')

    mnet.train()

    # Whether we train a classification or regression task?
    is_regression = 'regression' in shared.experiment_type

    if is_regression:
        assert config.ll_dist_std > 0
        eval_func = reg_bbb.evaluate
        ll_scale = 1. / config.ll_dist_std**2
    else:
        assert np.all([shared.softmax_temp[i] == 1 \
                       for i in range(config.num_tasks)])
        eval_func = class_bbb.evaluate

    # Which outputs should we consider from the main network for each task.
    allowed_outputs = [pmutils.out_units_of_task(config, data_handlers[i], i,
        config.num_tasks) for i in range(config.num_tasks)]

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
    mnet_kwargs = [pmutils.mnet_kwargs(config, i, mnet) \
                   for i in range(config.num_tasks)]

    num_train_samples = int(np.sum([data_handlers[i].num_train_samples \
                                    for i in range(config.num_tasks)]))
    num_train_iter, iter_per_epoch = sutils.calc_train_iter( \
        num_train_samples, config.batch_size, num_iter=config.n_iter,
        epochs=config.epochs)

    for i in range(num_train_iter):
        #########################
        ### Evaluate networks ###
        #########################
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            for task_id in range(config.num_tasks):
                eval_func(task_id, data_handlers[task_id], mnet, None, device,
                          config, shared, logger, writer, i)
            mnet.train()

        if i % 100 == 0:
            logger.debug('Training iteration: %d.' % i)

        ##########################
        ### Train Current Task ###
        ##########################
        optimizer.zero_grad()

        # Choose number of samples for each task:
        batch_tasks, batch_sizes_tmp = np.unique(np.random.randint(0,
            high=config.num_tasks, size=config.batch_size), return_counts=True)
        batch_sizes_tmp = batch_sizes_tmp.tolist()
        batch_sizes = []
        for ii in range(config.num_tasks):
            if ii in batch_tasks:
                batch_sizes.append(batch_sizes_tmp.pop(0))
            else:
                batch_sizes.append(0)

        loss_nll = 0
        mean_train_acc = 0
        for task_id, data in enumerate(data_handlers):
            if batch_sizes[task_id] == 0:
                continue

            ### Compute negative log-likelihood (NLL).
            batch = data.next_train_batch(batch_sizes[task_id])
            X = data.input_to_torch_tensor(batch[0], device, mode='train')
            T = data.output_to_torch_tensor(batch[1], device, mode='train')

            if not is_regression:
                # Modify 1-hot encodings according to CL scenario.
                assert T.shape[1] == data.num_classes
                # Modify the targets, if softmax spans multiple heads.
                T = pmutils.fit_targets_to_softmax(config, shared, device, data,
                                                   task_id, T)

                _, labels = torch.max(T, 1) # Integer labels.
                labels = labels.detach()

            Y = mnet.forward(X, **mnet_kwargs[task_id])
            if allowed_outputs[task_id] is not None:
                Y = Y[:, allowed_outputs[task_id]]

            # Task-specific loss.
            if is_regression:
                loss_nll += 0.5 * ll_scale * F.mse_loss(Y, T, reduction='sum')
            else:
                # Note, that `cross_entropy` also computed the softmax for us.
                loss_nll += F.cross_entropy(Y, labels, reduction='sum')

                # Compute accuracy on batch.
                # Note, softmax wouldn't change the argmax.
                _, pred_labels = torch.max(Y, 1)
                mean_train_acc += torch.sum(pred_labels == labels).item()

        loss_nll *= num_train_samples / config.batch_size
        if not is_regression:
            mean_train_acc *= 100. / config.batch_size

        loss = loss_nll

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
        # TODO For Plateau Scheduler: Implement validation score.
        pmutils.apply_lr_schedulers(config, shared, logger, None, None, mnet,
            None, device, i, iter_per_epoch, plateau_scheduler,
            lambda_scheduler, hhnet=None, method='bbb')

        ###########################
        ### Tensorboard summary ###
        ###########################
        if i % 50 == 0:
            writer.add_scalar('train/loss', loss, i)
            if not is_regression:
                writer.add_scalar('train/accuracy', mean_train_acc, i)

    logger.info('Training network on all tasks ... Done')

def run(config, experiment='regression_mt'):
    """Run the Multitask training for the given experiment.

    Args:
        config (argparse.Namespace): Command-line arguments.
        experiment (str): Which kind of experiment should be performed?

            - ``'regression_mt'``: Regression tasks with multitask training
            - ``'gmm_mt'``: GMM Data with multitask training
            - ``'split_mnist_mt'``: SplitMNIST with multitask training
            - ``'perm_mnist_mt'``: PermutedMNIST with multitask training
            - ``'cifar_resnet_mt'``: CIFAR-10/100 with multitask training
    """
    assert experiment in ['regression_mt', 'gmm_mt', 'split_mnist_mt',
                          'perm_mnist_mt', 'cifar_resnet_mt']

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
        rutils.setup_summary_dict(config, shared, 'mt', config.num_tasks, mnet)

    # Add hparams to tensorboard, such that the identification of runs is
    # easier.
    writer.add_hparams(hparam_dict={**vars(config), **{
        'num_weights_main': shared.summary['num_weights_main'] \
            if is_classification else shared.summary['aa_num_weights_main']
    }}, metric_dict={})

    ### Train on all tasks.
    logger.info('### Training ###')
    # Note, since we are training on all tasks; all output heads can at all
    # times be considered as trained!
    shared.num_trained = config.num_tasks
    train(dhandlers, mnet, device, config, shared, logger, writer)

    logger.info('### Testing ###')
    test(dhandlers, mnet, device, config, shared, logger, writer, test_ids=None)

    if config.store_final_model:
        logger.info('Checkpointing final model ...')
        pmutils.checkpoint_nets(config, shared, config.num_tasks-1, mnet, None)

    ### Plot final classification scores.
    if is_classification:
        logger.info('Final accuracies (task identity given): ' + \
                    '%s (avg: %.2f%%).' % \
            (np.array2string(np.array(shared.summary['acc_task_given']),
                             precision=2, separator=','),
             shared.summary['acc_avg_task_given']))

    ### Plot final regression scores.
    if not is_classification:
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


