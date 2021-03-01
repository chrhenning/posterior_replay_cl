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
# @title          :probabilistic/prob_cifar/train_avb.py
# @author         :ch, mc
# @contact        :henningc@ethz.ch
# @created        :01/30/2020
# @version        :1.0
# @python_version :3.6.9
"""
Using AVB for Prior-Focused and Posterior-Replay Continual Learning
-------------------------------------------------------------------

The module :mod:`probabilistic.prob_cifar.train_avb` trains a probabilistic
classifier in a continual learning setting using AVB. I.e., given a sequence of 
tasks, the goal is to obtain a Bayesian Neural Network that performs well and 
provides meaningful predictive uncertainties on all tasks (see also module
:mod:`probabilistic.prob_mnist.train_bbb`).

Specifically, this module uses the algorithm AVB proposed in

    Mescheder et al., "Adversarial Variational Bayes: Unifying Variational
    Autoencoders and Generative Adversarial Networks", 2018.
    https://arxiv.org/abs/1701.04722

to learn an implicit weight posterior, that is realized through a hypernetwork.
When using prior-focused CL, the implicit distribution (hypernetwork) from the
previous task is used as prior.

When learning a posterior per task, the implicit distribution in the
hypernetwork is protected via a hyper-hypernetwork.
"""
from argparse import Namespace
import numpy as np
import os
from time import time
import torch

from probabilistic.prob_cifar import train_utils as pcutils
from probabilistic.prob_mnist import train_bbb
from probabilistic.prob_mnist import train_utils as pmutils
from probabilistic.regression import train_utils as rutils
from probabilistic import train_vi as tvi
from utils import sim_utils as sutils

def run(config, experiment='split_mnist_avb'):
    """Run the training.

    Args:
        config (argparse.Namespace): Command-line arguments.
        experiment (str): Which kind of experiment should be performed?

            - "gmm_avb": Synthetic GMM data with Posterior Replay via AVB
            - "gmm_avb_pf": Synthetic GMM data with Prior-Focused CL via AVB
            - "split_mnist_avb": Split MNIST with Posterior Replay via AVB
            - "perm_mnist_avb": Permuted MNIST with Posterior Replay via AVB
            - "split_mnist_avb_pf": Split MNIST with Prior-Focused CL via AVB
            - "perm_mnist_avb_pf": Permuted MNIST with Prior-Focused CL via AVB
            - "cifar_zenke_avb": CIFAR-10/100 with Posterior Replay using a
              ZenkeNet and AVB
            - "cifar_resnet_avb": CIFAR-10/100 with Posterior Replay using a
              Resnet and AVB
            - "cifar_zenke_avb_pf": CIFAR-10/100 with Prior-Focused CL using a
              ZenkeNet and AVB
            - "cifar_resnet_avb_pf": CIFAR-10/100 with Prior-Focused CL using a
              Resnet and AVB
            - "gmm_ssge": Synthetic GMM data with Posterior Replay via SSGE
            - "gmm_ssge_pf": Synthetic GMM data with Prior-Focused CL via SSGE
            - "split_mnist_ssge": Split MNIST with Posterior Replay via SSGE
            - "perm_mnist_ssge": Permuted MNIST with Posterior Replay via SSGE
            - "split_mnist_ssge_pf": Split MNIST with Prior-Focused CL via SSGE
            - "perm_mnist_ssge_pf": Permuted MNIST with Prior-Focused CL via
              SSGE
            - "cifar_resnet_ssge": CIFAR-10/100 with Posterior Replay using a
              Resnet and SSGE
            - "cifar_resnet_ssge_pf": CIFAR-10/100 with Prior-Focused CL using a
              Resnet and SSGE
    """
    assert experiment in ['gmm_avb', 'gmm_avb_pf',
                          'split_mnist_avb', 'split_mnist_avb_pf',
                          'perm_mnist_avb', 'perm_mnist_avb_pf',
                          'cifar_zenke_avb', 'cifar_zenke_avb_pf',
                          'cifar_resnet_avb', 'cifar_resnet_avb_pf',
                          'gmm_ssge', 'gmm_ssge_pf',
                          'split_mnist_ssge', 'split_mnist_ssge_pf',
                          'perm_mnist_ssge', 'perm_mnist_ssge_pf',
                          'cifar_resnet_ssge', 'cifar_resnet_ssge_pf']

    script_start = time()

    if 'avb' in experiment:
        method = 'avb'
        use_dis = True # whether a discriminator network is used
    elif 'ssge' in experiment:
        method = 'ssge'
        use_dis = False

    device, writer, logger = sutils.setup_environment(config,
        logger_name=experiment + 'logger')

    rutils.backup_cli_command(config)

    if experiment.endswith('pf'):
        prior_focused_cl = True
        logger.info('Running a prior-focused CL experiment ...')
    else:
        prior_focused_cl = False
        logger.info('Learning task-specific posteriors sequentially ...')

    ### Create tasks.
    dhandlers = pmutils.load_datasets(config, logger, experiment, writer)

    ### Simple struct, that is used to share data among functions.
    shared = Namespace()
    shared.experiment_type = experiment
    shared.all_dhandlers = dhandlers
    shared.prior_focused = prior_focused_cl

    ### Generate networks.
    mnet, hnet, hhnet, dis = pcutils.generate_networks(config, shared, logger,
                                                       dhandlers, device,
                                                       create_dis=use_dis)
    if method == 'ssge':
        assert dis is None

    ### Add more information to shared.
    # Mean and variance of prior that is used for variational inference.
    # For a prior-focused training, this prior will only be used for the
    # first task.
    #plogvar = np.log(config.prior_variance)
    pstd = np.sqrt(config.prior_variance)
    shared.prior_mean = [torch.zeros(*s).to(device) \
                         for s in mnet.param_shapes]
    #shared.prior_logvar = [plogvar * torch.ones(*s).to(device) \
    #                       for s in mnet.param_shapes]
    shared.prior_std = [pstd * torch.ones(*s).to(device) \
                        for s in mnet.param_shapes]

    # The output weights of the hyper-hyper network right after training on
    # a task (can be used to assess how close the final weights are to the
    # original ones).
    shared.during_weights = [-1] * config.num_tasks if hhnet is not None \
        else None

    # Where to save network checkpoints?
    shared.ckpt_dir = os.path.join(config.out_dir, 'checkpoints')
    # Note, some networks have stuff to store such as batch statistics for
    # batch norm. So it is wise to always checkpoint all networks, even if they
    # where constructed without weights.
    shared.ckpt_mnet_fn = os.path.join(shared.ckpt_dir, 'mnet_task_%d')
    shared.ckpt_hnet_fn = os.path.join(shared.ckpt_dir, 'hnet_task_%d')
    shared.ckpt_hhnet_fn = os.path.join(shared.ckpt_dir, 'hhnet_task_%d')
    #shared.ckpt_dis_fn = os.path.join(shared.ckpt_dir, 'dis_task_%d')

    # Initialize the softmax temperature per-task with one. Might be changed
    # later on to calibrate the temperature.
    shared.softmax_temp = [torch.ones(1).to(device) \
                           for _ in range(config.num_tasks)]
    shared.num_trained = 0

    # Setup coresets iff regularization on all tasks is allowed.
    if config.coreset_size != -1 and config.past_and_future_coresets:
        for i in range(config.num_tasks):
            pmutils.update_coreset(config, shared, i, dhandlers[i], None,
                None, device, logger, None, hhnet=None, method='avb')

    ### Initialize summary.
    pcutils.setup_summary_dict(config, shared, experiment, mnet, hnet=hnet,
                               hhnet=hhnet, dis=dis)
    logger.info('Ratio num hnet weights / num mnet weights: %f.'
                % shared.summary['num_weights_hm_ratio'])
    if 'num_weights_hhm_ratio' in shared.summary.keys():
        logger.info('Ratio num hyper-hnet weights / num mnet weights: %f.'
                    % shared.summary['num_weights_hhm_ratio'])
    if method == 'avb':
        logger.info('Ratio num dis weights / num mnet weights: %f.'
                    % shared.summary['num_weights_dm_ratio'])

    # Add hparams to tensorboard, such that the identification of runs is
    # easier.
    hparams_extra_dict = {
        'num_weights_hm_ratio': shared.summary['num_weights_hm_ratio'],
    }
    if 'num_weights_dm_ratio' in shared.summary.keys():
        hparams_extra_dict = {**hparams_extra_dict,
            **{'num_weights_dm_ratio': \
               shared.summary['num_weights_dm_ratio']}}
    if 'num_weights_hhm_ratio' in shared.summary.keys():
        hparams_extra_dict = {**hparams_extra_dict,
            **{'num_weights_hhm_ratio': \
               shared.summary['num_weights_hhm_ratio']}}
    writer.add_hparams(hparam_dict={**vars(config), **hparams_extra_dict},
                       metric_dict={})

    during_acc_criterion = pmutils.parse_performance_criterion(config, shared,
                                                               logger)

    ### Train on tasks sequentially.
    for i in range(config.num_tasks):
        logger.info('### Training on task %d ###' % (i+1))
        data = dhandlers[i]

        # Train the network.
        shared.num_trained += 1
        if config.distill_iter == -1:
            tvi.train(i, data, mnet, hnet, hhnet, dis, device, config, shared,
                  logger, writer, method=method)
        else:
            assert hhnet is not None
            # Train main network only.
            tvi.train(i, data, mnet, hnet, None, dis, device, config, shared,
                      logger, writer, method=method)

            # Distill `hnet` into `hhnet`.
            train_bbb.distill_net(i, data, mnet, hnet, hhnet, device, config,
                                  shared, logger, writer)

            # Create a new main network before training the next task.
            mnet, hnet, _, _ = pcutils.generate_networks(config, shared, logger,
                dhandlers, device, create_dis=False, create_hhnet=False)

        ### Temperature Calibration.
        if config.calibrate_temp:
            pcutils.calibrate_temperature(i, data, mnet, hnet, hhnet, device,
                                          config, shared, logger, writer)

        ### Test networks.
        test_ids = None
        if config.full_test_interval != -1:
            if i == config.num_tasks-1 or \
                    (i > 0 and i % config.full_test_interval == 0):
                test_ids = None # Test on all tasks.
            else:
                test_ids = [i] # Only test on current task.
        tvi.test(dhandlers[:(i+1)], mnet, hnet, hhnet, device, config, shared,
             logger, writer, test_ids=test_ids, method=method)

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
            # Note, we only need the discriminator as helper for training,
            # so we don't checkpoint it.
            pmutils.checkpoint_nets(config, shared, i, mnet, hnet, hhnet=hhnet,
                                    dis=None)

            mnet, hnet, hhnet, dis = pcutils.generate_networks(config, shared,
                logger, dhandlers, device, create_dis=use_dis)

        elif dis is not None and not config.no_dis_reinit and \
                i < config.num_tasks-1:
            logger.debug('Reinitializing discriminator network ...')
            # FIXME Build a new network as this init doesn't effect batchnorm
            # weights atm.
            dis.custom_init(normal_init=config.normal_init,
                            normal_std=config.std_normal_init, zero_bias=True)

    if config.store_final_model:
        logger.info('Checkpointing final model ...')
        pmutils.checkpoint_nets(config, shared, config.num_tasks-1, mnet, hnet,
                                hhnet=hhnet, dis=None)

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
    pmutils.save_summary_dict(config, shared, experiment)

    writer.close()

    logger.info('Program finished successfully in %f sec.'
                % (time() - script_start))

if __name__ == '__main__':
    # Consult README file!
    raise Exception('Script is not executable!')

