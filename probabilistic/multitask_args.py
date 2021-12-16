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
# @title          :probabilistic/multitask_args.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :09/03/2021
# @version        :1.0
# @python_version :3.8.10
"""
Command-line arguments for training a multitask baseline
--------------------------------------------------------
"""
import argparse
from datetime import datetime
import warnings

from probabilistic.prob_cifar import train_args as pcta
from probabilistic.prob_mnist import train_args as pmta
from probabilistic.regression import train_args as rta
import utils.cli_args as cli


def parse_cmd_arguments(mode='regression_mt', default=False, argv=None):
    """Parse command-line arguments for Multitask experiments.

    Args:
        mode (str): For what script should the parser assemble the set of
            command-line parameters? Options:

                - ``'regression_mt'``
                - ``'gmm_mt'``
                - ``'split_mnist_mt'``
                - ``'perm_mnist_mt'``
                - ``'cifar_resnet_mt'``
        default (bool, optional): If ``True``, command-line arguments will be
            ignored and only the default values will be parsed.
        argv (list, optional): If provided, it will be treated as a list of
            command- line argument that is passed to the parser in place of
            :code:`sys.argv`.

    Returns:
        (argparse.Namespace): The Namespace object containing argument names and
            values.
    """
    if mode == 'regression_mt':
        description = 'Multitask training on toy regression tasks'
    elif mode == 'gmm_mt':
        description = 'Multitask training on GMM tasks'
    elif mode == 'split_mnist_mt':
        description = 'Multitask training on Split MNIST'
    elif mode == 'perm_mnist_mt':
        description = 'Multitask training on Permuted MNIST'
    elif mode == 'cifar_resnet_mt':
        description = 'Multitask training on CIFAR-10/100'
    else:
        raise ValueError('Mode "%s" unknown.' % (mode))

    parser = argparse.ArgumentParser(description=description)

    # If needed, add additional parameters.
    if mode == 'regression_mt':
        dout_dir = './out_mt/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=False,
                    show_from_scratch=False, show_multi_head=True)
        train_agroup = cli.train_args(parser, show_lr=True, dbatch_size=32,
            dn_iter=5001, dlr=1e-3, show_use_adam=True, show_use_rmsprop=True,
            show_use_adadelta=True, show_use_adagrad=True, show_epochs=True,
            show_clip_grad_value=True, show_clip_grad_norm=True)
        cli.main_net_args(parser, allowed_nets=['mlp'], dmlp_arch='10,10',
            dnet_act='sigmoid', show_no_bias=True, show_batchnorm=True,
            show_no_batchnorm=False, show_bn_no_running_stats=True,
            show_bn_distill_stats=False, show_bn_no_stats_checkpointing=False,
            show_specnorm=False, show_dropout_rate=True, ddropout_rate=-1,
            show_net_act=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=False,
                                    dval_batch_size=10000, dval_iter=250)
        rta.data_args(parser)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=False,
            dout_dir=dout_dir, show_publication_style=True)

        pmta.special_train_options(train_agroup, show_soft_targets=False)

        rta.mc_args(train_agroup, eval_agroup, show_train_sample_size=False,
                    show_val_sample_size=False)
        rta.train_args(train_agroup, show_prior_variance=False,
                       show_ll_dist_std=True, show_local_reparam_trick=False,
                       show_kl_scale=False, show_radial_bnn=False)
        rta.miscellaneous_args(misc_agroup, show_mnet_only=False,
            show_use_logvar_enc=False, show_disable_lrt_test=False,
            show_mean_only=False)

    elif mode == 'gmm_mt':
        dout_dir = './out_gmm_mt/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=False,
            show_from_scratch=False, show_multi_head=False,
            show_cl_scenario=True, show_split_head_cl3=True,
            show_num_tasks=False, show_num_classes_per_task=False)
        train_agroup = cli.train_args(parser, show_lr=True, dbatch_size=32,
            dn_iter=2000, dlr=1e-3, show_use_adam=True, show_use_rmsprop=True,
            show_use_adadelta=True, show_use_adagrad=True, show_epochs=True,
            show_clip_grad_value=True, show_clip_grad_norm=True)
        cli.main_net_args(parser, allowed_nets=['mlp'], dmlp_arch='10,10',
            dnet_act='sigmoid', show_no_bias=True, show_batchnorm=True,
            show_no_batchnorm=False, show_bn_no_running_stats=True,
            show_bn_distill_stats=False, show_bn_no_stats_checkpointing=False,
            show_specnorm=False, show_dropout_rate=True, ddropout_rate=-1,
            show_net_act=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                                    dval_batch_size=10000, dval_iter=100)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=False,
            dout_dir=dout_dir)

        pmta.special_train_options(train_agroup, show_soft_targets=False)
        pmta.train_args(train_agroup, show_calc_hnet_reg_targets_online=False,
               show_hnet_reg_batch_size=False, show_init_with_prev_emb=False,
               show_use_prev_post_as_prior=False, show_kl_schedule=False,
               show_num_kl_samples=False, show_training_set_size=True)

        rta.mc_args(train_agroup, eval_agroup, show_train_sample_size=False,
                    show_val_sample_size=False)
        rta.miscellaneous_args(misc_agroup, show_mnet_only=False,
            show_use_logvar_enc=False, show_disable_lrt_test=False,
            show_mean_only=False, show_during_acc_criterion=False)

    elif mode == 'split_mnist_mt':
        dout_dir = './out_split_mt/run_' + \
                   datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=False,
                                 show_from_scratch=False, show_multi_head=False,
                                 show_cl_scenario=True,
                                 show_split_head_cl3=True,
                                 show_num_tasks=True, dnum_tasks=5,
                                 show_num_classes_per_task=True,
                                 dnum_classes_per_task=2)
        train_agroup = cli.train_args(parser, show_lr=True, dbatch_size=128,
                                      dn_iter=2000, dlr=1e-3,
                                      show_use_adam=True, show_use_rmsprop=True,
                                      show_use_adadelta=True,
                                      show_use_adagrad=True, show_epochs=True,
                                      show_clip_grad_value=True,
                                      show_clip_grad_norm=True)
        cli.main_net_args(parser, allowed_nets=['mlp', 'lenet', 'resnet', 'wrn'],
                          dmlp_arch='400,400', dlenet_type='mnist_small',
                          show_no_bias=True, show_batchnorm=True,
                          show_no_batchnorm=False,
                          show_bn_no_running_stats=True,
                          show_bn_distill_stats=False,
                          show_bn_no_stats_checkpointing=False,
                          show_specnorm=False, show_dropout_rate=True,
                          ddropout_rate=-1, show_net_act=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
                                             synthetic_data=False,
                                             show_plots=False, no_cuda=False,
                                             dout_dir=dout_dir)

        pmta.special_train_options(train_agroup, show_soft_targets=False)
        pmta.train_args(train_agroup, show_calc_hnet_reg_targets_online=False,
               show_hnet_reg_batch_size=False, show_init_with_prev_emb=False,
               show_use_prev_post_as_prior=False, show_kl_schedule=False,
               show_num_kl_samples=False, show_training_set_size=True)

        rta.mc_args(train_agroup, eval_agroup, show_train_sample_size=False,
                    show_val_sample_size=False)
        rta.miscellaneous_args(misc_agroup, show_mnet_only=False,
                               show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_mean_only=False,
                               show_during_acc_criterion=False)

    elif mode == 'perm_mnist_mt':
        dout_dir = './out_perm_mt/run_' + \
                   datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=False,
                                 show_from_scratch=False, show_multi_head=False,
                                 show_cl_scenario=True,
                                 show_split_head_cl3=True,
                                 show_num_tasks=True, dnum_tasks=10,
                                 show_num_classes_per_task=False)
        train_agroup = cli.train_args(parser, show_lr=True, dbatch_size=128,
                                      dn_iter=5000, dlr=1e-4,
                                      show_use_adam=True, show_use_rmsprop=True,
                                      show_use_adadelta=True,
                                      show_use_adagrad=True, show_epochs=True,
                                      show_clip_grad_value=True,
                                      show_clip_grad_norm=True)
        cli.main_net_args(parser, allowed_nets=['mlp'], dmlp_arch='1000,1000',
                          show_no_bias=True, show_batchnorm=True,
                          show_no_batchnorm=False,
                          show_bn_no_running_stats=True,
                          show_bn_distill_stats=False,
                          show_bn_no_stats_checkpointing=False,
                          show_specnorm=False, show_dropout_rate=True,
                          ddropout_rate=-1, show_net_act=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        pmta.special_train_options(train_agroup, show_soft_targets=False)
        pmta.train_args(train_agroup, show_calc_hnet_reg_targets_online=False,
               show_hnet_reg_batch_size=False, show_init_with_prev_emb=False,
               show_use_prev_post_as_prior=False, show_kl_schedule=False,
               show_num_kl_samples=False, show_training_set_size=True)

        rta.mc_args(train_agroup, eval_agroup, show_train_sample_size=False,
                    show_val_sample_size=False)
        rta.miscellaneous_args(misc_agroup, show_mnet_only=False,
                               show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_mean_only=False,
                               show_during_acc_criterion=False)

        pmta.perm_args(parser)

    elif mode == 'cifar_resnet_mt':
        dout_dir = './out_resnet_mt/run_' + \
                   datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=False,
                                 show_from_scratch=False, show_multi_head=False,
                                 show_cl_scenario=True,
                                 show_split_head_cl3=True,
                                 show_num_tasks=True, dnum_tasks=6,
                                 show_num_classes_per_task=True,
                                 dnum_classes_per_task=10)
        pcta.extra_cl_args(cl_argroup)
        train_agroup = cli.train_args(parser, show_lr=True, dbatch_size=32,
                                      dn_iter=2000, dlr=1e-3,
                                      show_use_adam=True, show_use_rmsprop=True,
                                      show_use_adadelta=True,
                                      show_use_adagrad=True, show_epochs=True,
                                      depochs=200, show_clip_grad_value=True,
                                      show_clip_grad_norm=True)
        cli.main_net_args(parser,
            allowed_nets=['resnet', 'wrn', 'iresnet', 'lenet', 'zenke', 'mlp'],
            dmlp_arch='10,10', dlenet_type='cifar', show_no_bias=True,
            show_batchnorm=False, show_no_batchnorm=True,
            show_bn_no_running_stats=True, show_bn_distill_stats=False,
            show_bn_no_stats_checkpointing=False,
            show_specnorm=False, show_dropout_rate=True,
            ddropout_rate=-1, show_net_act=True)
        cli.data_args(parser, show_disable_data_augmentation=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                                    dval_batch_size=1000,
                                    show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
                                             synthetic_data=False,
                                             show_plots=False, no_cuda=False,
                                             dout_dir=dout_dir)

        pmta.special_train_options(train_agroup, show_soft_targets=False)
        pmta.train_args(train_agroup, show_calc_hnet_reg_targets_online=False,
               show_hnet_reg_batch_size=False, show_init_with_prev_emb=False,
               show_use_prev_post_as_prior=False, show_kl_schedule=False,
               show_num_kl_samples=False, show_training_set_size=True)

        rta.mc_args(train_agroup, eval_agroup, show_train_sample_size=False,
                    show_val_sample_size=False)
        rta.miscellaneous_args(misc_agroup, show_mnet_only=False,
                               show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_mean_only=False,
                               show_during_acc_criterion=False)

    args = None
    if argv is not None:
        if default:
            warnings.warn('Provided "argv" will be ignored since "default" ' +
                          'option was turned on.')
        args = argv
    if default:
        args = []
    config = parser.parse_args(args=args)

    ### Add constant arguments.
    # Training from scratch doesn't apply to multitask learning. But many
    # functions assume the argument exists.
    config.train_from_scratch = False
    # Stats checkpointing doesn't make sense here.
    config.bn_no_stats_checkpointing = True
    config.mnet_only = True

    ### Check argument values!
    cli.check_invalid_argument_usage(config)
    rta.check_invalid_args_general(config)
    pmta.check_invalid_args_general(config)
    check_invalid_args_mt(config)

    if mode == 'regression_mt':
        if config.batchnorm:
            # Not properly handled in test and eval function!
            raise NotImplementedError()

    return config

def check_invalid_args_mt(config):
    """Sanity check for some multitask command-line arguments.

    Args:
        config: Parsed command-line arguments.
    """
    if config.plateau_lr_scheduler:
        raise NotImplementedError('Plateau-Scheduler not implemented yet for ' +
                                  'multitask training.')

    #if config.cl_scenario == 3 and config.split_head_cl3:
    #    warn('Split-head CL3 does not make sense for multitask learning.')

if __name__ == '__main__':
    pass


