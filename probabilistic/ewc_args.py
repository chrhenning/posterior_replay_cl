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
# @title          :probabilistic/ewc_args.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/10/2021
# @version        :1.0
# @python_version :3.8.5
"""
Command-line arguments for training with EWC
--------------------------------------------
"""
import argparse
from datetime import datetime
import warnings

from probabilistic.prob_mnist import train_args as pmta
from probabilistic.regression import train_args as rta
import utils.cli_args as cli

def parse_cmd_arguments(mode='regression_ewc', default=False, argv=None):
    """Parse command-line arguments for EWC experiments.

    Args:
        mode (str): For what script should the parser assemble the set of
            command-line parameters? Options:

                - ``'regression_ewc'``
                - ``'gmm_ewc'``
                - ``'split_mnist_ewc'``
                - ``'perm_mnist_ewc'``
                - ``'cifar_resnet_ewc'``
        default (bool, optional): If ``True``, command-line arguments will be
            ignored and only the default values will be parsed.
        argv (list, optional): If provided, it will be treated as a list of
            command- line argument that is passed to the parser in place of
            :code:`sys.argv`.

    Returns:
        (argparse.Namespace): The Namespace object containing argument names and
            values.
    """
    if mode == 'regression_ewc':
        description = 'Toy regression with tasks trained via EWC'
    elif mode == 'gmm_ewc':
        description = 'Probabilistic CL on GMM Datasets via EWC'
    elif mode == 'split_mnist_ewc':
        description = 'Probabilistic CL on Split MNIST via EWC'
    elif mode == 'perm_mnist_ewc':
        description = 'Probabilistic CL on Permuted MNIST via EWC'
    elif mode == 'cifar_resnet_ewc':
        description = 'Probabilistic CL on CIFAR-10/100 via EWC on a Resnet'
    else:
        raise ValueError('Mode "%s" unknown.' % (mode))

    parser = argparse.ArgumentParser(description=description)

    # If needed, add additional parameters.
    if mode == 'regression_ewc':
        dout_dir = './out_ewc/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=False,
                    show_from_scratch=True, show_multi_head=True)
        train_agroup = cli.train_args(parser, show_lr=True, dbatch_size=32,
            dn_iter=5001, dlr=1e-3, show_use_adam=True, show_use_rmsprop=True,
            show_use_adadelta=True, show_use_adagrad=True, show_epochs=True,
            show_clip_grad_value=True, show_clip_grad_norm=True)
        cli.main_net_args(parser, allowed_nets=['mlp'], dmlp_arch='10,10',
            dnet_act='sigmoid', show_no_bias=True, show_batchnorm=True,
            show_no_batchnorm=False, show_bn_no_running_stats=True,
            show_bn_distill_stats=False, show_bn_no_stats_checkpointing=True,
            show_specnorm=False, show_dropout_rate=True, ddropout_rate=-1,
            show_net_act=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=False,
                                    dval_batch_size=10000, dval_iter=250)
        rta.data_args(parser)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=False,
            dout_dir=dout_dir, show_publication_style=True)
        ewc_args(parser, dewc_lambda=1., dn_fisher=-1)

        pmta.special_train_options(train_agroup, show_soft_targets=False)

        rta.mc_args(train_agroup, eval_agroup, show_train_sample_size=False,
                    dval_sample_size=10)
        rta.train_args(train_agroup, show_ll_dist_std=True,
                       show_local_reparam_trick=False, show_kl_scale=False,
                       show_radial_bnn=False)
        rta.miscellaneous_args(misc_agroup, show_mnet_only=False,
            show_use_logvar_enc=False, show_disable_lrt_test=False,
            show_mean_only=False)

    elif mode == 'gmm_ewc':
        dout_dir = './out_gmm_ewc/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=False,
            show_from_scratch=True, show_multi_head=False,
            show_cl_scenario=True, show_split_head_cl3=True,
            show_num_tasks=False, show_num_classes_per_task=False)
        train_agroup = cli.train_args(parser, show_lr=True, dbatch_size=32,
            dn_iter=2000, dlr=1e-3, show_use_adam=True, show_use_rmsprop=True,
            show_use_adadelta=True, show_use_adagrad=True, show_epochs=True,
            show_clip_grad_value=True, show_clip_grad_norm=True)
        cli.main_net_args(parser, allowed_nets=['mlp'], dmlp_arch='10,10',
            dnet_act='sigmoid', show_no_bias=True, show_batchnorm=True,
            show_no_batchnorm=False, show_bn_no_running_stats=True,
            show_bn_distill_stats=False, show_bn_no_stats_checkpointing=True,
            show_specnorm=False, show_dropout_rate=True, ddropout_rate=-1,
            show_net_act=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                                    dval_batch_size=10000, dval_iter=100)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=False,
            dout_dir=dout_dir)
        ewc_args(parser, dewc_lambda=1., dn_fisher=-1)

        pmta.special_train_options(train_agroup, show_soft_targets=False)

        cl_args(cl_argroup)
        pmta.eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup, show_train_sample_size=False,
                    dval_sample_size=10)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=False,
                       show_radial_bnn=False)
        rta.miscellaneous_args(misc_agroup, show_mnet_only=False,
            show_use_logvar_enc=False, show_disable_lrt_test=False,
            show_mean_only=False, show_during_acc_criterion=True)

    elif mode == 'split_mnist_ewc':
        dout_dir = './out_split_ewc/run_' + \
                   datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=False,
                                 show_from_scratch=True, show_multi_head=False,
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
                          show_bn_no_stats_checkpointing=True,
                          show_specnorm=False, show_dropout_rate=True,
                          ddropout_rate=-1, show_net_act=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
                                             synthetic_data=False,
                                             show_plots=False, no_cuda=False,
                                             dout_dir=dout_dir)
        ewc_args(parser, dewc_lambda=1., dn_fisher=-1)

        pmta.special_train_options(train_agroup, show_soft_targets=False)

        cl_args(cl_argroup)
        pmta.eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup, show_train_sample_size=False,
                    dval_sample_size=10)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=False,
                       show_radial_bnn=False)
        rta.miscellaneous_args(misc_agroup, show_mnet_only=False,
                               show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_mean_only=False,
                               show_during_acc_criterion=True)

    elif mode == 'perm_mnist_ewc':
        dout_dir = './out_perm_ewc/run_' + \
                   datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=False,
                                 show_from_scratch=True, show_multi_head=False,
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
                          show_bn_no_stats_checkpointing=True,
                          show_specnorm=False, show_dropout_rate=True,
                          ddropout_rate=-1, show_net_act=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)
        ewc_args(parser, dewc_lambda=1., dn_fisher=-1)

        pmta.special_train_options(train_agroup, show_soft_targets=False)

        cl_args(cl_argroup)
        pmta.eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup, show_train_sample_size=False,
                    dval_sample_size=10)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=False,
                       show_radial_bnn=False)
        rta.miscellaneous_args(misc_agroup, show_mnet_only=False,
                               show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_mean_only=False,
                               show_during_acc_criterion=True)

        pmta.perm_args(parser)

    elif mode == 'cifar_resnet_ewc':
        dout_dir = './out_resnet_ewc/run_' + \
                   datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=False,
                                 show_from_scratch=True, show_multi_head=False,
                                 show_cl_scenario=True,
                                 show_split_head_cl3=True,
                                 show_num_tasks=True, dnum_tasks=6,
                                 show_num_classes_per_task=True,
                                 dnum_classes_per_task=10)
        train_agroup = cli.train_args(parser, show_lr=True, dbatch_size=32,
                                      dn_iter=2000, dlr=1e-3,
                                      show_use_adam=True, show_use_rmsprop=True,
                                      show_use_adadelta=True,
                                      show_use_adagrad=True, show_epochs=True,
                                      depochs=200, show_clip_grad_value=True,
                                      show_clip_grad_norm=True)
        cli.main_net_args(parser,
            allowed_nets=['resnet', 'wrn', 'lenet', 'zenke', 'mlp'],
            dmlp_arch='10,10', dlenet_type='cifar', show_no_bias=True,
            show_batchnorm=False, show_no_batchnorm=True,
            show_bn_no_running_stats=True, show_bn_distill_stats=False,
            show_bn_no_stats_checkpointing=True,
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
        ewc_args(parser, dewc_lambda=1., dn_fisher=-1)

        pmta.special_train_options(train_agroup, show_soft_targets=False)

        cl_args(cl_argroup)
        pmta.eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup, show_train_sample_size=False,
                    dval_sample_size=10)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=False,
                       show_radial_bnn=False)
        rta.miscellaneous_args(misc_agroup, show_mnet_only=False,
                               show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_mean_only=False,
                               show_during_acc_criterion=True)

    args = None
    if argv is not None:
        if default:
            warnings.warn('Provided "argv" will be ignored since "default" ' +
                          'option was turned on.')
        args = argv
    if default:
        args = []
    config = parser.parse_args(args=args)

    ### Check argument values!
    cli.check_invalid_argument_usage(config)
    rta.check_invalid_args_general(config)
    pmta.check_invalid_args_general(config)
    check_invalid_args_ewc(config)

    if mode == 'regression_ewc':
        if config.batchnorm:
            # Not properly handled in test and eval function!
            raise NotImplementedError()

    return config

def ewc_args(parser, dewc_lambda=1., dn_fisher=-1):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    an argument group for options regarding EWC.

    Arguments specified in this function:
        - `ewc_gamma`
        - `ewc_lambda`
        - `n_fisher`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        dewc_lambda (float): Default value of option `ewc_lambda`.
        dn_fisher (int): Default value of option `n_fisher`.

    Returns:
        The created argument group, in case more options should be added.
    """
    agroup = parser.add_argument_group('EWC options')

    agroup.add_argument('--ewc_gamma', type=float, default=1.,
                         help='Graceful forgetting. Values smaller than 1 ' +
                              'introduce exponential decay in Fisher matrix ' +
                              'accumulation. Default: %(default)s')
    agroup.add_argument('--ewc_lambda', type=float, default=dewc_lambda,
                         help='Regularization strength. ' +
                              'Default: %(default)s.')
    agroup.add_argument('--n_fisher', type=int, default=dn_fisher,
                        help='Number of training samples to be used for the ' +
                             'estimation of the diagonal Fisher elements. If ' +
                             '"-1", all training samples are used. ' +
                             'Default: %(default)s.')
    return agroup

def cl_args(clgroup):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the continual learning argument group for options specific
    to continual learning with EWC.

    Args:
        clgroup: The argument group returned by method
            :func:`utils.cli_args.cl_args`.
    """
    clgroup.add_argument('--non_growing_sf_cl3', action='store_true',
                         help='Applies only to CL3. Rather than learning ' +
                              'a growing softmax, the terminal size of the ' +
                              'softmax is used from the beginning. ' +
                              'Disatvantage: number of tasks cannot ' +
                              'flexibly be increased.')

def check_invalid_args_ewc(config):
    """Sanity check for some EWC command-line arguments.

    Args:
        config: Parsed command-line arguments.
    """
    if config.train_from_scratch and config.ewc_lambda > 0:
        raise ValueError('CL regularizer has to be turned off when training ' +
                         'from scratch.')

    if hasattr(config, 'multi_head') and not config.multi_head and \
            config.val_sample_size > 1:
        warnings.warn('Setting "val_sample_size" to 1.')
        config.val_sample_size = 1
    if hasattr(config, 'cl_scenario') and not (config.cl_scenario == 1 or \
            config.cl_scenario == 3 and config.split_head_cl3) and \
            config.val_sample_size > 1:
        warnings.warn('Setting "val_sample_size" to 1.')
        config.val_sample_size = 1

if __name__ == '__main__':
    pass


