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
# @title           :probabilistic/regression/train_args.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :08/25/2019
# @version         :1.0
# @python_version  :3.6.8
"""
Command-line arguments for probabilistic toy example
----------------------------------------------------
"""
import argparse
from datetime import datetime
import warnings

from probabilistic.prob_mnist import train_args as pmta
from probabilistic.prob_cifar import train_args as pcta
import utils.cli_args as cli

def parse_cmd_arguments(mode='regression_bbb', default=False,
                        argv=None):
    """Parse command-line arguments.

    Args:
        mode: For what script should the parser assemble the set of command-line
            parameters? Options:

                - ``'regression_bbb'``
                - ``'regression_avb'``
                - ``'regression_ssge'``

        default (optional): If True, command-line arguments will be ignored and
            only the default values will be parsed.
        argv (optional): If provided, it will be treated as a list of command-
            line argument that is passed to the parser in place of sys.argv.

    Returns:
        The Namespace object containing argument names and values.
    """
    if mode == 'regression_bbb':
        description = 'Toy regression with tasks trained by BbB and ' + \
            'protected by a hypernetwork'
    elif mode == 'regression_avb' :
        description = 'Toy regression with tasks trained by implicit model ' \
                      'using AVB and protected by a hypernetwork'
    elif mode == 'regression_ssge' :
        description = 'Toy regression with tasks trained by implicit model ' \
                      'using SSGE and protected by a hypernetwork'
    else:
        raise ValueError('Mode "%s" unknown.' % (mode))

    parser = argparse.ArgumentParser(description=description)

    # Default hnet keyword arguments.
    hnet_args_kw = { # Function `cli.hnet_args`
        # Note, the first list element denotes the default hnet.
        'allowed_nets': ['hmlp', 'chunked_hmlp', 'hdeconv', 'chunked_hdeconv'],
        'dhmlp_arch': '10,10',
        'show_cond_emb_size': True,
        'dcond_emb_size': 2,
        'dchmlp_chunk_size': 64,
        'dchunk_emb_size': 8,
        'show_use_cond_chunk_embs': True,
        'show_net_act': True,
        'dnet_act': 'sigmoid',
        'show_no_bias': True,
        'show_dropout_rate': True,
        'ddropout_rate': -1,
        'show_specnorm': True,
        'show_batchnorm': False,
        'show_no_batchnorm': False
    }

    # If needed, add additional parameters.
    if mode == 'regression_bbb':
        dout_dir = './out_bbb/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=True, dbeta=0.005,
                    show_from_scratch=True, show_multi_head=True)
        train_argroup = cli.train_args(parser, show_lr=True, dn_iter=10001,
            dlr=1e-2, show_clip_grad_value=True, show_clip_grad_norm=True,
            show_momentum=False, show_adam_beta1=True)
        cli.main_net_args(parser, allowed_nets=['mlp'], dmlp_arch='10,10',
                          dnet_act='sigmoid', show_no_bias=True)
        cli.hnet_args(parser, **hnet_args_kw)
        init_agroup = cli.init_args(parser, custom_option=False)
        eval_agroup = cli.eval_args(parser, dval_iter=250)
        data_args(parser)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=True,
            show_publication_style=True, dout_dir=dout_dir)

        mc_args(train_argroup, eval_agroup)
        train_args(train_argroup, show_local_reparam_trick=True,
            show_radial_bnn=True)
        cl_args(cl_argroup)
        init_args(init_agroup)
        miscellaneous_args(misc_agroup, show_use_logvar_enc=True,
                           show_disable_lrt_test=True, show_mean_only=True)
        pmta.train_args(train_argroup, show_init_with_prev_emb=False,
            show_use_prev_post_as_prior=True, show_num_kl_samples=True,
            show_training_set_size=False)

    #rtr config parameters for implicit model
    else:
        method = 'avb'
        if mode == 'regression_ssge':
            method = 'ssge'
        dout_dir = './out_%s/run_' % method + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=True, dbeta=0.005,
                                 show_from_scratch=True, show_multi_head=True)
        train_argroup = cli.train_args(parser, show_lr=True, dbatch_size=32,
            dn_iter=5001, dlr=1e-3, show_use_adam=True, show_use_rmsprop=True,
            show_use_adadelta=True, show_use_adagrad=True, show_epochs=True,
            show_clip_grad_value=True, show_clip_grad_norm=True)
        # Main network.
        cli.main_net_args(parser, allowed_nets=['mlp'], dmlp_arch='10,10',
                          dnet_act='sigmoid', show_no_bias=True)
        if mode == 'regression_avb':
            # Discriminator.
            cli.main_net_args(parser, allowed_nets=['mlp', 'chunked_mlp'],
                dmlp_arch='10,10', dcmlp_arch='10,10',
                dcmlp_chunk_arch='10,10', dcmlp_in_cdim=32,
                dcmlp_out_cdim=8, dcmlp_cemb_dim=8, dnet_act='sigmoid',
                show_no_bias=True, prefix='dis_', pf_name='discriminator')
        # Hypernetwork (weight generator).
        imp_hargs_kw = dict(hnet_args_kw)
        imp_hargs_kw['show_cond_emb_size'] = False
        imp_hargs_kw['show_use_cond_chunk_embs'] = False
        imp_hargs_kw['dcond_emb_size'] = 0 # Not used for implicit hnet!
        cli.hnet_args(parser, **imp_hargs_kw, prefix='imp_', pf_name='implicit')
        # Hyper-hypernetwork.
        hhargs_kw = dict(hnet_args_kw)
        cli.hnet_args(parser, **hhargs_kw, prefix='hh_', pf_name='hyper-hyper')
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, dval_iter=250)
        data_args(parser)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=True,
            show_publication_style=True, dout_dir=dout_dir)

        pmta.special_train_options(train_argroup, show_soft_targets=False)
        mc_args(train_argroup, eval_agroup)
        train_args(train_argroup, show_local_reparam_trick=False,
                   show_kl_scale=True)
        miscellaneous_args(misc_agroup, show_store_during_models=True)

        pmta.train_args(train_argroup, show_init_with_prev_emb=True,
            show_use_prev_post_as_prior=True, show_num_kl_samples=True,
            show_calc_hnet_reg_targets_online=True,
            show_hnet_reg_batch_size=True, show_training_set_size=False)
        pmta.ind_posterior_args(train_argroup, show_distill_iter=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=True)
        pcta.imp_args(parser, dlatent_dim=8, show_prior_focused=True)

        if mode == 'regression_avb':
            pcta.avb_args(parser)
        elif mode == 'regression_ssge':
            pcta.ssge_args(parser)

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
    check_invalid_args_general(config)
    pmta.check_invalid_args_general(config)

    if mode == 'regression_bbb':
        check_invalid_bbb_args(config)
    else:
        pass

    return config

def data_args(parser):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add an argument group for options specific to the dataset generation.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.`.

    Returns:
        The generated argument group.
    """
    ### Data options.
    dgroup = parser.add_argument_group('Dataset options')
    dgroup.add_argument('--used_task_set', type=int, default=1,
                        help='The set of tasks to be used. ' +
                             'Default: %(default)s.')
    return dgroup

def mc_args(tgroup, vgroup, show_train_sample_size=True,
            show_val_sample_size=True, dval_sample_size=10):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the training and validation argument group for options
    specific to the Monte-Carlos sampling procedure used to approximate the loss
    and the predictive distribution.

    Args:
        tgroup: The argument group returned by method
            :func:`utils.cli_args.train_args`.
        vgroup: The argument group returned by method
            :func:`utils.cli_args.eval_args`.
        show_train_sample_size (bool): Whether option `train_sample_size`
            should be shown.
        show_val_sample_size (bool): Whether option `val_sample_size`
            should be shown.
        dval_sample_size (int): Default value of option `val_sample_size`.
    """
    if show_train_sample_size:
        tgroup.add_argument('--train_sample_size', type=int, metavar='N',
                            default=10,
                            help='How many samples should be used for the ' +
                                 'approximation of the negative log ' +
                                 'likelihood in the loss. ' +
                                 'Default: %(default)s.')
    if show_val_sample_size:
        vgroup.add_argument('--val_sample_size', type=int, metavar='N',
                            default=dval_sample_size,
                            help='How many weight samples should be drawn to ' +
                                 'calculate an MC sample of the predictive ' +
                                 'distribution. Default: %(default)s.')


def train_args(tgroup, show_prior_variance=True, show_ll_dist_std=True,
               show_local_reparam_trick=False, show_kl_scale=False,
               show_radial_bnn=False):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the training argument group specific to training
    probabilistic models.

    Args:
        tgroup: The argument group returned by function
            :func:`utils.cli_args.train_args`.
        show_prior_variance (bool): Whether the option
            `prior_variance` should be provided.
        show_ll_dist_std (bool): Whether the option
            `local_reparam_trick` should be provided.
        show_local_reparam_trick (bool): Whether the option
            `ll_dist_std` should be provided.
        show_kl_scale (bool): Whether the option
            `kl_scale` should be provided.
        show_radial_bnn (bool): Whether the option
            `radial_bnn` should be provided.
    """
    if show_prior_variance:
        tgroup.add_argument('--prior_variance', type=float, default=1.0,
                            help='Variance of the Gaussian prior. ' +
                                 'Default: %(default)s.')
    if show_ll_dist_std:
        tgroup.add_argument('--ll_dist_std', type=float, default=0.1,
                            help='The standard deviation of the likelihood ' +
                                 'distribution. Note, this value should be ' +
                                 'fixed but reasonable for a given dataset.' +
                                 'Default: %(default)s.')
    if show_local_reparam_trick:
        tgroup.add_argument('--local_reparam_trick',action='store_true',
                            help='Use the local reparametrization trick.')
    if show_kl_scale:
        tgroup.add_argument('--kl_scale', type=float, default=1.,
                        help='A scaling factor for the prior matching term ' +
                             'in the variational inference loss. NOTE, this ' +
                             'option should be used with caution as it is ' +
                             'not part of the ELBO when deriving it ' +
                             'mathematically. ' +
                             'Default: %(default)s.')
    if show_radial_bnn:
        tgroup.add_argument('--radial_bnn', action='store_true',
                        help='Sample the weights of the BNN using a `Radial ' +
                             'BNN` posterior distribution instead of a ' +
                             'Gaussian. Note that this is useful to sidestep '+
                             'the soap bubble behavior of Gaussians in high ' +
                             'dimensions.')


def cl_args(clgroup):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the continous learning argument group for options specific
    to continual learning in a probabilistic setting (with Gaussian posteriors).

    Args:
        clgroup: The argument group returned by method
            :func:`utils.cli_args.cl_args`.
    """
    clgroup.add_argument('--regularizer', type=str, default='mse',
                        choices=['mse', 'fkl', 'rkl', 'w2'],
                        help='Type of regularizer for continual learning. ' +
                             'Options are: "mse", the mean-squared error ' +
                             'between the main net parameters before and ' +
                             'after learning the current task; "fkl", the ' +
                             'forward KL-divergence between target posterior ' +
                             'and current posterior per task, i.e., ' +
                             'KL(targets || hnet-output); "rkl", the reverse ' +
                             'KL-divergence, i.e., KL(hnet-output || targets)' +
                             '; "w2", the 2-wasserstein distance between the ' +
                             'posterior distributions before and after ' +
                             'learning the current task. Default: %(default)s.')
    clgroup.add_argument('--hnet_out_masking', type=float, default=0,
                        help='Fraction of task-conditioned hypernetwork ' +
                             'outputs that should be masked using a per-layer '+
                             'task-specific binary mask. A value of 0 means ' +
                             'that no outputs are masked while a value of 1 ' +
                             'means that all weights other than the output ' +
                             'weights are masked. Default: %(default)s.')

def init_args(agroup):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the initialization argument group.

    Args:
        agroup: The argument group returned by method
            :func:`utils.cli_args.init_args`.
    """
    agroup.add_argument('--keep_orig_init', action='store_true',
                        help='When converting the neural network into a ' +
                             'network with Gaussian weights, the main ' +
                             'network initialization (e.g., Xavier) will ' +
                             'be overwritten. This option assures that the ' +
                             'main network initialization is kept as an ' +
                             'initialization of the mean parameters in the ' +
                             'BNN. This option has an effect if no ' +
                             'hypernetwork is used or if the option ' +
                             '"hyper_gauss_init" is enabled.')
    agroup.add_argument('--hyper_gauss_init', action='store_true',
                        help='Initialize the hypernetwork such that the ' +
                             'variances of the Gaussian BNN initialization ' +
                             'are respected (i.e., the ones for means and ' +
                             'variances). Note, the initial expected value of ' +
                             'the variances is asserted due to a constant ' +
                             'offset. Option "std_normal_emb" will have no ' +
                             'effect if enabled.')

def miscellaneous_args(agroup, show_mnet_only=True, show_use_logvar_enc=False,
                       show_disable_lrt_test=False, show_mean_only=False,
                       show_during_acc_criterion=False,
                       show_store_during_models=False):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the miscellaneous argument group.

    Args:
        agroup: The argument group returned by method
            :func:`utils.cli_args.miscellaneous_args`.
        show_mnet_only (bool): Whether the option `mnet_only` should be
            provided.
        show_use_logvar_enc (bool): Whether the option
            `use_logvar_enc` should be provided.
        show_disable_lrt_test (bool): Whether the option
            `disable_lrt_test` should be provided.
        show_mean_only (bool): Whether the option `mean_only` should be
            provided.
        show_during_acc_criterion (bool): Whether the option
            `during_acc_criterion` should be provided.
        show_store_during_models (bool): Whether the option
            `store_during_models` should be provided.
    """
    if show_mnet_only:
        agroup.add_argument('--mnet_only', action='store_true',
                            help='Train without a hypernetwork (or ' +
                                 'hypernetworks).')
    if show_store_during_models:
        agroup.add_argument('--store_during_models', action='store_true',
                            help='Whether the during models (after training ' +
                                 'each task) should be checkpointed.')
    agroup.add_argument('--store_final_model', action='store_true',
                        help='Whether the final models (after training on ' +
                             'all tasks) should be checkpointed.')
    if show_use_logvar_enc:
        agroup.add_argument('--use_logvar_enc', action='store_true',
                            help='Use the log-variance encoding for the ' +
                                 'variance parameters of the Gaussian weight ' +
                                 'posterior.')
    if show_disable_lrt_test:
        agroup.add_argument('--disable_lrt_test', action='store_true',
                            help='If activated, the local-reparametrization ' +
                                 'trick will be disabled during testing, ' +
                                 'i.e., all test samples are processed using ' +
                                 'the same set of models.')
    if show_mean_only:
        agroup.add_argument('--mean_only', action='store_true',
                            help='Train deterministic network. Note, option ' +
                                 '"kl_scale" needs to be zero in this case, ' +
                                 'as no prior-matching can be applied.')
    if show_during_acc_criterion:
        agroup.add_argument('--during_acc_criterion', type=str, default='-1',
                        help='If "-1", the criterion is deactivated. ' +
                             'Otherwise, a list of comma-separated numbers ' +
                             'representing accuracies (between 0 - 100) is ' +
                             'expected. A run will be stopped if the during ' +
                             'accuracy of any task (except the last one) is ' +
                             'smaller than this value. Hence, this is an ' +
                             'easy way to avoid wasting ressources during ' +
                             'hyperparameter search. Note, the list should ' +
                             'either contain a single number or ' +
                             '"num_tasks-1" numbers. A value of "-1" would ' +
                             'deactivate the criterion for a task. ' +
                             'Default: %(default)s')


def check_invalid_bbb_args(config):
    """Sanity check for BbB command-line arguments.

    Args:
        config: Parsed command-line arguments.
    """
    if config.mnet_only and config.hyper_gauss_init:
        warnings.warn('Option "hyper_gauss_init" has no effect if no ' +
                      'hypernetwork is used.')
    if config.keep_orig_init and not \
            (config.mnet_only or config.hyper_gauss_init):
        warnings.warn('Option "keep_orig_init" has no effect if main ' +
                      'network has no parameters or option ' +
                      '"hyper_gauss_init" is not activated.')
    if not config.mnet_only and config.hyper_gauss_init and \
            config.normal_init:
        warnings.warn('Option "normal_init" has no effect if ' +
                      '"hyper_gauss_init" is activated.')
    if config.mnet_only and not config.keep_orig_init and \
            config.normal_init:
        warnings.warn('Option "normal_init" has no effect for main net ' +
                      'initialization if "keep_orig_init" is not ' +
                      'activated.')
    if config.local_reparam_trick:
        if hasattr(config, 'dropout_rate') and config.dropout_rate != -1:
            raise ValueError('Dropout not implemented for network with ' +
                             'local reparametrization trick.')
        if hasattr(config, 'specnorm') and config.specnorm:
            raise ValueError('Spectral norm not implemented for network ' +
                             'with local reparametrization trick.')
        if hasattr(config, 'batchnorm') and config.batchnorm or \
                hasattr(config, 'no_batchnorm') and not config.no_batchnorm:
            raise ValueError('Batchnorm not implemented for network ' +
                             'with local reparametrization trick.')
    if not config.local_reparam_trick and config.disable_lrt_test:
        warnings.warn('Option "disable_lrt_test" has no effect if the local-'
                      'reparametrization trick is not used.')

    if hasattr(config, 'mean_only') and config.mean_only:
        if hasattr(config, 'kl_scale') and config.kl_scale != 0 or \
                hasattr(config, 'kl_schedule') and config.kl_schedule != 0:
            raise ValueError('Prior-matching is not applicable for ' +
                             'deterministic networks.')
        if config.regularizer != 'mse':
            raise ValueError('Only "mse" regularizer can be applied to ' +
                             'deterministic networks.')
        if config.local_reparam_trick:
            raise ValueError('Local-reparametrization trick cannot be ' +
                             'applied to non-Gaussian networks.')
        if config.hyper_gauss_init:
            raise ValueError('Gaussian-hypernet init cannot be applied to ' +
                             'non-Gaussian networks.')
        if hasattr(config, 'use_prev_post_as_prior') and \
                config.use_prev_post_as_prior:
            raise ValueError('Option "use_prev_post_as_prior" cannot be ' +
                             'enforced for deterministic networks.')
        if config.train_sample_size > 1:
            warnings.warn('A "train_sample_size" greater than 1 doesn\'t ' +
                          'make sense for a deterministic network.')
        if config.val_sample_size > 1:
            warnings.warn('A "val_sample_size" greater than 1 doesn\'t ' +
                          'make sense for a deterministic network.')
        if config.disable_lrt_test:
            warnings.warn('Option "disable_lrt_test" not applicable to ' +
                          'deterministic networks.')
        if config.use_logvar_enc:
            warnings.warn('Option "use_logvar_enc" not applicable to ' +
                          'deterministic networks.')

    if config.radial_bnn:
        if config.local_reparam_trick:
            raise ValueError('Local-reparametrization trick is not compatible '+
                             'with Radial BNNs since the weights posterior is '+
                             'not Gaussian anymore.')
        if config.regularizer != 'mse':
            raise NotImplementedError('Only the MSE regularizer has been ' +
                                      'implemented for radial BNN ' +
                                      'distributions.')
        if hasattr(config, 'use_prev_post_as_prior') and \
                config.use_prev_post_as_prior:
            raise NotImplementedError('Option "use_prev_post_as_prior" not ' +
                                      'implemented for Radial BNN.')

def check_invalid_args_general(config):
    """Sanity check for some general command-line arguments.

    Args:
        config: Parsed command-line arguments.
    """
    # Not mathematically correct, but might be required if prior is not
    # appropriate.
    if hasattr(config, 'kl_scale') and  config.kl_scale != 1.0:
        warnings.warn('Prior matching term will be scaled by %f.'
                      % config.kl_scale)

    if hasattr(config, 'store_final_model') and \
            hasattr(config, 'train_from_scratch') and \
            config.store_final_model and config.train_from_scratch:
        warnings.warn('Note, when training from scratch, the final model is ' +
                      'only trained on the last task!')

    if hasattr(config, 'hnet_out_masking'):
        if config.hnet_out_masking > 1. or config.hnet_out_masking < 0.:
            raise ValueError('Fraction of hypernetwork outputs to be masked ' +
                             'should be between 0 and 1.')
        if config.hnet_out_masking != 0:
            if not hasattr(config, 'mean_only') or not config.mean_only:
                # Prior-matching needs to be adapted for non-det methods and
                # for masks need to be synchronized if means and variances are
                # in the main net.
                raise NotImplementedError('Masking of the hnet output only ' +
                    'implemented yet for deterministic solutions.')
    if hasattr(config, 'supsup_task_inference'):
        if config.supsup_task_inference:
            pass
        if not config.supsup_task_inference:
            if config.supsup_grad_steps != 1:
                warnings.warn('The number of SupSup steps is irrelevant if ' +
                              '"supsup_task_inference" is not activated.')

if __name__ == '__main__':
    pass
