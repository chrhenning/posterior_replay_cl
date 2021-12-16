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
# title          :probabilistic/prob_mnist/train_args.py
# author         :ch
# contact        :henningc@ethz.ch
# created        :09/06/2019
# version        :1.0
# python_version :3.6.8
"""
Command-line argument definition and parsing
--------------------------------------------

All command-line arguments and default values for this subpackage are handled
in this module.
"""
import argparse
from datetime import datetime
import warnings

from probabilistic.regression import train_args as rta
from probabilistic.prob_cifar import train_args as pcta
import utils.cli_args as cli

def parse_cmd_arguments(mode='split_mnist_bbb', default=False, argv=None):
    """Parse command-line arguments.

    Args:
        mode (str): For what script should the parser assemble the set of
            command-line parameters? Options:

                - "gmm_bbb"
                - "gmm_avb"
                - "gmm_avb_pf"
                - "gmm_ssge"
                - "gmm_ssge_pf"
                - "split_mnist_bbb"
                - "perm_mnist_bbb"
                - "cifar_zenke_bbb"
                - "cifar_resnet_bbb"
                - "split_mnist_avb"
                - "split_mnist_avb_pf"
                - "perm_mnist_avb"
                - "perm_mnist_avb_pf"
                - "cifar_zenke_avb"
                - "cifar_zenke_avb_pf"
                - "cifar_resnet_avb"
                - "cifar_resnet_avb_pf"
                - "split_mnist_ssge"
                - "split_mnist_ssge_pf"
                - "perm_mnist_ssge"
                - "perm_mnist_ssge_pf"
                - "cifar_resnet_ssge"
                - "cifar_resnet_ssge_pf"

        default (bool, optional): If ``True``, command-line arguments will be
            ignored and only the default values will be parsed.
        argv (list, optional): If provided, it will be treated as a list of
            command- line argument that is passed to the parser in place of
            :code:`sys.argv`.

    Returns:
        (argparse.Namespace): The Namespace object containing argument names and
            values.
    """
    if mode == 'gmm_bbb':
        description = 'Probabilistic CL on GMM Datasets via BbB'
    elif mode == 'split_mnist_bbb':
        description = 'Probabilistic CL on Split MNIST via BbB'
    elif mode == 'perm_mnist_bbb':
        description = 'Probabilistic CL on Permuted MNIST via BbB'
    elif mode == 'cifar_zenke_bbb':
        description = 'Probabilistic CL on CIFAR-10/100 via BbB on the ZenkeNet'
    elif mode == 'cifar_resnet_bbb':
        description = 'Probabilistic CL on CIFAR-10/100 via BbB on a Resnet'
    elif mode == 'gmm_avb':
        description = 'Posterior-Replay CL on GMM Datasets via AVB'
    elif mode == 'gmm_avb_pf':
        description = 'Prior-Focused CL on GMM Datasets via AVB'
    elif mode == 'split_mnist_avb':
        description = 'Posterior-Replay CL on Split MNIST via AVB'
    elif mode == 'split_mnist_avb_pf':
        description = 'Prior-Focused CL on Split MNIST via AVB'
    elif mode == 'perm_mnist_avb':
        description = 'Posterior-Replay CL on Permuted MNIST via AVB'
    elif mode == 'perm_mnist_avb_pf':
        description = 'Prior-Focused CL on Permuted MNIST via AVB'
    elif mode == 'cifar_zenke_avb':
        description = 'Posterior-Replay CL on CIFAR-10/100 via AVB on a ' + \
                      'ZenkeNet'
    elif mode == 'cifar_zenke_avb_pf':
        description = 'Prior-Focused CL on CIFAR-10/100 via AVB on a ZenkeNet'
    elif mode == 'cifar_resnet_avb':
        description = 'Posterior-Replay CL on CIFAR-10/100 via AVB on a Resnet'
    elif mode == 'cifar_resnet_avb_pf':
        description = 'Prior-Focused CL on CIFAR-10/100 via AVB on a Resnet'
    elif mode == 'gmm_ssge':
        description = 'Posterior-Replay CL on GMM Datasets via SSGE'
    elif mode == 'gmm_ssge_pf':
        description = 'Prior-Focused CL on GMM Datasets via SSGE'
    elif mode == 'split_mnist_ssge':
        description = 'Posterior-Replay CL on Split MNIST via SSGE'
    elif mode == 'split_mnist_ssge_pf':
        description = 'Prior-Focused CL on Split MNIST via SSGE'
    elif mode == 'perm_mnist_ssge':
        description = 'Posterior-Replay CL on Permuted MNIST via SSGE'
    elif mode == 'perm_mnist_ssge_pf':
        description = 'Prior-Focused CL on Permuted MNIST via SSGE'
    elif mode == 'cifar_resnet_ssge':
        description = 'Posterior-Replay CL on CIFAR-10/100 via SSGE on a Resnet'
    elif mode == 'cifar_resnet_ssge_pf':
        description = 'Prior-Focused CL on CIFAR-10/100 via SSGE on a Resnet'
    else:
        raise Exception('Mode "%s" unknown.' % (mode))

    parser = argparse.ArgumentParser(description=description)

    ### Shared keyword arguments of CLI helper functions.
    # To decrease code copying, we specify dictionaries of keyword arguments
    # (e.g., for functions from `cli_args`), if most experiments handled below
    # use the same keywords.
    hnet_args_kw = { # Function `cli.hnet_args`
        # We exclude `'structured_hmlp'` below, as not all experiments will
        # support it!
        # Note, the first list element denotes the default hnet.
        'allowed_nets': ['hmlp', 'chunked_hmlp', 'hdeconv', 'chunked_hdeconv'],
        'dhmlp_arch': '125,250,500',
        'show_cond_emb_size': True,
        'dcond_emb_size': 32,
        'dchmlp_chunk_size': 1500,
        'dchunk_emb_size': 32,
        'show_use_cond_chunk_embs': True,
        'show_net_act': True,
        'dnet_act': 'relu',
        'show_no_bias': True,
        'show_dropout_rate': True,
        'ddropout_rate': -1,
        'show_specnorm': True,
        'show_batchnorm': False,
        'show_no_batchnorm': False
    }
    hnet_args_kw_with_shmlp = dict(hnet_args_kw)
    hnet_args_kw_with_shmlp['allowed_nets'] = hnet_args_kw['allowed_nets'] + \
        ['structured_hmlp']
    imp_hnet_args_kw = dict(hnet_args_kw) # KWs for implicit hnet.
    imp_hnet_args_kw['show_cond_emb_size'] = False
    imp_hnet_args_kw['dcond_emb_size'] = 0
    imp_hnet_args_kw['show_use_cond_chunk_embs'] = False
    imp_hnet_args_kw['prefix'] = 'imp_'
    imp_hnet_args_kw['pf_name'] ='implicit'
    imp_hnet_args_kw_with_shmlp = dict(imp_hnet_args_kw)
    imp_hnet_args_kw_with_shmlp['allowed_nets'] = \
        imp_hnet_args_kw['allowed_nets'] + ['structured_hmlp']
    hh_hnet_args_kw = dict(hnet_args_kw) # KWs for hyper-hnet.
    hh_hnet_args_kw['prefix'] = 'hh_'
    hh_hnet_args_kw['pf_name'] ='hyper-hyper'

    cl_args_kw = { # Method `cli.cl_args`
        'show_beta': True,
        'dbeta': 1.,
        'show_from_scratch': True,
        'show_multi_head': False,
        'show_cl_scenario': True,
        'show_split_head_cl3': True,
        'show_num_tasks': True,
        'dnum_tasks': 5,
        'show_num_classes_per_task': True,
        'dnum_classes_per_task': 2,
    }
    gmm_cl_args_kw = dict(cl_args_kw)
    gmm_cl_args_kw['show_num_tasks'] = False
    gmm_cl_args_kw['show_num_classes_per_task'] = False
    split_cl_args_kw = dict(cl_args_kw)
    split_cl_args_kw['dbeta'] = .01
    split_cl_args_kw['dnum_tasks'] = 5
    split_cl_args_kw['dnum_classes_per_task'] = 2
    perm_cl_args_kw = dict(cl_args_kw)
    perm_cl_args_kw['dbeta'] = .01
    perm_cl_args_kw['dnum_tasks'] = 10
    perm_cl_args_kw['show_num_classes_per_task'] = False
    cifar_cl_args_kw = dict(cl_args_kw)
    cifar_cl_args_kw['dbeta'] = .01
    cifar_cl_args_kw['dnum_tasks'] = 6
    cifar_cl_args_kw['dnum_classes_per_task'] = 10

    train_args_kw = { # Method `cli.train_args`
        'show_lr': True,
        'dlr': .001,
        'dbatch_size': 128,
        'dn_iter': 2000,
        'show_use_adam': True,
        'show_use_rmsprop': True,
        'show_use_adadelta': True,
        'show_use_adagrad': True,
        'show_epochs': True,
        'show_clip_grad_value': True,
        'show_clip_grad_norm': True,
    }
    gmm_train_args_kw = dict(train_args_kw)
    gmm_train_args_kw['dbatch_size'] = 32
    split_train_args_kw = dict(train_args_kw)
    perm_train_args_kw = dict(train_args_kw)
    perm_train_args_kw['dn_iter'] = 5000
    perm_train_args_kw['dlr'] = .0001
    zenke_train_args_kw = dict(train_args_kw)
    zenke_train_args_kw['dlr'] = .0001
    zenke_train_args_kw['depochs'] = 80
    zenke_train_args_kw['dbatch_size'] = 256
    zenke_train_args_kw['dadam_beta1'] = .5
    resnet_train_args_kw = dict(train_args_kw)
    resnet_train_args_kw['depochs'] = 200
    resnet_train_args_kw['dbatch_size'] = 32

    main_args_kw = { # Method `cli.main_net_args`
        'allowed_nets': ['mlp', 'lenet', 'resnet', 'wrn'],
        'dmlp_arch': '400,400',
        'dlenet_type': 'mnist_small',
        'show_no_bias': True,
        'show_batchnorm': True,
        'show_no_batchnorm': False,
        'show_bn_no_running_stats': True,
        'show_bn_distill_stats': False,
        'show_bn_no_stats_checkpointing': True,
        'show_specnorm': False,
        'show_dropout_rate': True,
        'ddropout_rate': -1,
        'show_net_act': True,
    }
    gmm_main_args_kw = dict(main_args_kw)
    gmm_main_args_kw['allowed_nets'] = ['mlp']
    gmm_main_args_kw['dmlp_arch'] = '10,10'
    split_main_args_kw = dict(main_args_kw)
    perm_main_args_kw = dict(main_args_kw)
    perm_main_args_kw['allowed_nets'] = ['mlp']
    perm_main_args_kw['dmlp_arch'] = '1000,1000'
    zenke_main_args_kw = dict(main_args_kw)
    zenke_main_args_kw['allowed_nets'] = ['zenke']
    zenke_main_args_kw['show_no_bias'] = False
    zenke_main_args_kw['show_batchnorm'] = False
    zenke_main_args_kw['show_bn_no_running_stats'] = False
    zenke_main_args_kw['show_bn_no_stats_checkpointing'] = False
    zenke_main_args_kw['ddropout_rate'] = 0.25
    zenke_main_args_kw['show_net_act'] = False
    resnet_main_args_kw = dict(main_args_kw)
    resnet_main_args_kw['allowed_nets'] = ['resnet', 'wrn', 'iresnet', 'lenet',
                                           'zenke', 'mlp']
    resnet_main_args_kw['dlenet_type'] = 'cifar'
    resnet_main_args_kw['show_batchnorm'] = False
    resnet_main_args_kw['show_no_batchnorm'] = True

    dis_args_kw = { # Method `cli.main_net_args`
        'allowed_nets': ['mlp', 'chunked_mlp'],
        'dmlp_arch': '100,100',
        'dcmlp_arch': '10,10',
        'dcmlp_chunk_arch': '10,10',
        'dcmlp_in_cdim': 100,
        'dcmlp_out_cdim': 10,
        'dcmlp_cemb_dim': 8,
        'show_no_bias': True,
        'prefix': 'dis_',
        'pf_name': 'discriminator',
    }
    gmm_dis_args_kw = dict(dis_args_kw)
    gmm_dis_args_kw['dmlp_arch'] = '10,10'
    gmm_dis_args_kw['dcmlp_in_cdim'] = 32
    gmm_dis_args_kw['dcmlp_out_cdim'] = 8
    split_dis_args_kw = dict(dis_args_kw)
    perm_dis_args_kw = dict(dis_args_kw)
    zenke_dis_args_kw = dict(dis_args_kw)
    resnet_dis_args_kw = dict(dis_args_kw)

    ################
    ### GMM BbB ###
    ###############
    if mode == 'gmm_bbb':
        dout_dir = './out_gmm_bbb/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **gmm_cl_args_kw)
        train_agroup = cli.train_args(parser, **gmm_train_args_kw)
        cli.main_net_args(parser, **gmm_main_args_kw)
        tmp_hargs_kw = dict(hnet_args_kw)
        tmp_hargs_kw['dhmlp_arch'] = '10,10'
        tmp_hargs_kw['dcond_emb_size'] = 2
        tmp_hargs_kw['dchunk_emb_size'] = 2
        cli.hnet_args(parser, **tmp_hargs_kw)
        init_agroup = cli.init_args(parser, custom_option=False)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                                    dval_batch_size=10000, dval_iter=100)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup, show_supsup_task_inference=True)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=True, show_kl_scale=True,
                       show_radial_bnn=True)
        rta.cl_args(cl_argroup)
        rta.init_args(init_agroup)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=True,
                               show_disable_lrt_test=True, show_mean_only=True,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_calc_hnet_reg_targets_online=True,
                   show_hnet_reg_batch_size=True, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_experience_replay=True)
        prob_args(parser)

    ######################
    ### SplitMNIST BbB ###
    ######################
    if mode == 'split_mnist_bbb':
        dout_dir = './out_split_bbb/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **split_cl_args_kw)
        train_agroup = cli.train_args(parser, **split_train_args_kw)
        cli.main_net_args(parser, **split_main_args_kw)
        hnet_agroup = cli.hnet_args(parser, **hnet_args_kw_with_shmlp)
        hnet_args(hnet_agroup)
        init_agroup = cli.init_args(parser, custom_option=False)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup, show_supsup_task_inference=True)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=True, show_kl_scale=True,
                       show_radial_bnn=True)
        rta.cl_args(cl_argroup)
        rta.init_args(init_agroup)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=True,
                               show_disable_lrt_test=True, show_mean_only=True,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_calc_hnet_reg_targets_online=True,
                   show_hnet_reg_batch_size=True, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_experience_replay=True)
        prob_args(parser)

    ##########################
    ### PermutedMNIST BbB ###
    #########################
    elif mode == 'perm_mnist_bbb':
        dout_dir = './out_perm_bbb/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_agroup = cli.cl_args(parser, **perm_cl_args_kw)
        train_agroup = cli.train_args(parser, **perm_train_args_kw)
        # Note, we don't add the LeNet option, as permuted MNIST doesn't make
        # sense with convolutional networks.
        cli.main_net_args(parser, **perm_main_args_kw)
        cli.hnet_args(parser, **hnet_args_kw)
        init_agroup = cli.init_args(parser, custom_option=False)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup, show_supsup_task_inference=True)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=True, show_kl_scale=True,
                       show_radial_bnn=True)
        rta.cl_args(cl_agroup)
        rta.init_args(init_agroup)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=True,
                               show_disable_lrt_test=True, show_mean_only=True,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_calc_hnet_reg_targets_online=True,
                   show_hnet_reg_batch_size=True, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_experience_replay=True)
        prob_args(parser)

        perm_args(parser)

    ##########################
    ### ZenkeNet CIFAR BbB ###
    ##########################
    elif mode == 'cifar_zenke_bbb':
        dout_dir = './out_zenke_bbb/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **cifar_cl_args_kw)
        pcta.extra_cl_args(cl_argroup)
        train_agroup = cli.train_args(parser, **zenke_train_args_kw)
        cli.main_net_args(parser, **zenke_main_args_kw)
        cli.hnet_args(parser, **hnet_args_kw)
        init_agroup = cli.init_args(parser, custom_option=False)
        cli.data_args(parser, show_disable_data_augmentation=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup, show_supsup_task_inference=True)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=True, show_kl_scale=True,
                       show_radial_bnn=True)
        rta.cl_args(cl_argroup)
        rta.init_args(init_agroup)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=True,
                               show_disable_lrt_test=True, show_mean_only=True,
                               show_during_acc_criterion=True)

        train_args(train_agroup, show_calc_hnet_reg_targets_online=True,
                   show_hnet_reg_batch_size=True, show_num_kl_samples=True)
        ind_posterior_args(train_agroup)
        prob_args(parser)

    ########################
    ### Resnet CIFAR BbB ###
    ########################
    elif mode == 'cifar_resnet_bbb':
        dout_dir = './out_resnet_bbb/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **cifar_cl_args_kw)
        pcta.extra_cl_args(cl_argroup)
        train_agroup = cli.train_args(parser, **resnet_train_args_kw)
        cli.main_net_args(parser, **resnet_main_args_kw)
        hnet_agroup = cli.hnet_args(parser, **hnet_args_kw_with_shmlp)
        hnet_args(hnet_agroup)
        init_agroup = cli.init_args(parser, custom_option=False)
        cli.data_args(parser, show_disable_data_augmentation=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup, show_supsup_task_inference=True)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=True, show_kl_scale=True,
                       show_radial_bnn=True)
        rta.cl_args(cl_argroup)
        rta.init_args(init_agroup)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=True,
                               show_disable_lrt_test=True, show_mean_only=True,
                               show_during_acc_criterion=True)

        train_args(train_agroup, show_calc_hnet_reg_targets_online=True,
                   show_hnet_reg_batch_size=True, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_experience_replay=True)
        prob_args(parser)

    ################
    ### GMM AVB ###
    ###############
    if mode == 'gmm_avb':
        dout_dir = './out_gmm_avb/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **gmm_cl_args_kw)
        train_agroup = cli.train_args(parser, **gmm_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **gmm_main_args_kw)
        # Discriminator.
        cli.main_net_args(parser, **gmm_dis_args_kw)
        # Hypernetwork (weight generator).
        tmp_imp_hargs_kw = dict(imp_hnet_args_kw)
        tmp_imp_hargs_kw['dhmlp_arch'] = '10,10'
        tmp_imp_hargs_kw['dchunk_emb_size'] = 2
        cli.hnet_args(parser, **tmp_imp_hargs_kw)
        # Hyper-hypernetwork.
        tmp_hh_hargs_kw = dict(hh_hnet_args_kw)
        tmp_hh_hargs_kw['dhmlp_arch'] = '10,10'
        tmp_hh_hargs_kw['dcond_emb_size'] = 2
        tmp_hh_hargs_kw['dchunk_emb_size'] = 2
        cli.hnet_args(parser, **tmp_hh_hargs_kw)
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                                    dval_batch_size=10000, dval_iter=100)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=True,
            show_use_prev_post_as_prior=True, show_num_kl_samples=True,
            show_calc_hnet_reg_targets_online=True,
            show_hnet_reg_batch_size=True)
        ind_posterior_args(train_agroup, show_distill_iter=True,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=True)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.avb_args(parser)
        prob_args(parser)

    ##################
    ### GMM AVB-PF ###
    ##################
    if mode == 'gmm_avb_pf':
        dout_dir = './out_gmm_avb_pf/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Note, no `beta` in prior-focused CL, since there is not our hnet
        # regularizer.
        tmp_gmm_cl_args_kw = dict(gmm_cl_args_kw)
        tmp_gmm_cl_args_kw['show_beta'] = False
        cl_argroup = cli.cl_args(parser, **tmp_gmm_cl_args_kw)
        train_agroup = cli.train_args(parser, **gmm_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **gmm_main_args_kw)
        # Discriminator.
        cli.main_net_args(parser, **gmm_dis_args_kw)
        # Hypernetwork (weight generator).
        tmp_imp_hargs_kw = dict(imp_hnet_args_kw)
        tmp_imp_hargs_kw['dhmlp_arch'] = '10,10'
        tmp_imp_hargs_kw['dchunk_emb_size'] = 2
        cli.hnet_args(parser, **tmp_imp_hargs_kw)

        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                                    dval_batch_size=10000, dval_iter=100)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=False,
            show_use_prev_post_as_prior=False, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_distill_iter=False,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=False)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.avb_args(parser)
        prob_args(parser)

    ######################
    ### SplitMNIST AVB ###
    ######################
    elif mode == 'split_mnist_avb':
        dout_dir = './out_split_avb/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **split_cl_args_kw)
        train_agroup = cli.train_args(parser, **split_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **split_main_args_kw)
        # Discriminator.
        cli.main_net_args(parser, **split_dis_args_kw)
        # Hypernetwork (weight generator).
        hnet_agroup = cli.hnet_args(parser, **imp_hnet_args_kw_with_shmlp)
        hnet_args(hnet_agroup, prefix='imp_')
        # Hyper-hypernetwork.
        cli.hnet_args(parser, **hh_hnet_args_kw)
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=True,
            show_use_prev_post_as_prior=True, show_num_kl_samples=True,
            show_calc_hnet_reg_targets_online=True,
            show_hnet_reg_batch_size=True)
        ind_posterior_args(train_agroup, show_distill_iter=True,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=True)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.avb_args(parser)
        prob_args(parser)

    #########################
    ### SplitMNIST AVB-PF ###
    #########################
    elif mode == 'split_mnist_avb_pf':
        dout_dir = './out_split_avb_pf/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Note, no `beta` in prior-focused CL, since there is not our hnet
        # regularizer.
        tmp_split_cl_args_kw = dict(split_cl_args_kw)
        tmp_split_cl_args_kw['show_beta'] = False
        cl_argroup = cli.cl_args(parser, **tmp_split_cl_args_kw)
        train_agroup = cli.train_args(parser, **split_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **split_main_args_kw)
        # Discriminator.
        cli.main_net_args(parser, **split_dis_args_kw)
        # Hypernetwork (weight generator).
        hnet_agroup = cli.hnet_args(parser, **imp_hnet_args_kw_with_shmlp)
        hnet_args(hnet_agroup, prefix='imp_')

        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=False,
            show_use_prev_post_as_prior=False, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_distill_iter=False,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=False)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.avb_args(parser)
        prob_args(parser)

    #########################
    ### PermutedMNIST AVB ###
    #########################
    elif mode == 'perm_mnist_avb':
        dout_dir = './out_perm_avb/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **perm_cl_args_kw)
        train_agroup = cli.train_args(parser, **perm_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **perm_main_args_kw)
        # Discriminator.
        cli.main_net_args(parser, **perm_dis_args_kw)
        # Hypernetwork (weight generator).
        cli.hnet_args(parser, **imp_hnet_args_kw)
        # Hyper-hypernetwork.
        cli.hnet_args(parser, **hh_hnet_args_kw)
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=True,
            show_use_prev_post_as_prior=True, show_num_kl_samples=True,
            show_calc_hnet_reg_targets_online=True,
            show_hnet_reg_batch_size=True)
        ind_posterior_args(train_agroup, show_distill_iter=True,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=True)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.avb_args(parser)
        prob_args(parser)

        perm_args(parser)

    ############################
    ### PermutedMNIST AVB-PF ###
    ############################
    elif mode == 'perm_mnist_avb_pf':
        dout_dir = './out_perm_avb_pf/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Note, no `beta` in prior-focused CL, since there is not our hnet
        # regularizer.
        tmp_perm_cl_args_kw = dict(perm_cl_args_kw)
        tmp_perm_cl_args_kw['show_beta'] = False
        cl_argroup = cli.cl_args(parser, **tmp_perm_cl_args_kw)
        train_agroup = cli.train_args(parser, **perm_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **perm_main_args_kw)
        # Discriminator.
        cli.main_net_args(parser, **perm_dis_args_kw)
        # Hypernetwork (weight generator).
        cli.hnet_args(parser, **imp_hnet_args_kw)

        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=False,
            show_use_prev_post_as_prior=False, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_distill_iter=False,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=False)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.avb_args(parser)
        prob_args(parser)

        perm_args(parser)

    ##########################
    ### ZenkeNet CIFAR AVB ###
    ##########################
    elif mode == 'cifar_zenke_avb':
        dout_dir = './out_zenke_avb/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **cifar_cl_args_kw)
        pcta.extra_cl_args(cl_argroup)
        train_agroup = cli.train_args(parser, **zenke_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **zenke_main_args_kw)
        # Discriminator.
        cli.main_net_args(parser, **zenke_dis_args_kw)
        # Hypernetwork (weight generator).
        cli.hnet_args(parser, **imp_hnet_args_kw)
        # Hyper-hypernetwork.
        cli.hnet_args(parser, **hh_hnet_args_kw)
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        cli.data_args(parser, show_disable_data_augmentation=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=True,
            show_use_prev_post_as_prior=True, show_num_kl_samples=True,
            show_calc_hnet_reg_targets_online=True,
            show_hnet_reg_batch_size=True)
        ind_posterior_args(train_agroup, show_distill_iter=True,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=True)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.avb_args(parser)
        prob_args(parser)

    ##############################
    ### ZenkeNet CIFAR AVB-PF ###
    #############################
    elif mode == 'cifar_zenke_avb_pf':
        dout_dir = './out_zenke_avb_pf/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Note, no `beta` in prior-focused CL, since there is not our hnet
        # regularizer.
        tmp_cifar_cl_args_kw = dict(cifar_cl_args_kw)
        tmp_cifar_cl_args_kw['show_beta'] = False
        cl_argroup = cli.cl_args(parser, **tmp_cifar_cl_args_kw)
        pcta.extra_cl_args(cl_argroup)
        train_agroup = cli.train_args(parser, **zenke_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **zenke_main_args_kw)
        # Discriminator.
        cli.main_net_args(parser, **zenke_dis_args_kw)
        # Hypernetwork (weight generator).
        cli.hnet_args(parser, **imp_hnet_args_kw)

        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        cli.data_args(parser, show_disable_data_augmentation=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=False,
            show_use_prev_post_as_prior=False, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_distill_iter=False,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=False)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.avb_args(parser)
        prob_args(parser)

    ########################
    ### Resnet CIFAR AVB ###
    ########################
    elif mode == 'cifar_resnet_avb':
        dout_dir = './out_resnet_avb/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **cifar_cl_args_kw)
        pcta.extra_cl_args(cl_argroup)
        train_agroup = cli.train_args(parser, **resnet_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **resnet_main_args_kw)
        # Discriminator.
        cli.main_net_args(parser, **resnet_dis_args_kw)
        # Hypernetwork (weight generator).
        hnet_agroup = cli.hnet_args(parser, **imp_hnet_args_kw_with_shmlp)
        hnet_args(hnet_agroup, prefix='imp_')
        # Hyper-hypernetwork.
        cli.hnet_args(parser, **hh_hnet_args_kw)
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        cli.data_args(parser, show_disable_data_augmentation=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=True,
            show_use_prev_post_as_prior=True, show_num_kl_samples=True,
            show_calc_hnet_reg_targets_online=True,
            show_hnet_reg_batch_size=True)
        ind_posterior_args(train_agroup, show_distill_iter=True,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=True)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.avb_args(parser)
        prob_args(parser)

    ############################
    ### Resnet CIFAR AVB-PF ###
    ###########################
    elif mode == 'cifar_resnet_avb_pf':
        dout_dir = './out_resnet_avb_pf/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Note, no `beta` in prior-focused CL, since there is not our hnet
        # regularizer.
        tmp_cifar_cl_args_kw = dict(cifar_cl_args_kw)
        tmp_cifar_cl_args_kw['show_beta'] = False
        cl_argroup = cli.cl_args(parser, **tmp_cifar_cl_args_kw)
        pcta.extra_cl_args(cl_argroup)
        train_agroup = cli.train_args(parser, **resnet_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **resnet_main_args_kw)
        # Discriminator.
        cli.main_net_args(parser, **resnet_dis_args_kw)
        # Hypernetwork (weight generator).
        hnet_agroup = cli.hnet_args(parser, **imp_hnet_args_kw_with_shmlp)
        hnet_args(hnet_agroup, prefix='imp_')

        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        cli.data_args(parser, show_disable_data_augmentation=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=False,
            show_use_prev_post_as_prior=False, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_distill_iter=False,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=False)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.avb_args(parser)
        prob_args(parser)

    ################
    ### GMM SSGE ###
    ################
    if mode == 'gmm_ssge':
        dout_dir = './out_gmm_ssge/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **gmm_cl_args_kw)
        train_agroup = cli.train_args(parser, **gmm_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **gmm_main_args_kw)
        # Hypernetwork (weight generator).
        tmp_imp_hargs_kw = dict(imp_hnet_args_kw)
        tmp_imp_hargs_kw['dhmlp_arch'] = '10,10'
        tmp_imp_hargs_kw['dchunk_emb_size'] = 2
        cli.hnet_args(parser, **tmp_imp_hargs_kw)
        # Hyper-hypernetwork.
        tmp_hh_hargs_kw = dict(hh_hnet_args_kw)
        tmp_hh_hargs_kw['dhmlp_arch'] = '10,10'
        tmp_hh_hargs_kw['dcond_emb_size'] = 2
        tmp_hh_hargs_kw['dchunk_emb_size'] = 2
        cli.hnet_args(parser, **tmp_hh_hargs_kw)
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                                    dval_batch_size=10000, dval_iter=100)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=True,
            show_use_prev_post_as_prior=True, show_num_kl_samples=True,
            show_calc_hnet_reg_targets_online=True,
            show_hnet_reg_batch_size=True)
        ind_posterior_args(train_agroup, show_distill_iter=True,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=True)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.ssge_args(parser)
        prob_args(parser)

    ###################
    ### GMM SSGE-PF ###
    ###################
    if mode == 'gmm_ssge_pf':
        dout_dir = './out_gmm_ssge_pf/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Note, no `beta` in prior-focused CL, since there is not our hnet
        # regularizer.
        tmp_gmm_cl_args_kw = dict(gmm_cl_args_kw)
        tmp_gmm_cl_args_kw['show_beta'] = False
        cl_argroup = cli.cl_args(parser, **tmp_gmm_cl_args_kw)
        train_agroup = cli.train_args(parser, **gmm_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **gmm_main_args_kw)
        # Hypernetwork (weight generator).
        tmp_imp_hargs_kw = dict(imp_hnet_args_kw)
        tmp_imp_hargs_kw['dhmlp_arch'] = '10,10'
        tmp_imp_hargs_kw['dchunk_emb_size'] = 2
        cli.hnet_args(parser, **tmp_imp_hargs_kw)

        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                                    dval_batch_size=10000, dval_iter=100)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=False,
            show_use_prev_post_as_prior=False, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_distill_iter=False,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=False)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.ssge_args(parser)
        prob_args(parser)

    #######################
    ### SplitMNIST SSGE ###
    #######################
    elif mode == 'split_mnist_ssge':
        dout_dir = './out_split_ssge/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **split_cl_args_kw)
        train_agroup = cli.train_args(parser, **split_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **split_main_args_kw)
        # Hypernetwork (weight generator).
        hnet_agroup = cli.hnet_args(parser, **imp_hnet_args_kw_with_shmlp)
        hnet_args(hnet_agroup, prefix='imp_')
        # Hyper-hypernetwork.
        cli.hnet_args(parser, **hh_hnet_args_kw)
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=True,
            show_use_prev_post_as_prior=True, show_num_kl_samples=True,
            show_calc_hnet_reg_targets_online=True,
            show_hnet_reg_batch_size=True)
        ind_posterior_args(train_agroup, show_distill_iter=True,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=True)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.ssge_args(parser)
        prob_args(parser)

    ##########################
    ### SplitMNIST SSGE-PF ###
    ##########################
    elif mode == 'split_mnist_ssge_pf':
        dout_dir = './out_split_ssge_pf/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Note, no `beta` in prior-focused CL, since there is not our hnet
        # regularizer.
        tmp_split_cl_args_kw = dict(split_cl_args_kw)
        tmp_split_cl_args_kw['show_beta'] = False
        cl_argroup = cli.cl_args(parser, **tmp_split_cl_args_kw)
        train_agroup = cli.train_args(parser, **split_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **split_main_args_kw)
        # Hypernetwork (weight generator).
        hnet_agroup = cli.hnet_args(parser, **imp_hnet_args_kw_with_shmlp)
        hnet_args(hnet_agroup, prefix='imp_')

        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=False,
            show_use_prev_post_as_prior=False, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_distill_iter=False,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=False)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.ssge_args(parser)
        prob_args(parser)

    ##########################
    ### PermutedMNIST SSGE ###
    ##########################
    elif mode == 'perm_mnist_ssge':
        dout_dir = './out_perm_ssge/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **perm_cl_args_kw)
        train_agroup = cli.train_args(parser, **perm_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **perm_main_args_kw)
        # Hypernetwork (weight generator).
        cli.hnet_args(parser, **imp_hnet_args_kw)
        # Hyper-hypernetwork.
        cli.hnet_args(parser, **hh_hnet_args_kw)
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=True,
            show_use_prev_post_as_prior=True, show_num_kl_samples=True,
            show_calc_hnet_reg_targets_online=True,
            show_hnet_reg_batch_size=True)
        ind_posterior_args(train_agroup, show_distill_iter=True,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=True)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.ssge_args(parser)
        prob_args(parser)

        perm_args(parser)

    #############################
    ### PermutedMNIST SSGE-PF ###
    #############################
    elif mode == 'perm_mnist_ssge_pf':
        dout_dir = './out_perm_ssge_pf/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Note, no `beta` in prior-focused CL, since there is not our hnet
        # regularizer.
        tmp_perm_cl_args_kw = dict(perm_cl_args_kw)
        tmp_perm_cl_args_kw['show_beta'] = False
        cl_argroup = cli.cl_args(parser, **tmp_perm_cl_args_kw)
        train_agroup = cli.train_args(parser, **perm_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **perm_main_args_kw)
        # Hypernetwork (weight generator).
        cli.hnet_args(parser, **imp_hnet_args_kw)

        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=False,
            show_use_prev_post_as_prior=False, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_distill_iter=False,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=False)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.ssge_args(parser)
        prob_args(parser)

        perm_args(parser)

    #########################
    ### Resnet CIFAR SSGE ###
    #########################
    elif mode == 'cifar_resnet_ssge':
        dout_dir = './out_resnet_ssge/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, **cifar_cl_args_kw)
        pcta.extra_cl_args(cl_argroup)
        train_agroup = cli.train_args(parser, **resnet_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **resnet_main_args_kw)
        # Hypernetwork (weight generator).
        hnet_agroup = cli.hnet_args(parser, **imp_hnet_args_kw_with_shmlp)
        hnet_args(hnet_agroup, prefix='imp_')
        # Hyper-hypernetwork.
        cli.hnet_args(parser, **hh_hnet_args_kw)
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        cli.data_args(parser, show_disable_data_augmentation=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=True,
            show_use_prev_post_as_prior=True, show_num_kl_samples=True,
            show_calc_hnet_reg_targets_online=True,
            show_hnet_reg_batch_size=True)
        ind_posterior_args(train_agroup, show_distill_iter=True,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=True)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.ssge_args(parser)
        prob_args(parser)

    ############################
    ### Resnet CIFAR SSGE-PF ###
    ############################
    elif mode == 'cifar_resnet_ssge_pf':
        dout_dir = './out_resnet_ssge_pf/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Note, no `beta` in prior-focused CL, since there is not our hnet
        # regularizer.
        tmp_cifar_cl_args_kw = dict(cifar_cl_args_kw)
        tmp_cifar_cl_args_kw['show_beta'] = False
        cl_argroup = cli.cl_args(parser, **tmp_cifar_cl_args_kw)
        pcta.extra_cl_args(cl_argroup)
        train_agroup = cli.train_args(parser, **resnet_train_args_kw)
        # Main network.
        cli.main_net_args(parser, **resnet_main_args_kw)
        # Hypernetwork (weight generator).
        hnet_agroup = cli.hnet_args(parser, **imp_hnet_args_kw_with_shmlp)
        hnet_args(hnet_agroup, prefix='imp_')

        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        cli.data_args(parser, show_disable_data_augmentation=True)
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
            dval_batch_size=1000, show_val_set_size=True, dval_set_size=0)
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=False, show_plots=False, no_cuda=False,
            dout_dir=dout_dir)

        special_train_options(train_agroup, show_soft_targets=False)

        eval_args(eval_agroup)
        rta.mc_args(train_agroup, eval_agroup)
        rta.train_args(train_agroup, show_ll_dist_std=False,
                       show_local_reparam_trick=False, show_kl_scale=True)
        rta.miscellaneous_args(misc_agroup, show_use_logvar_enc=False,
                               show_disable_lrt_test=False,
                               show_during_acc_criterion=True)
        train_args(train_agroup, show_init_with_prev_emb=False,
            show_use_prev_post_as_prior=False, show_num_kl_samples=True)
        ind_posterior_args(train_agroup, show_distill_iter=False,
                           show_final_coresets_finetune=False)
        pcta.miscellaneous_args(misc_agroup, show_no_hhnet=False)
        pcta.imp_args(parser, dlatent_dim=8)
        pcta.ssge_args(parser)
        prob_args(parser)

    ##################################
    ### Finish-up Argument Parsing ###
    ##################################
    # Including all necessary sanity checks.

    args = None
    if argv is not None:
        if default:
            warnings.warn('Provided "argv" will be ignored since "default" ' +
                          'option was turned on.')
        args = argv
    if default:
        args = []
    config = parser.parse_args(args=args)

    if not hasattr(config, "distill_iter"):
        # Distillation makes no sense in prior-focused CL, since we want to come
        # up with one posterior for all tasks.
        assert mode.endswith('pf')
        config.distill_iter = -1

    if not hasattr(config, "beta"):
        # No hnet output regularizer for prior-focused methods.
        assert mode.endswith('pf')
        config.beta = 0

    if mode.endswith('pf') and config.cl_scenario == 2 and \
            config.coreset_size != -1:
        raise ValueError('Coreset usage doesn\'t make sense when training a ' +
                         'prior-focused method (single posterior across all ' +
                         'tasks) on a single-head network.')

    if hasattr(config, 'train_from_scratch') and hasattr(config, 'beta') and \
            config.train_from_scratch and config.beta > 0:
        warnings.warn('The hypernet regularizer will not be used when ' +
                      'training from scratch. "beta" is set to 0.')
        config.beta = 0

    ### Check argument values!
    cli.check_invalid_argument_usage(config)

    rta.check_invalid_args_general(config)
    check_invalid_args_general(config)

    if config.plateau_lr_scheduler and (not hasattr(config, 'val_set_size') or \
                                        config.val_set_size <= 0):
        warnings.warn('Test set will be used for plateau lr scheduler, since ' +
                      'no validation set exists!')

    if config.plateau_lr_scheduler and config.epochs == -1:
        raise ValueError('Flag "plateau_lr_scheduler" can only be used if ' +
                         '"epochs" was set.')
    if config.lambda_lr_scheduler and config.epochs == -1:
        raise ValueError('Flag "lambda_lr_scheduler" can only be used if ' +
                         '"epochs" was set.')

    if hasattr(config, 'num_classes_per_task') and \
            config.num_classes_per_task < 2:
        raise ValueError('Each task needs to have at least 2 classes!')

    if mode.startswith('split'):
        if config.num_classes_per_task != 2:
            warnings.warn('SplitMNIST typically has 2 classes per task. ' +
                          'Running an experiment with %d classes per task ...' %
                          config.num_classes_per_task)

    if mode.startswith('cifar'):
        if config.num_classes_per_task != 10:
            warnings.warn('SplitCIFAR typically has 10 classes per task. ' +
                          'Running an experiment with %d classes per task ...' %
                          config.num_classes_per_task)

    if 'bbb' in mode:
        rta.check_invalid_bbb_args(config)

    return config

def perm_args(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    an argument group for special options regarding the Permuted MNIST dataset.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### Permuted MNIST Options
    agroup = parser.add_argument_group('Permuted MNIST Options')
    agroup.add_argument('--padding', type=int, default=2,
                        help='Padding the images with zeros for the ' +
                             'permutation experiments. This is done to ' +
                             'relate to results from ' +
                             'arxiv.org/pdf/1809.10635.pdf. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--trgt_padding', type=int, default=0,
                        help='Pad target 1-hot encodings of each task with ' +
                             'the given amount of 0s, to increase the ' +
                             'softmax size.')

    return agroup


def train_args(tgroup, show_calc_hnet_reg_targets_online=False,
               show_hnet_reg_batch_size=False, show_init_with_prev_emb=True,
               show_use_prev_post_as_prior=True, show_kl_schedule=True,
               show_num_kl_samples=False, show_training_set_size=True):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the training argument group specific to training
    probabilistic models.

    Args:
        tgroup: The argument group returned by function
            :func:`utils.cli_args.train_args`.
        show_calc_hnet_reg_targets_online (bool): Whether the option
            `calc_hnet_reg_targets_online` should be provided.
        show_hnet_reg_batch_size (bool): Whether the option
            `hnet_reg_batch_size` should be provided.
        show_init_with_prev_emb (bool): Whether the option
            `init_with_prev_emb` should be provided.
        show_use_prev_post_as_prior (bool): Whether the option
            `use_prev_post_as_prior` should be provided.
        show_kl_schedule (bool): Whether the option `show_kl_schedule` should be
            provided.
        show_num_kl_samples (bool): Whether the option
            `num_kl_samples` should be provided.
        show_training_set_size (bool): Whether the option
            `training_set_size` should be provided.
    """
    if show_calc_hnet_reg_targets_online:
        tgroup.add_argument('--calc_hnet_reg_targets_online',
                            action='store_true',
                            help='For our hypernet CL regularizer, this ' +
                                 'option will ensure that the targets are ' +
                                 'computed on the fly, using the hypernet ' +
                                 'weights acquired after learning the ' +
                                 'previous task. Note, this option ensures ' +
                                 'that there is almost no memory grow with ' +
                                 'an increasing number of tasks (except ' +
                                 'from an increasing number of task ' +
                                 'embeddings). If this option is ' +
                                 'deactivated, the more computationally ' +
                                 'efficient way is chosen of computing all ' +
                                 'main network weight targets (from all ' +
                                 'previous tasks) ones before learning a new ' +
                                 'task.')
    if show_hnet_reg_batch_size:
        tgroup.add_argument('--hnet_reg_batch_size', type=int, default=-1,
                            metavar='N',
                            help='If not "-1", then this number will ' +
                                 'determine the maximum number of previous ' +
                                 'tasks that are are considered when ' +
                                 'computing the regularizer. Hence, if the ' +
                                 'number of previous tasks is greater than ' 
                                 'this number, then the regularizer will be ' +
                                 'computed only over a random subset of ' +
                                 'previous tasks. Default: %(default)s.')
    if show_init_with_prev_emb:
        tgroup.add_argument('--init_with_prev_emb', action='store_true',
                            help='Initialize embeddings of new tasks with ' +
                                 'the embedding of the most recent task.')
    if show_use_prev_post_as_prior:
        tgroup.add_argument('--use_prev_post_as_prior', action='store_true',
                            help='Use the previous posterior as prior when ' +
                                 'training a new task.')
    if show_kl_schedule:
        tgroup.add_argument('--kl_schedule', type=int, metavar='N', default=0,
                            help='If not "0" the KL term will undergo a burn-' +
                                 'in  (or annealing) phase, i.e., the ' +
                                 'influence of the prior-matching will be ' +
                                 'incrementally  increased (or decreased) ' +
                                 'until it reaches its final strength of "1" ' +
                                 '(or "kl_scale"). This option determines the' +
                                 'number of steps taken to linearly increase ' +
                                 'the KL strength from an initial value ' +
                                 'determined by option "kl_scale"  to the ' +
                                 'final value "1". If the provided number ' +
                                 'is negative, then an annealing schedule ' +
                                 'from 1 "kl_scale" is followed. ' +
                                 'Default: %(default)s.')
    if show_num_kl_samples:
        tgroup.add_argument('--num_kl_samples', type=int, metavar='N',
                            default=1,
                            help='If the VI prior-matching term cannot be ' +
                                 'computed analytically, it needs to be ' +
                                 'approximated via a Monte-Carlo estimate. ' +
                                 'The number of samples used for this ' +
                                 'estimate are determined by this option. ' +
                                 'Default: %(default)s.')

    if show_training_set_size:
        tgroup.add_argument('--training_set_size', type=int, default=-1,
                            metavar='N',
                            help='If not "-1", then the training set size ' +
                                 'per task will be clipped to the given ' +
                                 'value. This can be useful when assessing ' +
                                 'Bayesian methods in the limit of very ' +
                                 'small datasets (where model uncertainty ' +
                                 'should increase). Default: %(default)s.')

def ind_posterior_args(tgroup, show_distill_iter=True,
                       show_final_coresets_finetune=True,
                       show_experience_replay=False):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the training argument group that can be used to enforce
    a stronger independence of the posteriors learned sequentially (i.e., to
    actively counteract forward transfer, which may be hurtful if it causes low
    uncertainty on samples from previous tasks).
    
    Args:
        tgroup: The argument group returned by function
            :func:`utils.cli_args.train_args`.
        show_distill_iter (bool): Whether the option
            `distill_iter` should be provided.
        show_final_coresets_finetune (bool): Whether the option
            `final_coresets_finetune` should be provided.
        show_experience_replay (bool): Whether the option
            `coresets_for_experience_replay` should be provided.
    """
    if show_distill_iter:
        tgroup.add_argument('--distill_iter', type=int, metavar='N', default=-1,
                            help='If not "-1" then a separate main network ' +
                                 'per task will be trained and the obtained ' +
                                 'weights will be distilled into the ' +
                                 'hypernetwork afterwards. The length of ' +
                                 'this distillation phase (in terms of ' +
                                 'iterations) is determined by this ' +
                                 'argument. Default: %(default)s.')

    tgroup.add_argument('--coreset_size', type=int, metavar='N', default=-1,
                        help='If not "-1" then a coreset of the given size ' +
                             'is maintained in order to regularize for high ' +
                             'predictive uncertainty on previous tasks when ' +
                             'training a posterior on the current task. ' +
                             'This option determines the size of the coreset ' +
                             'for all previous tasks, except option ' +
                             '"per_task_coreset" is set.' +
                             'Default: %(default)s.')
    tgroup.add_argument('--per_task_coreset', action='store_true',
                        help='Store a coreset of size "coreset_size" for ' +
                             'each previous task.')
    tgroup.add_argument('--coreset_reg', type=float, default=1.,
                        help='If "coreset_size" is set, then this option ' +
                             'determines the regularization strength of the ' +
                             'coreset regularizer. Default: %(default)s.')
    tgroup.add_argument('--coreset_batch_size', type=int, default=-1,
                        help='If "coreset_size" is set, then this option ' +
                             'determines the size of the coreset batches. ' +
                             'If the value is not provided, the same ' +
                             'batch size as for the standard training set ' +
                             'will be used. Default: %(default)s.')
    if show_experience_replay:
        tgroup.add_argument('--coresets_for_experience_replay',
                            action='store_true',
                            help='If this option is activated, the coresets ' +
                                 'will be used for experience replay. This ' +
                                 'option is only compatible with ' +
                                 'deterministic runs. The options ' +
                                 '"coreset_size", "per_task_coreset" and ' +
                                 '"coreset_reg" are reused in this context.')
        tgroup.add_argument('--fix_coreset_size', action='store_true',
                            help='If this option is activated, the total ' +
                                 'number of coreset samples used per ' +
                                 'iteration (i.e. summed across tasks) will ' +
                                 'not increase with the number of tasks. ' +
                                 'Instead, "coreset_batch_size" is divided ' +
                                 'by the number of tasks each time. ' +
                                 'Only compatible with option ' +
                                 '"coresets_for_experience_replay".')
    tgroup.add_argument('--past_and_future_coresets', action='store_true',
                        help='If this option is activated, the coreset ' +
                             'regularizer will operate on data from past and ' +
                             'future tasks. Note, this violates continual ' +
                             'learning as we have access to a coreset from ' +
                             'a future task. The option is currently only ' +
                             'implemented in combination with ' +
                             '"per_task_coreset".')
    if show_final_coresets_finetune:
        tgroup.add_argument('--final_coresets_finetune', action='store_true',
                            help='If this option is activated, the coresets ' +
                                 'will not be used for training, but instead ' +
                                 'kept aside for fine-tuning in a multitask ' +
                                 'fashion after training on all tasks.')
        tgroup.add_argument('--final_coresets_single_task',
                            action='store_true',
                            help='If "final_coresets_finetune" is set, this ' +
                                 'option indicates that the coreset of a ' +
                                 'given task will be used to fine-tune only ' +
                                 'the corresponding solution. Else, all ' +
                                 'coresets are used to fine-tune a task-' +
                                 'specific solution by enforcing high ' +
                                 'uncertainty with coresets from other tasks.')
        tgroup.add_argument('--final_coresets_use_random_labels', 
                            action='store_true',
                            help='If "final_coresets_finetune" is set, and ' +
                                 '"final_coresets_single_head" is not active, '+
                                 'this option specifies that the coresets ' +
                                 'with inputs that do not correspond to the ' +
                                 'task being fine-tuned will be trained with ' +
                                 'random outputs, to incentivize model ' +
                                 'disagreement. Else, labels with maximum ' +
                                 'entropy will be used.')
        tgroup.add_argument('--final_coresets_kl_scale', type=float, default=-1,
                            help='If "final_coresets_finetune" is set, then ' +
                                 'this option determines the scale to be ' +
                                 'applied for the prior-matching term in ' +
                                 'the fine-tuning stage. For values of "-1" ' +
                                 'the value for "kl_scale" will be used. ' +
                                 'Default: %(default)s.')
        tgroup.add_argument('--final_coresets_n_iter', type=int, default=-1,
                            help='If "final_coresets_finetune" is set, then ' +
                                 'this option determines the number of ' +
                                 'iterations used for the final fine-tuning ' +
                                 'using the coresets. For values of "-1" ' +
                                 'the value for "n_iter" will be used. ' +
                                 'Default: %(default)s.')
        tgroup.add_argument('--final_coresets_epochs', type=int, default=-1,
                            help='If "final_coresets_finetune" is set, then ' +
                                 'this option determines the number of ' +
                                 'epochs used for the final fine-tuning ' +
                                 'using the coresets. For values of "-1" ' +
                                 'the value for "epochs" will be used. ' +
                                 'Default: %(default)s.')
        tgroup.add_argument('--final_coresets_balance', type=float, default=-1,
                            help='If "final_coresets_finetune" is set and ' +
                                 'not "final_coresets_single_task", then ' +
                                 'this option determines the sample balance ' +
                                 'within a mini-batch. "-1" refers to 1 / ' +
                                 'num_tasks, i.e., all tasks are equally ' +
                                 'balanced. If a higher number is chosen, ' +
                                 'then the actual task has more weight than ' +
                                 'other tasks and vice versa. ' +
                                 'Default: %(default)s.')

def prob_args(parser):
    """This is a helper function of the function
    :func:`parse_cmd_arguments` to add an argument group for options specific
    to probabilistic continual learning.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
    """
    ### Probabilistic Continual Learning Options.
    agroup = parser.add_argument_group('Probabilistic CL Options')
    agroup.add_argument('--calibrate_temp', action='store_true',
                        help='After training, the softmax temperature will ' +
                             'be calibrated using a proper score function on ' +
                             'the training set (note during training, the ' +
                             'softmax outputs were (in general) not ' +
                             'calibrated correctly as the loss is a ' +
                             'combination of loss terms including ' +
                             'regularizers that change as more tasks arrive.')
    agroup.add_argument('--cal_temp_iter', type=int, metavar='N', default=1000,
                        help='Number of iterations for calibrating the ' +
                             'temperature. Default: %(default)s.')
    agroup.add_argument('--cal_sample_size', type=int, metavar='N', default=-1,
                        help='Number of weight samples to be drawn during ' +
                             'temperature calibration. If "-1", then option ' +
                             '"train_sample_size" will be used. ' +
                             'Default: %(default)s.')

def eval_args(egroup, show_supsup_task_inference=False):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the evaluation argument group.

    Args:
        egroup: The argument group returned by function
            :func:`utils.cli_args.eval_args`.
        show_supsup_task_inference (bool): Whether the option
            `supsup_task_inference` should be provided.
    """
    egroup.add_argument('--full_test_interval', type=int, metavar='N',
                        default=-1,
                        help='Full testing (on all tasks trained so far) is ' +
                             'always invoked after training on each task. ' +
                             'To reduce this demanding computation (and only ' +
                             'always test on the task just trained), one can ' +
                             'specify an interval that determines after how ' +
                             'many trained tasks the full testing is ' +
                             'performed. Note, full testing is always ' +
                             'performed after training the last task. ' +
                             'Default: %(default)s.')
    if show_supsup_task_inference:
        egroup.add_argument('--supsup_task_inference', action='store_true',
                            help='If activated, gradient-based task ' +
                                 'inference as in the SupSup method will be ' +
                                 'computed alongside all other methods to ' +
                                 'perform task-inference.')
        egroup.add_argument('--supsup_grad_steps', type=int, default=1,
                        help='Number of entropy gradient steps to be used ' +
                             'for performing SupSup-like task inference. ' +
                             'Default: %(default)s')
        egroup.add_argument('--supsup_lr', type=float, default=1e-3,
                        help='The scaling for the update of the alpha ' +
                             'coefficients when doing SupSup task-inference. ' +
                             'Only relevant if the number of gradient steps ' +
                             'is larger than 1. Default: %(default)s')

def hnet_args(agroup, prefix=None):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    arguments to the argument group that characterizes hypernetworks.

    Args:
        agroup: The argument group returned by function
            :func:`utils.cli_args.hnet_args`.

    Returns:
        The modified argument group ``agroup``.
    """
    if prefix is None:
        prefix = ''
    p = prefix

    agroup.add_argument('--%sshmlp_gcd_chunking' % p, action='store_true',
                        help='Only applicable if hnet-type "structured_hmlp" ' +
                             'is used. If activated: output chunks will be ' +
                             'reduced in size and therewith more chunks will ' +
                             'have to be generated.')
    return agroup

def special_train_options(agroup, show_soft_targets=True):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the `training` argument group.

    Args:
        agroup: The argument group returned by method
            :func:`utils.cli_args.train_args`.
    """
    agroup.add_argument('--plateau_lr_scheduler', action='store_true',
                        help='Will enable the usage of the learning rate ' +
                             'scheduler torch.optim.lr_scheduler.' +
                             'ReduceLROnPlateau. Note, this option requires ' +
                             'that the argument "epochs" has been set.')
    agroup.add_argument('--lambda_lr_scheduler', action='store_true',
                        help='Will enable the usage of the learning rate ' +
                             'scheduler torch.optim.lr_scheduler.' +
                             'LambdaLR. Note, this option requires ' +
                             'that the argument "epochs" has been set. ' +
                             'The scheduler will behave as specified by ' +
                             'the function "lr_schedule" in ' +
                             'https://keras.io/examples/cifar10_resnet/.')
    if show_soft_targets:
        agroup.add_argument('--soft_targets', action='store_true',
                            help='Use soft targets for classification.')

def check_invalid_args_general(config):
    """Sanity check for some general command-line arguments.

    Args:
        config: Parsed command-line arguments.
    """
    # Not mathematically correct, but might be required if prior is not
    # appropriate.
    if hasattr(config, 'distill_iter') and  config.distill_iter != -1:
        if config.mnet_only:
            raise ValueError('Argument "mnet_only" can\'t be used when ' +
                             'using distillation.')

    if hasattr(config, 'coreset_size'):
        if config.coreset_size == -1 and config.per_task_coreset:
            warnings.warn('Option "per_task_coreset" only has an effect if ' +
                          '"coreset_size" is specified.')
        if config.coreset_size == -1 and config.past_and_future_coresets:
            warnings.warn('Option "past_and_future_coresets" only has an ' +
                          'effect if "coreset_size" is specified.')
        if config.coreset_size != -1 and config.past_and_future_coresets and \
                not config.per_task_coreset:
            warnings.warn('Option "past_and_future_coresets" can currently ' +
                          'only be used in combination with ' +
                          '"per_task_coreset", which will be automatically ' +
                          'set.')
            config.per_task_coreset = True
        if config.coreset_size != -1 and not config.coreset_size > 0:
            raise ValueError('Invalid "coreset_size" value.')
        if hasattr(config, 'coreset_batch_size') and \
                config.coreset_batch_size == -1:
            config.coreset_batch_size = config.batch_size
        if hasattr(config, 'coresets_for_experience_replay') and \
                    config.coresets_for_experience_replay:
            if config.final_coresets_finetune:
                raise ValueError('Option "coresets_for_experience_replay" ' +
                                 'cannot be used in combination with ' +
                                 '"final_coresets_finetune".')
            if not config.per_task_coreset:
                warnings.warn('Option "coresets_for_experience_replay" only ' +
                              'implemented for "per_task_coreset". ' +
                              'Setting this option to `True`.')
                config.per_task_coreset = True
            if not config.mnet_only:
                raise ValueError('Option "coresets_for_experience_replay" ' +
                                 'cannot be used in combination with ' +
                                 'a hypernetwork. Please select ' +
                                 '"mnet_only".')
            if not config.mean_only:
                raise ValueError('Option "coresets_for_experience_replay" ' +
                                 'cannot be used in combination with ' +
                                 'a stochastic main network. Please select ' +
                                 '"mean_only".')
        if hasattr(config, 'final_coresets_finetune'):
            if config.coreset_size == -1 and config.final_coresets_finetune:
                warnings.warn('Option "final_coresets_finetune" only has an ' +
                              'effect if "coreset_size" is specified.')
            if not config.final_coresets_finetune:
                if config.final_coresets_single_task:
                    warnings.warn('Option "final_coresets_single_task" has no '+
                                  'effect unless "final_coresets_finetune" is '+
                                  'active.')
                if config.final_coresets_use_random_labels:
                    warnings.warn('Option "final_coresets_use_random_labels" ' +
                                  'has no effect unless ' +
                                  '"final_coresets_finetune" is active.')
                if config.final_coresets_kl_scale != -1:
                    warnings.warn('The parameter "final_coresets_kl_scale" ' +
                                  'has no effect unless ' +
                                  '"final_coresets_finetune" is active.')
                if config.final_coresets_n_iter != -1:
                    warnings.warn('The parameter "final_coresets_n_iter" ' +
                                  'has no effect unless ' +
                                  '"final_coresets_finetune" is active.')
                if config.final_coresets_epochs != -1:
                    warnings.warn('The parameter "final_coresets_epochs" ' +
                                  'has no effect unless ' +
                                  '"final_coresets_finetune" is active.')
            else:
                if config.past_and_future_coresets:
                    warnings.warn('Option "past_and_future_coresets" has no ' +
                                  'effect when "final_coresets_finetune" is ' +
                                  'active.')
                if not config.per_task_coreset:
                    warnings.warn('Option "final_coresets_finetune" has to ' +
                                  'be applied with "per_task_coreset". ' +
                                  'Overwriting it.')
                    config.per_task_coreset = True

            if config.final_coresets_finetune:
                if hasattr(config, 'radial_bnn') and config.radial_bnn:
                    raise NotImplementedError()
                if hasattr(config, 'mean_only') and config.mean_only:
                    raise ValueError('Finetuning not applicable to ' +
                                     'deterministic network.')
                if config.final_coresets_balance != -1:
                    if config.final_coresets_balance < 0 or \
                            config.final_coresets_balance > 1.:
                        raise ValueError('"final_coresets_balance" must be ' +
                                         'between 0 and 1.')

    if hasattr(config, 'dis_batch_size') and config.dis_batch_size == 1 and \
            config.use_batchstats:
        warnings.warn('Using batch statistics for discriminator training ' +
                      'doesn\'t make sense when using a "dis_batch_size" of 1.')

    if hasattr(config, 'thr_ssge_eigenvals'):
        if config.thr_ssge_eigenvals != 1.0 and \
                config.num_ssge_eigenvals != -1:
            raise ValueError('When using SSGE, the number of eigenvalues ' +
                'to be used for the estimation can only be determined either ' +
                'by setting a number, or a threshold for the percentage of ' +
                'eigenvalues used, but not both.')

    if hasattr(config, 'cl_scenario') \
            and hasattr(config, 'non_growing_sf_cl3') \
            and config.cl_scenario != 3 and config.non_growing_sf_cl3:
        raise ValueError('Option "non_growing_sf_cl3" only for CL3!')

if __name__ == '__main__':
    pass


