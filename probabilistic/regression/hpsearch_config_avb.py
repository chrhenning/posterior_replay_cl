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
# title          :hpsearch_config_avb.py
# author         :ch
# contact        :henningc@ethz.ch
# created        :11/04/2020
# version        :1.0
# python_version :3.6.8
"""
A configuration file for our custom hyperparameter search script. This
configuration is meant for hyperparameter searches of the simulation defined by
:mod:`probabilistic.regression.train_avb`.

Checkout the documentation of :mod:`hpsearch.hpsearch_config_template` for
more information on this files content.
"""

from probabilistic.regression import hpsearch_config_bbb as hpbbb

##########################################
### Please define all parameters below ###
##########################################

grid = {
    ### Continual learning options ###
    'beta' : [0.005],
    #'train_from_scratch' : [False],
    #'multi_head' : [False],

    ### Training options ###
    #'batch_size' : [32],
    #'n_iter' : [5001],
    #'epochs' : [-1],
    #'lr' : [0.001],
    #'momentum' : [0],
    #'weight_decay' : [0],
    'use_adam' : [True],
    #'adam_beta1' : [0.9],
    #'use_rmsprop' : [False],
    #'use_adadelta' : [False],
    #'use_adagrad' : [False],
    #'clip_grad_value' : [-1],
    #'clip_grad_norm' : [-1],
    #'plateau_lr_scheduler': [False],
    #'lambda_lr_scheduler': [False],
    #'train_sample_size' : [10],
    #'prior_variance' : [1.],
    #'ll_dist_std' : [.1],
    #'kl_scale' : [1.],
    #'calc_hnet_reg_targets_online': [False],
    #'hnet_reg_batch_size': [-1],
    #'init_with_prev_emb': [False],
    #'use_prev_post_as_prior': [False],
    #'kl_schedule': [0],
    #'num_kl_samples': [1],
    #'coreset_size': [-1],
    #'per_task_coreset': [False],
    #'coreset_reg': [1.],
    #'past_and_future_coresets': [False],

    ### Main network options ###
    #'mlp_arch' : ['"10,10"'],
    #'net_act' : ['relu'],
    #'dropout_rate' : [-1],
    #'batchnorm' : [False],
    #'specnorm' : [False],
    #'no_bias' : [False],

    ### Discriminator options ###
    #'dis_net_type' : ['mlp'],
    #'dis_mlp_arch' : ['"10,10"'],
    #'dis_cmlp_arch' : ['"10,10"'],
    #'dis_cmlp_chunk_arch' : ['"10,10"'],
    #'dis_cmlp_in_cdim' : [32],
    #'dis_cmlp_out_cdim' : [8],
    #'dis_cmlp_cemb_dim' : [8],
    #'dis_net_act' : ['sigmoid'],
    #'dis_dropout_rate' : [-1],
    #'dis_batchnorm' : [False],
    #'dis_specnorm' : [False],
    #'dis_no_bias' : [False],

    ### Implicit-hypernet options ###
    'imp_hnet_type' : ['hmlp'], # 'hmlp', 'chunked_hmlp', 'structured_hmlp',
                                # 'hdeconv', 'chunked_hdeconv'
    #'imp_hmlp_arch' : ['"10,10"'],
    #'imp_chmlp_chunk_size' : [64],
    #'imp_chunk_emb_size' : ['"8"'],
    #'imp_hdeconv_shape' : ['"512,512,3"'],
    #'imp_hdeconv_num_layers' : [5],
    #'imp_hdeconv_filters' : ['"128,512,256,128"'],
    #'imp_hdeconv_kernels': ['"5"'],
    #'imp_hdeconv_attention_layers': ['"1,3"'],
    #'imp_hnet_net_act': ['sigmoid'],
    #'imp_hnet_no_bias': [False],
    #'imp_hnet_dropout_rate': [-1],
    #'imp_hnet_specnorm': [False],

    ### Hyper-hypernet options ###
    'hh_hnet_type' : ['hmlp'], # 'hmlp', 'chunked_hmlp', 'structured_hmlp',
                               # 'hdeconv', 'chunked_hdeconv'
    #'hh_hmlp_arch' : ['"10,10"'],
    #'hh_cond_emb_size' : [2],
    #'hh_chmlp_chunk_size' : [64],
    #'hh_chunk_emb_size' : ['"8"'],
    #'hh_use_cond_chunk_embs' : [False],
    #'hh_hdeconv_shape' : ['"512,512,3"'],
    #'hh_hdeconv_num_layers' : [5],
    #'hh_hdeconv_filters' : ['"128,512,256,128"'],
    #'hh_hdeconv_kernels': ['"5"'],
    #'hh_hdeconv_attention_layers': ['"1,3"'],
    #'hh_hnet_net_act': ['sigmoid'],
    #'hh_hnet_no_bias': [False],
    #'hh_hnet_dropout_rate': [-1],
    #'hh_hnet_specnorm': [False],

    ### Network initialization options ###
    #'normal_init' : [False],
    #'std_normal_init' : [0.02],
    #'std_normal_temb' : [1.],
    #'std_normal_emb' : [1.],
    #'hyper_fan_init' : [False],

    ### Evaluation options ###
    #'val_iter' : [250],
    #'val_sample_size' : [100],

    ### Dataset options ###
    'used_task_set' : [1],

    ### Miscellaneous options ###
    'use_cuda' : [True],
    #'deterministic_run': [False],
    #'random_seed': [42],
    #'data_random_seed': [42],
    #'mnet_only': [False],
    #'no_hhnet': [False],
    #'no_dis': [False],

    ### Implicit Distribution options ###
    #'latent_dim' : [8],
    #'latent_std' : [1.],
    'prior_focused' : [False],
    'full_support_perturbation' : [-1],

    ### AVB options ###
    #'dis_lr' : [-1.],
    #'dis_batch_size' : [1],
    #'num_dis_steps' : [1],
    #'no_dis_reinit' : [False],
    #'use_batchstats' : [False],
    #'no_adaptive_contrast' : [False],
    #'num_ac_samples' : [100],
}

conditions = [
    # TODO
]

####################################
### DO NOT CHANGE THE CODE BELOW ###
####################################
_SCRIPT_NAME = 'train_avb.py'
_SUMMARY_FILENAME = hpbbb._SUMMARY_FILENAME
_SUMMARY_KEYWORDS = [
    # The weird prefix "aa_" makes sure keywords appear first in the result csv.
    'aa_mse_during',
    'aa_mse_final',
    'aa_mse_during_mean',
    'aa_mse_final_mean',

    # Final task inference accuracies per task.
    'aa_task_inference',
    'aa_task_inference_mean',

    # If task identity has been inferred.
    'aa_mse_during_inferred',
    'aa_mse_final_inferred',
    'aa_mse_during_inferred_mean',
    'aa_mse_final_inferred_mean',
    
    # Discriminator accuracies.
    'aa_acc_avg_dis',
    'aa_acc_dis',

    'aa_num_weights_main',
    'aa_num_weights_hyper',
    'aa_num_weights_hyper_hyper',
    'aa_num_weights_dis',
    'aa_num_weights_hm_ratio', # Hypernet / Main
    'aa_num_weights_hhm_ratio', # Hyper-hypernet / Main
    'aa_num_weights_dm_ratio', # Discriminator / Main

    # Should be set in your program when the execution finished successfully.
    'finished'
]
_OUT_ARG = hpbbb._OUT_ARG
_SUMMARY_PARSER_HANDLE = hpbbb._SUMMARY_PARSER_HANDLE
_PERFORMANCE_EVAL_HANDLE = hpbbb._PERFORMANCE_EVAL_HANDLE

_PERFORMANCE_KEY = hpbbb._PERFORMANCE_KEY
assert(_PERFORMANCE_KEY is None or _PERFORMANCE_KEY in _SUMMARY_KEYWORDS)
# Whether the CSV should be sorted ascending or descending based on the
# `_PERFORMANCE_KEY`.
_PERFORMANCE_SORT_ASC = hpbbb._PERFORMANCE_SORT_ASC

# FIXME: This attribute will vanish in future releases.
# This attribute is only required by the `hpsearch_postprocessing` script.
# A function handle to the argument parser function used by the simulation
# script. The function handle should expect the list of command line options
# as only parameter.
# Example:
# >>> from probabilistic.prob_mnist import train_args as targs
# >>> f = lambda argv : targs.parse_cmd_arguments(mode='split_mnist_bbb',
# ...                                             argv=argv)
# >>> _ARGPARSE_HANDLE = f
import probabilistic.regression.train_args as targs
_ARGPARSE_HANDLE = lambda argv : targs.parse_cmd_arguments( \
    mode='regression_avb', argv=argv)

if __name__ == '__main__':
    pass


