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
# title          :probabilistic/prob_mnist/hpsearch_config_split_avb.py
# author         :ch
# contact        :henningc@ethz.ch
# created        :02/03/2020
# version        :1.0
# python_version :3.6.8
"""
Hyperparameter-search configuration for SplitMNIST using AVB
------------------------------------------------------------

A configuration file for our custom hyperparameter search script. This
configuration is meant for hyperparameter searches of the simulation defined by
:mod:`probabilistic.prob_mnist.train_split_avb`.

Note, to be comparable with
    van de Ven et al., "Three scenarios for continual learning", 2019,
    https://arxiv.org/abs/1904.07734
we will mark arguments that should not be changed from its default value by
"RELATED WORK".
"""

from probabilistic.prob_mnist import hpsearch_config_split_bbb as hpsplit

##########################################
### Please define all parameters below ###
##########################################

# Define a dictionary with parameter names as keys and a list of values for
# each parameter. For flag arguments, simply use the values [True, False].
# Note, the output directory is set by the hyperparameter search script.
#
# Example: {'option1': [12, 24], 'option2': [0.1, 0.5],
#           'option3': [True]}
# This disctionary would correspond to the following 4 configurations:
#   python3 SCRIPT_NAME.py --option1=12 --option2=0.1 --option3
#   python3 SCRIPT_NAME.py --option1=12 --option2=0.5 --option3
#   python3 SCRIPT_NAME.py --option1=24 --option2=0.1 --option3
#   python3 SCRIPT_NAME.py --option1=24 --option2=0.5 --option3
#
# If fields are commented out (missing), the default value is used.
# Note, that you can specify special conditions below.

grid = {
    ### Continual learning options ###
    'beta' : [0.01],
    #'train_from_scratch' : [False],
    #'cl_scenario' : [1], # 1, 2 or 3
    #'split_head_cl3' : [False],
    #'num_tasks' : [5],
    #'num_classes_per_task': [2],

    ### Training options ###
    #'batch_size' : [128], # RELATED WORK - 128
    #'n_iter' : [2000], # RELATED WORK - 2000
    #'epochs' : [-1],
    #'lr' : [0.001], # RELATED WORK - 0.001
    #'momentum' : [0.],
    #'weight_decay' : [0], # RELATED WORK - 0.
    'use_adam' : [True], # RELATED WORK - True
    #'adam_beta1' : [0.9], # RELATED WORK - 0.9
    #'use_rmsprop' : [False],
    #'use_adadelta' : [False],
    #'use_adagrad' : [False],
    #'clip_grad_value' : [-1],
    #'clip_grad_norm' : [-1],
    #'plateau_lr_scheduler': [False],
    #'lambda_lr_scheduler': [False],
    #'train_sample_size' : [1],
    #'prior_variance' : [1.],
    #'kl_scale' : [1.],
    #'init_with_prev_emb': [False],
    #'use_prev_post_as_prior': [False],
    #'kl_schedule': [0],
    #'num_kl_samples': [1],
    #'training_set_size': [-1],
    #'distill_iter': [-1],
    #'coreset_size': [-1],
    #'per_task_coreset': [False],
    #'coreset_reg': [1.],
    #'coreset_batch_size': [-1],
    #'past_and_future_coresets': [False],
    #'calc_hnet_reg_targets_online': [False],
    #'hnet_reg_batch_size': [-1],

    ### Main network options ###
    #'net_type' : ['mlp'], # RELATED WORK - 'mlp'
    #'mlp_arch' : ['"400,400"'], # RELATED WORK - '"400,400"'
    #'lenet_type' : ['mnist_small'], # 'mnist_small', 'mnist_large'
    #'resnet_block_depth': [5],
    #'resnet_channel_sizes': ['"16,16,32,64"'],
    #'net_act' : ['relu'], # RELATED WORK - 'relu'
    #'no_bias' : [False],
    #'dropout_rate' : [-1],
    #'specnorm' : [False],
    #'batchnorm' : [False],
    #'bn_no_running_stats': [False],
    #'bn_no_stats_checkpointing': [False],

    ### Discriminator options ###
    #'dis_net_type' : ['mlp'],
    #'dis_mlp_arch' : ['"100,100"'],
    #'dis_cmlp_arch' : ['"10,10"'],
    #'dis_cmlp_chunk_arch' : ['"10,10"'],
    #'dis_cmlp_in_cdim' : [100],
    #'dis_cmlp_out_cdim' : [10],
    #'dis_cmlp_cemb_dim' : [8],
    #'dis_net_act' : ['relu'],
    #'dis_dropout_rate' : [-1],
    #'dis_batchnorm' : [False],
    #'dis_specnorm' : [False],
    #'dis_no_bias' : [False],

    ### Implicit-hypernet options ###
    'imp_hnet_type' : ['hmlp'], # 'hmlp', 'chunked_hmlp', 'structured_hmlp',
                                # 'hdeconv', 'chunked_hdeconv'
    #'imp_hmlp_arch' : ['"125,250,500"'],
    #'imp_chmlp_chunk_size' : [1500],
    #'imp_chunk_emb_size' : ['"32"'],
    #'imp_use_cond_chunk_embs' : [False],
    #'imp_hdeconv_shape' : ['"512,512,3"'],
    #'imp_hdeconv_num_layers' : [5],
    #'imp_hdeconv_filters' : ['"128,512,256,128"'],
    #'imp_hdeconv_kernels': ['"5"'],
    #'imp_hdeconv_attention_layers': ['"1,3"'],
    #'imp_hnet_net_act': ['sigmoid'],
    #'imp_hnet_no_bias': [False],
    #'imp_hnet_dropout_rate': [-1],
    #'imp_hnet_specnorm': [False],
    #'full_support_perturbation': [-1],

    ### Hyper-hypernet options ###
    'hh_hnet_type' : ['hmlp'], # 'hmlp', 'chunked_hmlp', 'structured_hmlp',
                               # 'hdeconv', 'chunked_hdeconv'
    #'hh_hmlp_arch' : ['"125,250,500"'],
    #'hh_cond_emb_size' : [32],
    #'hh_chmlp_chunk_size' : [1500],
    #'hh_chunk_emb_size' : ['"32"'],
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
    #'val_iter' : [500],
    #'val_batch_size' : [1000],
    #'val_set_size' : [0],
    #'full_test_interval' : [-1],
    #'val_sample_size' : [100],

    ### Miscellaneous options ###
    #'no_cuda' : [False],
    #'deterministic_run': [True],
    #'random_seed': [42],
    #'mnet_only': [False],
    #'during_acc_criterion': ['"-1"'],
    #'no_hhnet': [False],
    #'no_dis': [False],

    ### AVB options ###
    #'latent_dim' : [8],
    #'latent_std' : [1.],
    #'dis_lr' : [-1.],
    #'dis_batch_size' : [1],
    #'num_dis_steps' : [1],
    #'no_dis_reinit' : [False],
    #'use_batchstats' : [False],
    #'no_adaptive_contrast' : [False],
    #'num_ac_samples' : [100],

    ### Probabilistic CL Options ###
    #'calibrate_temp': [False],
    #'cal_temp_iter': [1000],
    #'cal_sample_size': [-1],
}

# Sometimes, not the whole grid should be searched. For instance, if an SGD
# optimizer has been chosen, then it doesn't make sense to search over multiple
# beta2 values of an Adam optimizer.
# Therefore, one can specify special conditions.
# NOTE, all conditions that are specified here will be enforced. Thus, they
# overwrite the grid options above.
#
# How to specify a condition? A condition is a key value tuple: whereas as the
# key as well as the value is a dictionary in the same format as in the grid
# above. If any configurations matches the values specified in the key dict,
# The values specified in the values dict will be searched instead.
#
# Note, if arguments are commented out above but appear in the conditions, the
# condition will be ignored.
conditions = [
    # Note, we specify a particular set of base conditions below that should
    # always be enforces: "_BASE_CONDITIONS".

    ### Add your conditions here ###
    #({'clip_grad_value': [1.]}, {'clip_grad_norm': [-1]}),
    #({'clip_grad_norm': [1.]}, {'clip_grad_value': [-1]}),
]

####################################
### DO NOT CHANGE THE CODE BELOW ###
####################################
# Conditions specific to AVB training.
_AVB_CONDITIONS = [
    ({'no_adaptive_contrast': [True]}, {'num_ac_samples': [-1]}),
    ({'dis_batch_size': [1]}, {'use_batchstats': [False]}),
    # TODO if `no_dis`, then we don't need to explore any discriminator options.
]

conditions = conditions + hpsplit._BASE_CONDITIONS + _AVB_CONDITIONS

# Name of the script that should be executed by the hyperparameter search.
# Note, the working directory is set seperately by the hyperparameter search
# script.
_SCRIPT_NAME = 'train_split_avb.py'

# This file is expected to reside in the output folder of the simulation.
_SUMMARY_FILENAME = 'performance_overview.txt'
# These are the keywords that are supposed to be in the summary file.
# A summary file always has to include the keyword "finished"!.
_SUMMARY_KEYWORDS = [
    'acc_avg_final', # Depends on CL scenario.
    'acc_avg_during', # Depends on CL scenario.

    # Accuracy if task identity given.
    'acc_task_given',
    'acc_task_given_during',
    'acc_avg_task_given',
    'acc_avg_task_given_during',
    # Accuracy if task identity inferred using entropy.
    'acc_task_inferred_ent',
    'acc_task_inferred_ent_during',
    'acc_avg_task_inferred_ent',
    'acc_avg_task_inferred_ent_during',
    # Task-inference accuracy using entropy (how often has the correct
    # head/embedding been chosen).
    'avg_task_inference_acc_ent',
    # Accuracy if task identity inferred using confidence.
    'acc_avg_task_inferred_conf',
    # Task-inference accuracy using confidence.
    'avg_task_inference_acc_conf',
    # Accuracy if task identity inferred using model agreement.
    'acc_avg_task_inferred_agree',
    # Task-inference accuracy using model agreement.
    'avg_task_inference_acc_agree',

    # Discriminator accuracies.
    'acc_avg_dis',
    'acc_dis',

    'num_weights_main',
    'num_weights_hyper',
    'num_weights_hyper_hyper',
    'num_weights_dis',
    'num_weights_hm_ratio', # Hypernet / Main
    'num_weights_hhm_ratio', # Hyper-hypernet / Main
    'num_weights_dm_ratio', # Discriminator / Main

    # Should be set in your program when the execution finished successfully.
    'finished'
]

# The name of the command-line argument that determines the output folder
# of the simulation.
_OUT_ARG = 'out_dir'

# In case you need a more elaborate parser than the default one define by the
# function :func:`hpsearch.hpsearch._get_performance_summary`, you can pass a
# function handle to this attribute.
# Value `None` results in the usage of the default parser.
_SUMMARY_PARSER_HANDLE = None # Default parser is used.
#_SUMMARY_PARSER_HANDLE = _get_performance_summary # Custom parser is used.

# A function handle, that is used to evaluate the performance of a run.
_PERFORMANCE_EVAL_HANDLE =  hpsplit._performance_criteria

# A key that must appear in the `_SUMMARY_KEYWORDS` list. If `None`, the first
# entry in this list will be selected.
# The CSV file will be sorted based on this keyword. See also attribute
# `_PERFORMANCE_SORT_ASC`.
_PERFORMANCE_KEY = 'acc_avg_final'
assert(_PERFORMANCE_KEY is None or _PERFORMANCE_KEY in _SUMMARY_KEYWORDS)
# Whether the CSV should be sorted ascending or descending based on the
# `_PERFORMANCE_KEY`.
_PERFORMANCE_SORT_ASC = False

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
import probabilistic.prob_mnist.train_args as targs
_ARGPARSE_HANDLE = lambda argv : targs.parse_cmd_arguments( \
    mode='split_mnist_avb', argv=argv)

if __name__ == '__main__':
    pass


