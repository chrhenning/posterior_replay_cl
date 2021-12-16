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
# title          :hpsearch_config_bbb.py
# author         :ch
# contact        :henningc@ethz.ch
# created        :01/12/2019
# version        :1.0
# python_version :3.6.8
"""
A configuration file for our custom hyperparameter search script. This
configuration is meant for hyperparameter searches of the simulation defined by
:mod:`probabilistic.regression.train_bbb`.

Checkout the documentation of :mod:`hpsearch.hpsearch_config_template` for
more information on this files content.
"""

##########################################
### Please define all parameters below ###
##########################################

grid = {
    ### Continual learning options ###
    'beta' : [0.005],
    #'train_from_scratch' : [False],
    #'multi_head' : [False],
    #'regularizer' : ['mse'], # 'mse', 'fkl', 'rkl', 'w2'
    #'hnet_out_masking': [0],

    ### Training options ###
    #'batch_size' : [32],
    #'n_iter' : [10001],
    #'lr' : [0.01],
    #'weight_decay' : [0],
    #'adam_beta1' : [0.9],
    #'clip_grad_value' : [-1],
    #'clip_grad_norm' : [-1],
    #'train_sample_size' : [1],
    #'prior_variance' : [1.],
    #'ll_dist_std' : [.1],
    #'local_reparam_trick' : [False],
    'radial_bnn' : [False],
    #'use_prev_post_as_prior': [False],
    #'kl_schedule' : [0],
    #'num_kl_samples' : [1],

    ### Main network options ###
    #'mlp_arch' : ['"10,10"'],
    #'net_act' : ['relu'],
    #'dropout_rate' : [-1],
    #'batchnorm' : [False],
    #'specnorm' : [False],
    #'no_bias' : [False],

    ### Hypernet Options ###
    'hnet_type' : ['hmlp'], # 'hmlp', 'chunked_hmlp', 'structured_hmlp',
                            # 'hdeconv', 'chunked_hdeconv'
    #'hmlp_arch' : ['"10,10"'],
    #'cond_emb_size' : [2],
    #'chmlp_chunk_size' : [64],
    #'chunk_emb_size' : ['"8"'],
    #'use_cond_chunk_embs' : [False],
    #'hdeconv_shape' : ['"512,512,3"'],
    #'hdeconv_num_layers' : [5],
    #'hdeconv_filters' : ['"128,512,256,128"'],
    #'hdeconv_kernels': ['"5"'],
    #'hdeconv_attention_layers': ['"1,3"'],
    #'hnet_net_act': ['sigmoid'],
    #'hnet_no_bias': [False],
    #'hnet_dropout_rate': [-1],
    #'hnet_specnorm': [False],

    ### Network initialization options ###
    #'normal_init' : [False],
    #'std_normal_init' : [0.02],
    #'std_normal_temb' : [1.],
    #'std_normal_emb' : [1.],
    #'keep_orig_init' : [False],
    #'hyper_gauss_init' : [False],

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
    #'use_logvar_enc': [False],
    #'disable_lrt_test': [False],
    #'mean_only': [False],
}

conditions = [
    ({'normal_init': [False]}, {'std_normal_init': [-1]}),
    # TODO set all other hypernet options to their default if `mnet_only`.
    ({'mnet_only': [True]}, {'hyper_gauss_init': [False]}),
    ({'local_reparam_trick': [False]}, {'disable_lrt_test': [False]}),

    ### Add your conditions here ###
    # ({'clip_grad_value': [1.]}, {'clip_grad_norm': [-1]}),
    # ({'clip_grad_norm': [1.]}, {'clip_grad_value': [-1]}),
]

####################################
### DO NOT CHANGE THE CODE BELOW ###
####################################
_SCRIPT_NAME = 'train_bbb.py'
_SUMMARY_FILENAME = 'performance_overview.txt'
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

    'aa_num_weights_main',
    'aa_num_weights_hyper',
    'aa_num_weights_ratio',

    # Should be set in your program when the execution finished successfully.
    'finished'
]
_OUT_ARG = 'out_dir'
_SUMMARY_PARSER_HANDLE = None # Default parser is used.

def _performance_criteria(summary_dict, performance_criteria):
    """Evaluate whether a run meets a given performance criteria.

    This function is needed to decide whether the output directory of a run is
    deleted or kept.

    Args:
        summary_dict: The performance summary dictionary as returned by
            :attr:`_SUMMARY_PARSER_HANDLE`.
        performance_criteria (float): The performance criteria. E.g., see
            command-line option `performance_criteria` of script
            :mod:`hpsearch.hpsearch_postprocessing`.

    Returns:
        bool: If :code:`True`, the result folder will be kept as the performance
        criteria is assumed to be met.
    """
    performance = float(summary_dict['aa_mse_final_inferred_mean'][0])
    return performance < performance_criteria

_PERFORMANCE_EVAL_HANDLE = _performance_criteria

_PERFORMANCE_KEY = 'aa_mse_final_inferred_mean'
assert(_PERFORMANCE_KEY is None or _PERFORMANCE_KEY in _SUMMARY_KEYWORDS)
# Whether the CSV should be sorted ascending or descending based on the
# `_PERFORMANCE_KEY`.
_PERFORMANCE_SORT_ASC = True

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
    mode='regression_bbb', argv=argv)

if __name__ == '__main__':
    pass


