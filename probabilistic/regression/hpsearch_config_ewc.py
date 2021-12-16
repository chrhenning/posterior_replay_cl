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
# title          :hpsearch_config_ewc.py
# author         :ch
# contact        :henningc@ethz.ch
# created        :01/12/2021
# version        :1.0
# python_version :3.6.8
"""
A configuration file for our custom hyperparameter search script. This
configuration is meant for hyperparameter searches of the simulation defined by
:mod:`probabilistic.regression.train_ewc`.

Checkout the documentation of :mod:`hpsearch.hpsearch_config_template` for
more information on this files content.
"""

##########################################
### Please define all parameters below ###
##########################################

grid = {
    ### Continual learning options ###
    #'train_from_scratch' : [False],
    'multi_head' : [False],

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
    #'prior_variance' : [1.],
    #'ll_dist_std' : [.1],

    ### Main network options ###
    #'mlp_arch' : ['"10,10"'],
    #'net_act' : ['relu'],
    #'no_bias' : [False],
    #'dropout_rate' : [-1],
    #'batchnorm' : [False],
    #'bn_no_running_stats' : [False],
    #'bn_no_stats_checkpointing' : [False],

    ### Evaluation options ###
    #'val_iter' : [250],
    #'val_sample_size' : [100],

    ### Dataset options ###
    'used_task_set' : [1],

    ### Miscellaneous options ###
    'no_cuda' : [False],
    #'deterministic_run': [False],
    #'random_seed': [42],
    #'data_random_seed': [42],
    #'store_final_model': [False],

    ### EWC options ###
    'ewc_gamma' : [1.],
    'ewc_lambda' : [1.],
    'n_fisher' : [-1],
}

conditions = [
    ### Add your conditions here ###
    # ({'clip_grad_value': [1.]}, {'clip_grad_norm': [-1]}),
    # ({'clip_grad_norm': [1.]}, {'clip_grad_value': [-1]}),
]

####################################
### DO NOT CHANGE THE CODE BELOW ###
####################################
_SCRIPT_NAME = 'train_ewc.py'
_SUMMARY_FILENAME = 'performance_overview.txt'
_SUMMARY_KEYWORDS = [
    # The weird prefix "aa_" makes sure keywords appear first in the result csv.
    'aa_mse_during',
    'aa_mse_final',
    'aa_mse_during_mean',
    'aa_mse_final_mean',

    # Note, task inference with EWC only applies to the multi-head setting.
    # Final task inference accuracies per task.
    'aa_task_inference',
    'aa_task_inference_mean',

    # If task identity has been inferred.
    'aa_mse_during_inferred',
    'aa_mse_final_inferred',
    'aa_mse_during_inferred_mean',
    'aa_mse_final_inferred_mean',

    'aa_num_weights_main',

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
import probabilistic.ewc_args as targs
_ARGPARSE_HANDLE = lambda argv : targs.parse_cmd_arguments( \
    mode='regression_ewc', argv=argv)

if __name__ == '__main__':
    pass


