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
# title          :probabilistic/prob_mnist/hpsearch_config_split.py
# author         :ch
# contact        :henningc@ethz.ch
# created        :09/03/2021
# version        :1.0
# python_version :3.6.8
"""
Hyperparameter-search configuration for SplitMNIST with Multitask learning
--------------------------------------------------------------------------

A configuration file for our custom hyperparameter search script. This
configuration is meant for hyperparameter searches of the simulation defined by
:mod:`probabilistic.prob_mnist.train_split_mt`.
"""
from probabilistic.prob_mnist import hpsearch_config_split_bbb as hpsplitbbb

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
    #'training_set_size': [-1],

    ### Main network options ###
    #'net_type' : ['mlp'], # RELATED WORK - 'mlp'
    #'mlp_arch' : ['"400,400"'], # RELATED WORK - '"400,400"'
    #'lenet_type' : ['mnist_small'], # 'mnist_small', 'mnist_large'
    #'resnet_block_depth': [5],
    #'resnet_channel_sizes': ['"16,16,32,64"'],
    #'wrn_block_depth': [4],
    #'wrn_widening_factor': [10],
    #'wrn_use_fc_bias': [False],
    #'net_act' : ['relu'], # RELATED WORK - 'relu'
    #'no_bias' : [False],
    #'dropout_rate' : [-1],
    #'batchnorm' : [False],
    #'bn_no_running_stats': [False],

    ### Evaluation options ###
    #'val_iter' : [500],
    #'val_batch_size' : [1000],
    #'val_set_size' : [0],

    ### Miscellaneous options ###
    #'no_cuda' : [False],
    #'deterministic_run': [True],
    #'random_seed': [42],
    #'store_final_model': [False],
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
# The following set of conditions should always be enforced.
conditions = conditions + hpsplitbbb._BASE_CONDITIONS

# Name of the script that should be executed by the hyperparameter search.
# Note, the working directory is set seperately by the hyperparameter search
# script.
_SCRIPT_NAME = 'train_split_mt.py'

# This file is expected to reside in the output folder of the simulation.
_SUMMARY_FILENAME = 'performance_overview.txt'
# These are the keywords that are supposed to be in the summary file.
# A summary file always has to include the keyword "finished"!.
_SUMMARY_KEYWORDS = [
    # Note, that during scores don't make sense for multitask learning. They
    # just make the implementation easier.
    'acc_avg_final', # Depends on CL scenario.
    'acc_avg_during', # Depends on CL scenario.

    # Note, task can only be inferred if ``--cl_scenario=3 --split_head_cl3``.
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

    'num_weights_main',

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
    performance = float(summary_dict['acc_avg_final'][0])
    return performance > performance_criteria

# A function handle, that is used to evaluate the performance of a run.
_PERFORMANCE_EVAL_HANDLE = _performance_criteria

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
# >>> from classifier.imagenet import train_args as targs
# >>> f = lambda argv : targs.parse_cmd_arguments(mode='cl_ilsvrc_cub',
# ...                                             argv=argv)
# >>> _ARGPARSE_HANDLE = f
import probabilistic.multitask_args as targs
_ARGPARSE_HANDLE = lambda argv : targs.parse_cmd_arguments( \
    mode='split_mnist_mt', argv=argv)

if __name__ == '__main__':
    pass


