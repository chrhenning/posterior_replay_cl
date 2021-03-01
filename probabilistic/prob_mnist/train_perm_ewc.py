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
# @title          :probabilistic/prob_mnist/train_perm_ewc.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/10/2021
# @version        :1.0
# @python_version :3.8.5
"""
PermutedMNIST with EWC
----------------------

This script is used to run EWC experiments on PermutedMNIST.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from probabilistic import ewc_args
from probabilistic import ewc_utils

if __name__ == '__main__':
    config = ewc_args.parse_cmd_arguments(mode='perm_mnist_ewc')

    ewc_utils.run(config, experiment='perm_mnist_ewc')


