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
# title          :probabilistic/prob_mnist/train_perm_bbb.py
# author         :ch
# contact        :henningc@ethz.ch
# created        :09/06/2019
# version        :1.0
# python_version :3.6.8
"""
Train Gaussian per-task posteriors for PermutedMNIST
----------------------------------------------------

This script is used to run experiments on PermutedMNIST. It's role is
analogous to the one of the script
:mod:`probabilistic.prob_mnist.train_split_bbb`.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from probabilistic.prob_mnist import train_args
from probabilistic.prob_mnist import train_bbb

if __name__ == '__main__':
    config = train_args.parse_cmd_arguments(mode='perm_mnist_bbb')

    train_bbb.run(config, experiment='perm_bbb')
