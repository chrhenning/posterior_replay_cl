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
# @title          :probabilistic/prob_mnist/train_perm_avb_pf.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/30/2020
# @version        :1.0
# @python_version :3.6.9
"""
Train implicit posterior via AVB for prior-focused PermutedMNIST
----------------------------------------------------------------

The script :mod:`probabilistic.prob_mnist.train_perm_avb_pf` is used to run
experiments on PermutedMNIST. It's role is analogous to the one of the script
:mod:`probabilistic.prob_mnist.train_split_avb_pf`.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from probabilistic.prob_mnist import train_args
from probabilistic.prob_cifar import train_avb

if __name__ == '__main__':
    config = train_args.parse_cmd_arguments(mode='perm_mnist_avb_pf')

    train_avb.run(config, experiment='perm_mnist_avb_pf')


