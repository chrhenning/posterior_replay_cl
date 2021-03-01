#!/usr/bin/env python3
# Copyright 2020 Maria Cervera
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
# @title           :probabilistic/prob_mnist/train_perm_ssge.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :18/12/2020
# @version         :0.1
# @python_version  :3.7
"""
Per-task implicit posterior via Spectral Stein Gradient Estimator
-----------------------------------------------------------------

In this script, we train a target network via variational inference, where the
variational family is NOT restricted to a set of Gaussian distributions with
diagonal covariance matrix (as in
:mod:`probabilistic.prob_mnist.train_bbb`).
For the training we use an implicit method, the training method for this case
is described in

    Shi, Jiaxin, Shengyang Sun, and Jun Zhu. "A spectral approach to gradient 
    estimation for implicit distributions." ICML, 2018.
    https://arxiv.org/abs/1806.02925

Specifically, we use a hypernetwork to output the weights for the target
network of each task in a continual learning setup, where tasks are presented
sequentially and forgetting of previous tasks is prevented by the
regularizer proposed in

    https://arxiv.org/abs/1906.00695
"""
# Do not delete the following import for all executable scripts!
import __init__  # pylint: disable=unused-import

from probabilistic.prob_mnist import train_args
from probabilistic.prob_cifar import train_avb

if __name__ == '__main__':
    config = train_args.parse_cmd_arguments(mode='perm_mnist_ssge')

    train_avb.run(config, experiment='perm_mnist_ssge')
