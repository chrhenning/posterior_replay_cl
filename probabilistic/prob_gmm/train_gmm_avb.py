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
# @title          :probabilistic/prob_gmm/train_gmm_avb.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :03/10/2020
# @version        :1.0
# @python_version :3.6.10
"""
Train implicit per-task posteriors for GMM tasks with AVB
---------------------------------------------------------

The script  :mod:`probabilistic.prob_gmm.train_gmm_avb` is used to run a
probabilistic CL experiment on a toy classification problem using synthetic
data (:class:`data.special.GMMData`). Adversarial-Variational-Bayes (AVB) is
used to learn task-specific weight posteriors. At the moment, the script simply
takes care of providing the correct command-line arguments and default values to
the end user. Afterwards, it will simply call:
:mod:`probabilistic.prob_mnist.train_avb`.

See :ref:`prob-gmm-avb-readme-reference-label` for usage instructions.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from probabilistic.prob_cifar import train_avb
from probabilistic.prob_mnist import train_args

if __name__ == '__main__':
    config = train_args.parse_cmd_arguments(mode='gmm_avb')

    train_avb.run(config, experiment='gmm_avb')


