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
# @title          :probabilistic/regression/train_mt.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :09/03/2021
# @version        :1.0
# @python_version :3.8.10
"""
Multitask baseline for Toy regression
-------------------------------------

This script is used to run Multitask experiments on on toy regression datasets.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from probabilistic import multitask_args
from probabilistic import multitask_utils

if __name__ == '__main__':
    config = multitask_args.parse_cmd_arguments(mode='regression_mt')

    multitask_utils.run(config, experiment='regression_mt')

if __name__ == '__main__':
    pass


