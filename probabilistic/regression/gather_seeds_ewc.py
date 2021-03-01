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
# @title          :probabilistic/regression/gather_seeds_ewc.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/12/2021
# @version        :1.0
# @python_version :3.8.5
"""
Gather random seeds for experiments configured via
:mod:`probabilistic.regression.hpsearch_config_ewc`.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from hpsearch import gather_random_seeds
from probabilistic.regression import gather_seeds_bbb as gsbbb

if __name__ == '__main__':
    gather_random_seeds.run('probabilistic.regression.hpsearch_config_ewc',
        ignore_kwds=None, forced_params=gsbbb.FORCED_PARAMS,
        summary_keys=gsbbb.SUMMARY_KEYS, summary_sem=True,
        summary_precs=gsbbb.SUMMARY_PRECS, hpmod_path='../../hpsearch')


