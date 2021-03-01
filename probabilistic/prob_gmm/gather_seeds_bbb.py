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
# @title          :probabilistic/prob_gmm/gather_seeds_bbb.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/08/2021
# @version        :1.0
# @python_version :3.8.5
"""
Gather random seeds for experiments configured via
:mod:`probabilistic.prob_gmm.hpsearch_config_gmm_bbb`.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from hpsearch import gather_random_seeds

FORCED_PARAMS = {
    'val_sample_size': '100',
    'store_final_model': True,
    'during_acc_criterion': '-1',
}

SUMMARY_KEYS = [
    'acc_avg_task_given_during',
    'acc_avg_task_given',
    'acc_avg_task_inferred_ent_during',
    'acc_avg_task_inferred_ent',
    'acc_avg_task_inferred_conf',
    'acc_avg_task_inferred_agree',
]

if __name__ == '__main__':
    gather_random_seeds.run('probabilistic.prob_gmm.hpsearch_config_gmm_bbb',
        ignore_kwds=None, forced_params=FORCED_PARAMS,
        summary_keys=SUMMARY_KEYS, summary_sem=True,
        hpmod_path='../../hpsearch')


