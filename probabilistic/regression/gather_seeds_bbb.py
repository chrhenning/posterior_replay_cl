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
# @title          :probabilistic/regression/gather_seeds_bbb.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/08/2021
# @version        :1.0
# @python_version :3.8.5
"""
Gather random seeds for experiments configured via
:mod:`probabilistic.regression.hpsearch_config_bbb`.

Here is an example call for how to start the gathering on the cluster:

.. code-block:: console

   $ python3 gather_seeds_bbb.py --run_dir=out_bbb/example_bbb_run --config_name=./seeds_example_run_bbb --num_seeds=10 --start_gathering --run_cluster --hps_num_hours=4 --num_hours=4 --resources="\"rusage[mem=8000, ngpus_excl_p=1]\"" --num_jobs=10 --run_cwd="$(pwd -P)"

and here another example if the cluster is using the SLURM scheduler:

.. code-block:: console

   $ python3 gather_seeds_bbb.py --run_dir=out_bbb/example_bbb_run --config_name=./seeds_example_run_bbb --num_seeds=10 --start_gathering --run_cluster --scheduler=slurm --slurm_partition=vesta --slurm_qos=medium --hps_num_hours=4 --num_hours=24 --slurm_mem=8GB --run_cwd="$(pwd -P)"

and here another example if running on a multi-GPU workstation without job
scheduler:

.. code-block:: console

   $ python3 gather_seeds_perm_bbb.py --run_dir=out_bbb/example_bbb_run --config_name=./seeds_example_run_bbb --num_seeds=10 --start_gathering --allowed_load=0.5 --allowed_memory=0.5 --sim_startup_time=60 --visible_gpus="0,1,2,3" --max_num_jobs_per_gpu=1 --run_cwd="$(pwd -P)"
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from hpsearch import gather_random_seeds

FORCED_PARAMS = {
    'val_sample_size': '100',
    'store_final_model': True,
}

SUMMARY_KEYS = [
    'aa_mse_during_mean',
    'aa_mse_final_mean',
    'aa_mse_during_inferred_mean',
    'aa_mse_final_inferred_mean',
    'aa_task_inference_mean',
]

SUMMARY_PRECS=[5, 5, 5, 5, 2]

if __name__ == '__main__':
    gather_random_seeds.run('probabilistic.regression.hpsearch_config_bbb',
        ignore_kwds=None, forced_params=FORCED_PARAMS,
        summary_keys=SUMMARY_KEYS, summary_sem=True,
        summary_precs=SUMMARY_PRECS, hpmod_path='../../hpsearch')
