Toy classification problems for continual learning with probabilistic hypernetworks
===================================================================================

The set of tasks to be solved is defined in the function :func:`probabilistic.prob_gmm.train_utils.generate_datasets`. The task definition is hard-coded to allow maximum flexibility (which would be difficult to realize via command-line options).

Component Classification for Gaussian-Mixture Models
----------------------------------------------------

.. _prob-gmm-det-readme-reference-label:

Deterministic Task-specific Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we reproduce the setup of `von Oswald et al. <https://arxiv.org/abs/1906.00695>`__, where CL3 uses the algorithm
HNET+ENT proposed in the paper.

The following run achieves 100% CL1 and 45% CL3.

.. code-block:: console

    $ python3 train_gmm_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --beta=0.01 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=5001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --mlp_arch=10,10 --net_act=relu --hnet_type=hmlp --hmlp_arch=10,10 --cond_emb_size=2 --hnet_net_act=relu --std_normal_temb=1.0 --mean_only

.. _prob-gmm-bbb-readme-reference-label:

Task-specific Posterior with BbB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves 100% CL1 and 90% CL3 (agree).

.. code-block:: console

    $ python3 train_gmm_bbb.py --kl_schedule=0 --momentum=-1 --beta=0.1 --cl_scenario=3 --split_head_cl3 --regularizer=w2 --batch_size=32 --n_iter=5001 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --local_reparam_trick --kl_scale=1.0 --mlp_arch="10,10" --net_act=relu --hnet_type=hmlp --hmlp_arch="50,50" --cond_emb_size=32 --hnet_net_act=sigmoid --std_normal_temb=1.0 --hyper_gauss_init --disable_lrt_test

.. _prob-gmm-radial-readme-reference-label:

Task-specific Posterior using Radial approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves 95% CL1 and 76% CL3 (agree).

.. code-block:: console

    $ python3 train_gmm_bbb.py --regularizer=mse --kl_schedule=0 --momentum=-1 --beta=1.0 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=5001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --kl_scale=1.0 --radial_bnn --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --hnet_type=hmlp --hmlp_arch="50,50" --cond_emb_size=8 --hnet_net_act=relu --std_normal_temb=1.0 --keep_orig_init --hyper_gauss_init

.. _prob-gmm-avb-readme-reference-label:

Task-specific Posterior with AVB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The runs below use an **inflating hypernetwork** architecture that obtains a full-support implicit distribution via **noise perturbations** using option ``--full_support_perturbation``.

The following run achieves 100% CL1 and 100% CL3 after random seeds.

.. code-block:: console

    $ python3 train_gmm_avb.py --kl_schedule=0 --momentum=-1 --beta=0.1 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=5001 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=100 --prior_variance=1.0 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=1.0 --latent_dim=8 --full_support_perturbation=0.02 --dis_lr=-1.0 --dis_batch_size=10 --num_dis_steps=1 --num_ac_samples=100

The following run achieves 100.00% CL1 and 98.42% CL3-ent.

.. code-block:: console

    $ python3 train_gmm_avb.py --kl_schedule=0 --momentum=-1 --beta=0.01 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=20001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=100 --prior_variance=1.0 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch=10,10 --net_act=relu --dis_net_type=mlp --dis_mlp_arch=10,10 --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch=10,10 --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch=100,100 --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=1.0 --latent_dim=8 --full_support_perturbation=0.02 --dis_lr=-1.0 --dis_batch_size=10 --num_dis_steps=1 --num_ac_samples=100

.. _prob-gmm-avb-pf-readme-reference-label:

Prior-focused Continual Learning with AVB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves 98% CL1 and 65% CL3.

.. code-block:: console

    $ python3 train_gmm_avb_pf.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=5001 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --train_sample_size=1 --prior_variance=1.0 --kl_scale=0.01 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="100,100" --dis_net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=sigmoid --hyper_fan_init --latent_dim=32 --full_support_perturbation=0.02 --dis_lr=-1.0 --dis_batch_size=10 --num_dis_steps=5 --use_batchstats --num_ac_samples=100

.. _prob-gmm-ssge-readme-reference-label:

Task-specific Posterior with SSGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The runs below use an **inflating hypernetwork** architecture that obtains a full-support implicit distribution via **noise perturbations** using option ``--full_support_perturbation``.

The following run achieves 100% CL1 and 100% CL3.

.. code-block:: console

    $ python3 train_gmm_ssge.py --kl_schedule=0 --momentum=-1 --beta=0.1 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=20001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch=10,10 --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch=10,10 --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch=100,100 --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=1.0 --latent_dim=32 --full_support_perturbation=0.02 --rbf_kernel_width=1.0 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=0.9 --ssge_sample_size=100

The following run achieves 100.00% CL1 and 99.53% CL3-ent.

.. code-block:: console

    $ python3 train_gmm_ssge.py --beta=1.0 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=20001 --lr=0.001 --use_adam --clip_grad_value=-1 --clip_grad_norm=-1.0 --mlp_arch="10,10" --net_act="relu" --imp_hnet_type="hmlp" --imp_hmlp_arch="10,10" --imp_chmlp_chunk_size=1500 --imp_chunk_emb_size=2 --imp_hnet_net_act="relu" --hh_hnet_type="hmlp" --hh_hmlp_arch="100,100" --hh_cond_emb_size=8 --hh_chmlp_chunk_size=1500 --hh_chunk_emb_size=2 --hh_hnet_net_act="relu" --std_normal_temb=1.0 --std_normal_emb=1.0 --hyper_fan_init --train_sample_size=100 --val_sample_size="100" --prior_variance=1.0 --kl_scale=1.0 --num_kl_samples=10 --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.02 --rbf_kernel_width=1.0 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=100

Prior-focused Continual Learning with SSGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves 96% CL1 and 50% CL3 after random seeds.

.. code-block:: console

    $ python3 train_gmm_ssge_pf.py --kl_schedule=0 --momentum=-1 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=20001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --prior_variance=1.0 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=sigmoid --latent_dim=32 --full_support_perturbation=0.02 --rbf_kernel_width=1.0 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=10

Training separate deterministic main networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can use the code to train seperate deterministic main networks.

The following multi-head run achieves 100% CL1 and 71% CL3.

.. code-block:: console

    $ python3 train_gmm_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=10001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --mlp_arch="10,10" --net_act=relu --mnet_only --mean_only

Training separate Gaussian main networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this control, we train a separate main network via BbB for each task.

The following multi-head run achieves 100% CL1 and 88% CL3 (model agreement vs. 85% CL3 using entropy).

.. code-block:: console

    $ python3 train_gmm_bbb.py --disable_lrt_test --kl_schedule=0 --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=10001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=1 --prior_variance=1.0 --local_reparam_trick --kl_scale=1.0 --mlp_arch="10,10" --net_act=relu --mnet_only

The following single-head run achieves 100% CL1 and 88% CL3.

.. code-block:: console

    $ python3 train_gmm_bbb.py --disable_lrt_test --kl_schedule=0 --momentum=-1 --train_from_scratch --cl_scenario=2 --batch_size=32 --n_iter=5001 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --local_reparam_trick --kl_scale=1.0 --mlp_arch="10,10" --net_act=relu --mnet_only

Training separate implicit distributions via AVB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this control, we train a separate implicit distribution (i.e., hypernetwork) via AVB for each task.

The following multi-head run achieves 100% CL1 and 98.5% CL3 (model agreement vs. 95% CL3 using entropy).

.. code-block:: console

    $ python3 train_gmm_avb.py --kl_schedule=0 --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=5001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="100,100" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=relu --hyper_fan_init --no_hhnet --latent_dim=8 --full_support_perturbation=0.02 --dis_lr=-1.0 --dis_batch_size=10 --num_dis_steps=5 --use_batchstats --num_ac_samples=100

The following single-head run achieves 100% CL1 and 100% CL2 after random seeds.

.. code-block:: console

    $ python3 train_gmm_avb.py --kl_schedule=0 --momentum=-1 --train_from_scratch --cl_scenario=2 --batch_size=32 --n_iter=10001 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=sigmoid --no_hhnet --latent_dim=32 --full_support_perturbation=0.02 --dis_lr=-1.0 --dis_batch_size=10 --num_dis_steps=5 --num_ac_samples=100

The following single-head run achieves 100% CL1 and 98% CL2 after random seeds.

.. code-block:: console

    $ python3 train_gmm_avb.py --kl_schedule=0 --momentum=-1 --train_from_scratch --cl_scenario=2 --batch_size=32 --n_iter=5001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="100,100" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=relu --no_hhnet --latent_dim=32 --full_support_perturbation=0.02 --dis_lr=-1.0 --dis_batch_size=10 --num_dis_steps=1 --num_ac_samples=100

Elastic Weight Consolidation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **a growing softmax** achieves 50.00% CL3.

.. code-block:: console 

    $ python3 train_gmm_ewc.py --momentum=-1 --cl_scenario=3 --batch_size=32 --n_iter=10001 --lr=0.0005 --use_adam --clip_grad_value=1.0 --prior_variance=1.0 --mlp_arch=10,10 --net_act=relu --ewc_gamma=1.0 --ewc_lambda=1.0

The following run with **a non-growing softmax** achieves 66.97% 3.

.. code-block:: console 

    $ python3 train_gmm_ewc.py --momentum=-1 --cl_scenario=3 --non_growing_sf_cl3 --batch_size=32 --n_iter=2001 --lr=0.0001 --use_adam --clip_grad_value=1.0 --prior_variance=1.0 --mlp_arch=10,10 --net_act=relu --ewc_gamma=1.0 --ewc_lambda=10.0

Variational Continual Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following **multi-head** run with achieves 100.00% CL1 and 45.17% CL3-ent.

.. code-block:: console 

    $ python3 train_gmm_bbb.py --kl_schedule=0 --momentum=-1 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=10001 --lr=0.0005 --use_adam --clip_grad_norm=-1 --train_sample_size=100 --prior_variance=1.0 --kl_scale=1.0 --use_prev_post_as_prior --mlp_arch=10,10 --net_act=relu --mnet_only

Fine-Tuning
^^^^^^^^^^^

The following run with **only mnet** achieves 99.90% CL1 and 47.98% CL3-ent.

.. code-block:: console 

    $ python3 train_gmm_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --beta=0 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=2001 --lr=0.001 --use_adam --clip_grad_norm=-1 --mlp_arch=10,10 --net_act=relu --mnet_only --mean_only

The following run with **mnet+hnet** achieves 100.00% CL1 and 49.68% CL3-ent.

.. code-block:: console 

    $ python3 train_gmm_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --beta=0 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=10001 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --mlp_arch=10,10 --net_act=relu --hnet_type=hmlp --hmlp_arch=50,50 --cond_emb_size=8 --hnet_net_act=relu --std_normal_temb=1.0 --mean_only
