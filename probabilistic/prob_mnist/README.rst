MNIST experiments for continual learning with probabilistic hypernetworks
=========================================================================

Continual learning scenarios
----------------------------

We consider the continual learning scenarios proposed by `van de Ven et al. <https://arxiv.org/abs/1904.07734>`_.

- **CL1**: The task identity is known, i.e., we feed the hypernetwork with the correct task embedding. The main network is a multi-head network and we always pick the correct head.
- **CL2**: The main network is a single head network. In our case, we need to infer the task embedding to feed into the hypernetwork.
- **CL3 with growing softmax**: This is the standard scenario when passing ``--cl_scenario=3``. The main network has a seperate output neuron for each class of each task. The output layer is assumed to grow for each incoming task. Therefore, the softmax is always computed across all output neurons of previous tasks and the output neurons of the current task. The task identity needs to be inferred. Based on the task inference, we select the task embedding and the size of the softmax layer.
- **CL3 with separate softmax**: This scenario is active when passing ``--cl_scenario=3 --split_head_cl3``. Since we infer the task identiy anyway, there is no need to compute the softmax over more than the current output neurons (the ones that belong to the curren task). Therefore, this option is like **CL1** except that the task identity is inferred and not given.

Split MNIST
-----------

Please run the following command to see the available options for running Split MNIST experiments.

.. code-block:: console

    $ python3 train_split_bbb.py --help

Deterministic Task-specific Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we reproduce the setup of `von Oswald et al. <https://arxiv.org/abs/1906.00695>`__, where CL3 uses the algorithm
HNET+ENT proposed in the paper.

The following run with a **small MLP** achieves 99.72% CL1 and 63.41% CL3.

.. code-block:: console

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --chmlp_chunk_size=8000 --beta=10.0 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --n_iter=5000 --lr=0.0005 --use_adam --clip_grad_norm=-1 --net_type=mlp --mlp_arch="100,100" --net_act=relu --dropout_rate=-1 --hnet_type=chunked_hmlp --hmlp_arch="10,10,10,10" --cond_emb_size=32 --chunk_emb_size="32" --std_normal_temb=1.0 --std_normal_emb=0.1 --mean_only

The following run with a **large MLP** achieves 99.67% CL1 and 74.43% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --chmlp_chunk_size=4500 --beta=2.0 --cl_scenario=3 --split_head_cl3 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --num_kl_samples=20 --during_acc_criterion=0.95 --net_type=mlp --hnet_type=chunked_hmlp --hmlp_arch=100,100 --chunk_emb_size=32 --hnet_net_act=sigmoid --mean_only

The following run with a **large MLP** and single-task CL regularization achieves 99.64% CL1 and 71.34% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --beta=10.0 --cl_scenario=3 --split_head_cl3 --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --calc_hnet_reg_targets_online --hnet_reg_batch_size=1 --num_kl_samples=20 --during_acc_criterion=0.95 --net_type=mlp --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=32 --chmlp_chunk_size=4500 --chunk_emb_size=32 --hnet_net_act=sigmoid --std_normal_init=0.02 --std_normal_temb=1.0 --std_normal_emb=1.0 --mean_only

The following run with a **large Lenet** achieves 99.91% CL1 and 82.75% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --chmlp_chunk_size=38000 --beta=2.0 --cl_scenario=3 --split_head_cl3 --n_iter=5000 --lr=0.001 --use_adam --clip_grad_norm=-1 --num_kl_samples=10 --during_acc_criterion=0.95 --net_type=lenet --lenet_type=mnist_large --hnet_type=chunked_hmlp --hmlp_arch=10,10 --chunk_emb_size=16 --hnet_net_act=relu --mean_only


Deterministic Task-specific Solutions with SupSup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **large MLP** achieves 99.66% CL1 and 71.22% CL3-ent, with task-inference accuracy of 71% based on entropy and 88% based on the entropy gradient.

.. code-block:: console

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --hnet_reg_batch_size=-1 --chmlp_chunk_size=4500 --beta=0.1 --cl_scenario=3 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --num_kl_samples=1 --during_acc_criterion=0.95 --supsup_task_inference --supsup_lr=0.001 --supsup_grad_steps=20 --net_type=mlp --hnet_type=chunked_hmlp --hmlp_arch=100,100 --chunk_emb_size=32 --hnet_net_act=relu --mean_only

Task-specific Posterior with BbB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **small MLP** achieves 99.75% CL1, 70.07% CL3-ent and 70.11% CL3-agree.

.. code-block:: console

    $ python3 train_split_bbb.py --momentum=-1 --chmlp_chunk_size=2500 --beta=100.0 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=w2 --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.0001 --net_type=mlp --mlp_arch="100,100" --net_act=relu --dropout_rate=-1 --hnet_type=chunked_hmlp --hmlp_arch="" --cond_emb_size=32 --chunk_emb_size="32" --use_cond_chunk_embs --std_normal_temb=0.1 --std_normal_emb=0.1

The following run with a **large MLP** achieves 99.72% CL1 and 71.73% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --chmlp_chunk_size=1350 --beta=0.1 --cl_scenario=3 --split_head_cl3 --regularizer=fkl --n_iter=2000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --kl_scale=1e-06 --num_kl_samples=20 --net_type=mlp --hnet_type=chunked_hmlp --hmlp_arch=100,250,500 --cond_emb_size=64 --chunk_emb_size=32 --use_cond_chunk_embs --hnet_net_act=relu --keep_orig_init

The following run with a **large MLP** and single-task CL regularization achieves 99.73% CL1 and 72.38% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --beta=0.1 --cl_scenario=3 --split_head_cl3 --regularizer=fkl --batch_size=128 --n_iter=2000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --kl_scale=1e-06 --calc_hnet_reg_targets_online --hnet_reg_batch_size=1 --num_kl_samples=20 --net_type=mlp --hnet_type=chunked_hmlp --hmlp_arch=100,250,500 --cond_emb_size=64 --chmlp_chunk_size=1350 --chunk_emb_size=32 --use_cond_chunk_embs --hnet_net_act=relu --keep_orig_init

The following run with a **large Lenet** achieves 99.20% CL1 and 74.09% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --chmlp_chunk_size=78000 --beta=1.0 --cl_scenario=3 --split_head_cl3 --regularizer=fkl --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --kl_scale=0.001 --num_kl_samples=10 --net_type=lenet --lenet_type=mnist_large --hnet_type=chunked_hmlp --hmlp_arch=10,10 --cond_emb_size=64 --chunk_emb_size=32 --hnet_net_act=relu --keep_orig_init

Task-specific Posterior with Radial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **small MLP** achieves 99.43% CL1 and 63.00% CL3-ent.

.. code-block:: console

    $ python3 train_split_bbb.py --regularizer=mse --momentum=-1 --chmlp_chunk_size=2500 --beta=1.0 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --n_iter=2000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --prior_variance=1.0 --kl_scale=0.001 --radial_bnn --num_kl_samples=1 --net_type=mlp --mlp_arch=100,100 --net_act=relu --dropout_rate=-1 --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=32 --chunk_emb_size=32 --use_cond_chunk_embs --std_normal_temb=0.1 --std_normal_emb=1.0

The following run with a **large MLP** achieves 99.88% CL1 and 66.01% CL3-ent.

.. code-block:: console 

    $  python3 train_split_bbb.py --beta=0.05 --cl_scenario=3 --split_head_cl3 --batch_size=128 --n_iter=2000 --lr=0.0001 --momentum=-1.0 --use_adam  --clip_grad_value=1.0 --clip_grad_norm=-1.0 --net_type="mlp"  --net_act="relu" --hnet_type="chunked_hmlp" --hmlp_arch="100,100" --cond_emb_size=32 --chmlp_chunk_size=9000 --chunk_emb_size="32" --use_cond_chunk_embs --hnet_net_act="relu" --std_normal_init=0.02 --std_normal_temb=1.0 --std_normal_emb=1.0 --train_sample_size=10 --kl_scale=0.0001 --radial_bnn --regularizer="mse" --hyper_gauss_init --num_kl_samples=5

The following run with a **large Lenet** achieves 99.78% CL1 and 68.99% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --regularizer=mse --momentum=-1 --chmlp_chunk_size=78000 --beta=1.0 --cl_scenario=3 --split_head_cl3 --n_iter=2000 --lr=5e-05 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --kl_scale=0.0001 --radial_bnn --num_kl_samples=20 --net_type=lenet --lenet_type=mnist_large --hnet_type=chunked_hmlp --hmlp_arch=10,10 --cond_emb_size=64 --chunk_emb_size=16 --hnet_net_act=relu --keep_orig_init

Task-specific Posterior with SSGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **small MLP** achieves 99.65% CL1 and 66.15% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ssge.py --momentum=-1 --rbf_kernel_width=0.01 --hh_chmlp_chunk_size=650 --imp_chmlp_chunk_size=1300 --beta=100.0 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=128 --n_iter=5000 --lr=0.001 --use_adam --clip_grad_norm=100.0 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.0001 --num_kl_samples=1 --net_type=mlp --mlp_arch=100,100 --net_act=relu --dropout_rate=-1 --imp_hnet_type=chunked_hmlp --imp_hmlp_arch= --imp_chunk_emb_size=32 --imp_hnet_net_act=sigmoid --hh_hnet_type=chunked_hmlp --hh_hmlp_arch=100,100 --hh_cond_emb_size=32 --hh_chunk_emb_size=32 --hh_hnet_net_act=sigmoid --std_normal_temb=1.0 --std_normal_emb=1.0 --hyper_fan_init --during_acc_criterion=95,90,90,90 --latent_dim=32 --full_support_perturbation=0.0002 --num_ssge_eigenvals=-1 --ssge_sample_size=10

The following run with a **large MLP** achieves 99.77% CL1 and 71.91% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ssge.py --momentum=-1 --imp_chmlp_chunk_size=85000 --hh_chmlp_chunk_size=42000 --beta=10.0 --cl_scenario=3 --split_head_cl3 --n_iter=5000 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --kl_scale=1e-05 --num_kl_samples=10 --during_acc_criterion=0.95 --net_type=mlp --imp_hnet_type=chunked_hmlp --imp_hmlp_arch=10,10 --imp_chunk_emb_size=16 --imp_hnet_net_act=relu --full_support_perturbation=0.01 --hh_hnet_type=chunked_hmlp --hh_hmlp_arch=10,10 --hh_hnet_net_act=sigmoid --latent_dim=16 --latent_std=1.0 --thr_ssge_eigenvals=0.95 --ssge_sample_size=20 

The following run with a **large Lenet** achieves 99.89% CL1 and 77.56% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ssge.py --rbf_kernel_width=1.0 --momentum=-1 --hh_chmlp_chunk_size=4000 --beta=0.1 --cl_scenario=3 --split_head_cl3 --n_iter=5000 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=1 --kl_scale=1e-05 --num_kl_samples=20 --during_acc_criterion=0.95 --net_type=lenet --lenet_type=mnist_large --imp_hnet_type=chunked_hmlp --imp_hmlp_arch=100,100 --imp_chmlp_chunk_size=3000 --imp_chunk_emb_size=16 --imp_hnet_net_act=sigmoid --full_support_perturbation=0.1 --hh_hnet_type=chunked_hmlp --hh_hmlp_arch=100,100 --hh_chunk_emb_size=32 --hh_hnet_net_act=sigmoid --heuristic_kernel --thr_ssge_eigenvals=0.95 --ssge_sample_size=20 

Shared Posterior with SSGE
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **small MLP** achieves 96.48% CL1 and 51.26% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ssge_pf.py --rbf_kernel_width=1.0 --momentum=-1 --imp_chmlp_chunk_size=1300 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=128 --n_iter=2000 --lr=0.0005 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=1e-05 --num_kl_samples=1 --net_type=mlp --mlp_arch=100,100 --net_act=relu --dropout_rate=-1 --imp_hnet_type=chunked_hmlp --imp_hmlp_arch= --imp_chunk_emb_size=32 --imp_hnet_net_act=sigmoid --std_normal_temb=0.1 --std_normal_emb=1.0 --hyper_fan_init --latent_dim=8 --latent_std=0.1 --full_support_perturbation=0.02 --heuristic_kernel --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=0.9 --ssge_sample_size=10

The following run with a **large MLP** achieves 99.02% CL1 and 62.70% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ssge_pf.py --momentum=-1 --imp_chmlp_chunk_size=9500 --cl_scenario=3 --split_head_cl3 --n_iter=2000 --lr=5e-05 --use_adam --clip_grad_norm=-1 --train_sample_size=20 --kl_scale=0.0001 --num_kl_samples=50 --during_acc_criterion=80 --net_type=mlp --imp_hnet_type=chunked_hmlp --imp_hmlp_arch= --imp_chunk_emb_size=32 --imp_hnet_net_act=relu --full_support_perturbation=-1 --latent_dim=16 --latent_std=1.0 --thr_ssge_eigenvals=1.0 --ssge_sample_size=20

The following run with a **large Lenet** achieves 99.37% CL1 and 74.18% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ssge_pf.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --n_iter=2000 --lr=5e-05 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --kl_scale=0.0001 --num_kl_samples=10 --during_acc_criterion=80 --net_type=lenet --lenet_type=mnist_large --imp_hnet_type=chunked_hmlp --imp_hmlp_arch= --imp_chmlp_chunk_size=3000 --imp_chunk_emb_size=32 --imp_hnet_net_act=relu --full_support_perturbation=0.01 --thr_ssge_eigenvals=1.0 --ssge_sample_size=10

Shared Posterior with VCL
^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **small MLP** achieves 96.05% CL1 and 51.45% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=128 --n_iter=2000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=10.0 --use_prev_post_as_prior --net_type=mlp --mlp_arch=100,100 --net_act=relu --dropout_rate=-1 --keep_orig_init --mnet_only

The following run with a **large MLP** achieves 96.45% CL1 and 58.84% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --batch_size=128 --n_iter=2000 --lr=1e-05 --use_adam --clip_grad_norm=1 --train_sample_size=1 --kl_scale=0.0001 --use_prev_post_as_prior --net_type=mlp --mnet_only

The following run with a **large Lenet** achieves 97.43% CL1 and 63.05% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=5000 --lr=1e-05 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --kl_scale=0.01 --use_prev_post_as_prior --net_type=lenet --lenet_type=mnist_large --mnet_only

Experience replay
^^^^^^^^^^^^^^^^^

The following run with a **small MLP** achieves 86.84% CL1 and 86.84% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --cl_scenario=3 --batch_size=8 --n_iter=1000 --lr=0.0005 --use_adam --clip_grad_norm=100.0 --num_kl_samples=1 --net_type=mlp --mlp_arch=100,100 --net_act=relu --dropout_rate=-1 --coreset_size=100 --per_task_coreset --coreset_reg=100.0 --coreset_batch_size=16 --fix_coreset_size --coresets_for_experience_replay --mnet_only --mean_only --during_acc_criterion=85

The following run with a **large MLP** achieves 88.85% CL1 and 88.85% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --cl_scenario=3 --batch_size=8 --n_iter=1000 --lr=0.0001 --use_adam --clip_grad_norm=100.0 --num_kl_samples=1 --during_acc_criterion=90 --coreset_size=100 --per_task_coreset --coreset_reg=500 --coreset_batch_size=16 --fix_coreset_size --coresets_for_experience_replay --net_type=mlp --mlp_arch=400,400 --mnet_only --mean_only

Training separate Gaussian main networks
""""""""""""""""""""""""""""""""""""""""

In this control, we train a separate main network via BbB for each task.

The following run with a **large MLP** achieves 99.81% CL1 and 68.40% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --disable_lrt_test --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --batch_size=128 --n_iter=3000 --lr=0.0005 --use_adam --clip_grad_norm=-1 --train_sample_size=20 --prior_variance=1.0 --local_reparam_trick --kl_scale=0.001 --net_type=mlp --mnet_only

The following run with a **small MLP** achieves 99.79% CL1 and 68.85% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --beta=1.0 --train_from_scratch --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=rkl --batch_size=128 --n_iter=2000 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=50 --prior_variance=1.0 --kl_scale=0.01 --net_type=mlp --mlp_arch=100,100 --mnet_only 

The following run with a **large Lenet** achieves 99.93% CL1 and 85.52% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --n_iter=5000 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --kl_scale=0.01 --net_type=lenet --lenet_type=mnist_large --mnet_only

The following run with a **hypernet-powered large MLP** achieves 99.77% CL1 and 72.74% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --chmlp_chunk_size=9000 --train_from_scratch --cl_scenario=3 --split_head_cl3 --batch_size=32 --n_iter=2000 --lr=5e-05 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=1e-05 --net_type=mlp --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=32 --chunk_emb_size=64 --use_cond_chunk_embs --hnet_net_act=sigmoid --std_normal_temb=1.0 --std_normal_emb=0.1

The following run with a **hypernet-powered large Lenet** achieves 99.92% CL1 and 84.16% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --chmlp_chunk_size=78000 --train_from_scratch --cl_scenario=3 --split_head_cl3 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --kl_scale=0.001 --net_type=lenet --lenet_type=mnist_large --hnet_type=chunked_hmlp --hmlp_arch=10,10 --cond_emb_size=64 --chunk_emb_size=32 --hnet_net_act=sigmoid

Training separate deterministic main networks
"""""""""""""""""""""""""""""""""""""""""""""

We can use the code to train seperate deterministic main networks. The option ``--mean_only`` ensures that the Gaussian main network becomes a normal main network. The option ``--main_only`` ensures that we train without a hypernetwork. Via the option ``--train_from_scratch`` we are able to train separate networks.

Hence, this control can be viewed as training an ensemble of size 1 per task.

The following run with a **small MLP** achieves 99.77% CL1 and 67.89% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --beta=10.0 --train_from_scratch --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=128 --n_iter=5000 --lr=0.0005 --use_adam --clip_grad_norm=-1 --net_type=mlp --mlp_arch=100,100 --mnet_only --mean_only

The following run with a **large MLP** achieves 99.77% CL1 and 70.39% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --num_kl_samples=1 --during_acc_criterion=90 --net_type=mlp --mnet_only --mean_only

The following run with a **large Lenet** achieves 99.92% CL1 and 85.50% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --batch_size=128 --n_iter=2000 --lr=0.001 --use_adam --clip_grad_norm=-1 --num_kl_samples=1 --during_acc_criterion=95 --net_type=lenet --lenet_type=mnist_large --mnet_only --mean_only

The following run with a **hypernet-powered large MLP** achieves 99.79% CL1 and 71.84% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --chmlp_chunk_size=43000 --train_from_scratch --cl_scenario=3 --split_head_cl3 --n_iter=2000 --lr=0.001 --use_adam --clip_grad_norm=-1 --num_kl_samples=1 --during_acc_criterion=90 --net_type=mlp --hnet_type=chunked_hmlp --hmlp_arch=10,10 --chunk_emb_size=32 --hnet_net_act=sigmoid --mean_only

The following run with a **hypernet-powered large Lenet** achieves 99.91% CL1 and 82.85% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --chmlp_chunk_size=38000 --train_from_scratch --cl_scenario=3 --split_head_cl3 --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --num_kl_samples=1 --during_acc_criterion=0.95 --net_type=lenet --lenet_type=mnist_large --hnet_type=chunked_hmlp --hmlp_arch=10,10 --chunk_emb_size=32 --hnet_net_act=sigmoid --mean_only

Training separate SSGE posterior
""""""""""""""""""""""""""""""""

The following run with a **default MLP** achieves 99.76% CL1 and 71.53% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ssge.py --momentum=-1 --imp_chmlp_chunk_size=16000 --train_from_scratch --cl_scenario=3 --split_head_cl3 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --kl_scale=0.01 --num_kl_samples=20 --during_acc_criterion=0.95 --net_type=mlp --imp_hnet_type=chunked_hmlp --imp_hmlp_arch=50,50 --imp_chunk_emb_size=32 --imp_hnet_net_act=sigmoid --full_support_perturbation=0.01 --no_hhnet --latent_dim=16 --latent_std=0.1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=20

Elastic Weight Consolidation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **small MLP and growing head** achieves 28.15% CL3.

.. code-block:: console 

    $ python3 train_split_ewc.py --momentum=-1 --cl_scenario=3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --n_iter=2001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --prior_variance=1.0 --net_type=mlp --mlp_arch=100,100 --net_act=relu --dropout_rate=-1 --ewc_gamma=1.0 --ewc_lambda=1000.0

The following run with a **small MLP and non-growing head** achieves 29.67% CL3.

.. code-block:: console 

    $ python3 train_split_ewc.py --momentum=-1 --cl_scenario=3 --num_tasks=5 --num_classes_per_task=2 --non_growing_sf_cl3 --batch_size=32 --n_iter=2001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --prior_variance=1.0 --net_type=mlp --mlp_arch=100,100 --net_act=relu --dropout_rate=-1 --ewc_gamma=1.0 --ewc_lambda=1000.0 --n_fisher=200 --out_dir=./out/hyperparam_search/search_2021-01-19_07-56-32/sim_20210119075632_054

The following run with a **small MLP and multi head** achieves 97.79% CL1 and 46.61% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ewc.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --n_iter=2001 --lr=0.0005 --use_adam --clip_grad_norm=-1 --prior_variance=1.0 --net_type=mlp --mlp_arch=100,100 --net_act=relu --dropout_rate=-1 --ewc_gamma=1.0 --ewc_lambda=10.0

The following run with a **large MLP and growing head** achieves 27.32% CL1 and 27.32% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ewc.py --momentum=-1 --cl_scenario=3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --n_iter=2001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --prior_variance=1.0 --net_type=mlp --net_act=relu --dropout_rate=-1 --ewc_gamma=1.0 --ewc_lambda=1000.0 --n_fisher=200

The following run with a **large MLP and non-growing head** achieves 30.21% CL1 and 30.21% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ewc.py --momentum=-1 --cl_scenario=3 --num_tasks=5 --num_classes_per_task=2 --non_growing_sf_cl3 --batch_size=32 --n_iter=2001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --prior_variance=1.0 --net_type=mlp --net_act=relu --dropout_rate=-1 --ewc_gamma=1.0 --ewc_lambda=1000.0 --n_fisher=200 

The following run with a **large MLP and multi head** achieves 96.40% CL1 and 47.67% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ewc.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --n_iter=5001 --lr=0.001 --use_adam --clip_grad_norm=-1 --prior_variance=1.0 --net_type=mlp --net_act=relu --dropout_rate=-1 --ewc_gamma=1.0 --ewc_lambda=0.001 --n_fisher=200

The following run with a **large Lenet and growing head** achieves 27.62% CL1 and 27.62% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ewc.py --momentum=-1 --cl_scenario=3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --n_iter=2001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --prior_variance=1.0 --net_type=lenet --lenet_type=mnist_large --net_act=relu --dropout_rate=-1 --ewc_gamma=1.0 --ewc_lambda=10000.0 --n_fisher=200

The following run with a **large Lenet and non-growing head** achieves 26.01% CL1 and 26.01% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ewc.py --momentum=-1 --cl_scenario=3 --num_tasks=5 --num_classes_per_task=2 --non_growing_sf_cl3 --batch_size=32 --n_iter=2001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --prior_variance=1.0 --net_type=lenet --lenet_type=mnist_large --net_act=relu --dropout_rate=-1 --during_acc_criterion=90 --ewc_gamma=1.0 --ewc_lambda=1000.0 --n_fisher=200

The following run with a **large Lenet and multi head** achieves 97.17% CL1 and 49.78% CL3-ent.

.. code-block:: console 

    $ python3 train_split_ewc.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --n_iter=5001 --lr=0.0005 --use_adam --clip_grad_norm=-1 --prior_variance=1.0 --net_type=lenet --lenet_type=mnist_large --net_act=relu --dropout_rate=-1 --ewc_gamma=1.0 --ewc_lambda=0.01 --n_fisher=200

Experiments with coresets
^^^^^^^^^^^^^^^^^^^^^^^^^

Exerperiments using **coresets fine-tuning**.

The following run with a **small MLP** achieves 98.70% CL1 and 90.42% CL3-ent.

.. code-block:: console

    $ python3 train_split_bbb.py --momentum=-1 --chmlp_chunk_size=2500 --beta=100.0 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=w2 --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.0001 --coreset_size=100 --per_task_coreset --final_coresets_finetune --final_coresets_kl_scale=-1 --final_coresets_n_iter=-1 --net_type=mlp --mlp_arch=100,100 --net_act=relu --dropout_rate=-1 --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=32 --chunk_emb_size=32 --use_cond_chunk_embs --std_normal_temb=0.1 --std_normal_emb=0.1

The following run with a **small MLP** achieves 99.11% CL1 and 94.02% CL3-ent.

.. code-block:: console

    $ python3 train_split_bbb.py --momentum=-1 --chmlp_chunk_size=2500 --beta=100.0 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=w2 --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.0001 --coreset_size=500 --per_task_coreset --final_coresets_finetune --final_coresets_kl_scale=-1 --final_coresets_n_iter=-1 --net_type=mlp --mlp_arch=100,100 --net_act=relu --dropout_rate=-1 --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=32 --chunk_emb_size=32 --use_cond_chunk_embs --std_normal_temb=0.1 --std_normal_emb=0.1

The following run with the **default MLP** achieves 98.50% CL1 and 90.83% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --beta=0.1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=mse --batch_size=128 --n_iter=5000 --lr=5e-05 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --prior_variance=1.0 --kl_scale=1e-05 --coreset_size=100 --per_task_coreset --final_coresets_finetune --final_coresets_kl_scale=-1 --final_coresets_n_iter=-1 --net_type=mlp --mlp_arch=400,400 --net_act=relu --dropout_rate=-1 --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=32 --chmlp_chunk_size=9000 --chunk_emb_size=32 --use_cond_chunk_embs --hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=1.0

The following run with the **default MLP** achieves 98.66% CL1 and 94.03% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --beta=0.1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=mse --batch_size=128 --n_iter=5000 --lr=5e-05 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --prior_variance=1.0 --kl_scale=1e-05 --coreset_size=500 --per_task_coreset --final_coresets_finetune --final_coresets_use_random_labels --final_coresets_kl_scale=-1 --final_coresets_n_iter=-1 --net_type=mlp --mlp_arch=400,400 --net_act=relu --dropout_rate=-1 --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=32 --chmlp_chunk_size=9000 --chunk_emb_size=32 --use_cond_chunk_embs --hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=1.0 

The following run with a **large Lenet** achieves 99.55% CL1 and 93.91% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --beta=1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=fkl --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --prior_variance=1.0 --kl_scale=0.001 --coreset_size=50 --per_task_coreset --final_coresets_finetune --final_coresets_use_random_labels --final_coresets_kl_scale=-1 --final_coresets_n_iter=-1 --net_type=lenet --lenet_type=mnist_large --net_act=relu --dropout_rate=-1 --hnet_type=chunked_hmlp --hmlp_arch=10,10 --cond_emb_size=64 --chmlp_chunk_size=78000 --chunk_emb_size=32 --hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=1.0 --keep_orig_init

The following run with a **large Lenet** achieves 99.62% CL1 and 95.73% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --beta=1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=fkl --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --prior_variance=1.0 --kl_scale=0.001 --coreset_size=100 --per_task_coreset --final_coresets_finetune --final_coresets_kl_scale=-1 --final_coresets_n_iter=-1 --net_type=lenet --lenet_type=mnist_large --net_act=relu --dropout_rate=-1 --hnet_type=chunked_hmlp --hmlp_arch=10,10 --cond_emb_size=64 --chmlp_chunk_size=78000 --chunk_emb_size=32 --hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=1.0 --keep_orig_init

The following run with a **large Lenet** achieves 99.77% CL1 and 97.74% CL3-ent.

.. code-block:: console 

    $ python3 train_split_bbb.py --momentum=-1 --beta=1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=fkl --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --prior_variance=1.0 --kl_scale=0.001 --coreset_size=500 --per_task_coreset --final_coresets_finetune --final_coresets_kl_scale=-1 --final_coresets_n_iter=-1 --net_type=lenet --lenet_type=mnist_large --net_act=relu --dropout_rate=-1 --hnet_type=chunked_hmlp --hmlp_arch=10,10 --cond_emb_size=64 --chmlp_chunk_size=78000 --chunk_emb_size=32 --hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=1.0 --keep_orig_init

Permuted MNIST-10
-----------------

Please run the following command to see the available options for running Permuted MNIST experiments.

.. code-block:: console

    $ python3 train_perm_bbb.py --help

Fine-Tuning
^^^^^^^^^^^

The following run with a **small MLP** achieves 47.89% CL1-final.

.. code-block:: console

    $ python3 train_perm_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --beta=0 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=128 --n_iter=5000 --lr=0.001 --use_adam --clip_grad_norm=1.0 --mlp_arch=100,100 --net_act=relu --mnet_only --mean_only --padding=0

The following run with a **large MLP** achieves 90.08% CL1-final.

.. code-block:: console

    $ python3 train_perm_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --beta=0 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=128 --n_iter=5000 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --mnet_only --mean_only

Deterministic Task-specific Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **small MLP** achieves 95.05% CL1 and 75.84% CL3-ent.

.. code-block:: console

    $ python3 train_perm_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --chmlp_chunk_size=750 --beta=250 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=128 --n_iter=5000 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --mlp_arch="100,100" --net_act=relu --hnet_type=chunked_hmlp --hmlp_arch="100,100" --cond_emb_size=32 --chunk_emb_size="16" --use_cond_chunk_embs --std_normal_temb=1.0 --std_normal_emb=0.1 --mean_only --padding=0

The following run with a **large MLP** achieves 96.73% CL1 and 94.15% CL3-ent.

.. code-block:: console 

    $ python3 train_perm_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --chmlp_chunk_size=40000 --beta=100 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=32 --n_iter=5000 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --hnet_type=chunked_hmlp --hmlp_arch=50,50 --cond_emb_size=16 --chunk_emb_size=16 --std_normal_temb=1.0 --std_normal_emb=0.1 --mean_only

Task-specific Posterior with BbB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **small MLP** achieves 95.96% CL1 and 89.90% CL3-ent.

.. code-block:: console

    $ python3 train_perm_bbb.py --momentum=-1 --chmlp_chunk_size=2900 --beta=50 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --regularizer=w2 --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=1 --prior_variance=1.0 --kl_scale=0.001 --mlp_arch="100,100" --net_act=relu --hnet_type=chunked_hmlp --hmlp_arch="" --cond_emb_size=32 --chunk_emb_size="32" --use_cond_chunk_embs --std_normal_temb=0.1 --std_normal_emb=0.1 --keep_orig_init --padding=0

The following run with **large MLP** achieves 96.21% CL1 and 96.14% CL3-ent.

.. code-block:: console

    $ python3 train_perm_bbb.py --momentum=-1 --chmlp_chunk_size=32000 --beta=1 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --regularizer=fkl --n_iter=5000 --lr=1e-05 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.0001 --num_kl_samples=10 --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=64 --chunk_emb_size=32 --use_cond_chunk_embs --hnet_net_act=relu --std_normal_temb=0.1 --std_normal_emb=1.0 --keep_orig_init --during_acc_criterion=85

Task-specific Posterior with Radial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **small MLP** achieves 94.30% CL1 and 81.78% CL3-ent.

.. code-block:: console

    $ python3 train_perm_bbb.py --regularizer=mse --kl_schedule=0 --momentum=-1 --chmlp_chunk_size=2900 --beta=500 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=128 --n_iter=5000 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --kl_scale=1.0 --radial_bnn --num_kl_samples=1 --mlp_arch=100,100 --net_act=relu --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=16 --chunk_emb_size=32 --use_cond_chunk_embs --std_normal_temb=0.1 --std_normal_emb=1.0 --hyper_gauss_init --padding=0

The following run with a **large MLP** achieves 97.19% CL1 and 92.92% CL3-ent.

.. code-block:: console 

    $ python3 train_perm_bbb.py --regularizer=mse --momentum=-1 --chmlp_chunk_size=41000 --beta=10000.0 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=1 --prior_variance=1.0 --kl_scale=1e-05 --radial_bnn --num_kl_samples=10 --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=64 --chunk_emb_size=16 --use_cond_chunk_embs --hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=0.1 --keep_orig_init --during_acc_criterion=94 

Task-specific Posterior with SSGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **small MLP**  achieves 92.88% CL1 and 78.94% CL3-ent.

.. code-block:: console

    $ python3 train_perm_ssge.py --kl_schedule=0 --momentum=-1 --hh_chmlp_chunk_size=1400 --imp_chmlp_chunk_size=8500 --beta=500 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=128 --n_iter=2500 --lr=0.001 --use_adam --clip_grad_norm=100.0 --train_sample_size=1 --prior_variance=1.0 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch=100,100 --net_act=relu --imp_hnet_type=chunked_hmlp --imp_hmlp_arch=10,10,10,10 --imp_chunk_emb_size=16 --hh_hnet_type=chunked_hmlp --hh_hmlp_arch= --hh_cond_emb_size=16 --hh_chunk_emb_size=32 --hh_use_cond_chunk_embs --std_normal_temb=1.0 --std_normal_emb=1.0 --hyper_fan_init --during_acc_criterion=85 --latent_dim=8 --full_support_perturbation=0.0002 --rbf_kernel_width=0.01 --num_ssge_eigenvals=-1 --ssge_sample_size=10 --padding=0

The following run with a **large MLP** achieves 97.39% CL1 and 93.58% CL3-ent.

.. code-block:: console 

    $ python3 train_perm_ssge.py --momentum=-1 --imp_chmlp_chunk_size=70000 --hh_chmlp_chunk_size=20000 --beta=10.0 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=1 --kl_scale=0.01 --num_kl_samples=10 --during_acc_criterion=86 --imp_hnet_type=chunked_hmlp --imp_hmlp_arch=20,20 --imp_chunk_emb_size=32 --imp_hnet_net_act=relu --full_support_perturbation=0.01 --hh_hnet_type=chunked_hmlp --hh_hmlp_arch=100,100 --hh_cond_emb_size=32 --hh_chunk_emb_size=16 --hh_use_cond_chunk_embs --hh_hnet_net_act=relu --latent_dim=16 --latent_std=0.1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=1

Separate Posterior with BbB
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **large MLP** achieves 97.13% CL1 and 92.95% CL3-ent.

.. code-block:: console 

    $ python3 train_perm_bbb.py --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --kl_scale=5e-05 --num_kl_samples=10 --mnet_only

The following run with a **small MLP** achieves 98.21% CL1 and 98.12% CL3-ent.

.. code-block:: console 

    $ python3 train_perm_bbb.py --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=128 --n_iter=5000 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=50 --prior_variance=1.0 --kl_scale=0.1 --mlp_arch=100,100 --net_act=relu --mnet_only --padding=0 

Training separate deterministic main networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following run with a **large MLP** achieves 98.16% CL1 and 94.70% CL3-ent.

.. code-block:: console 

    $ python3 train_perm_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=128 --n_iter=5000 --lr=0.001 --use_adam --clip_grad_norm=1.0 --num_kl_samples=1 --mnet_only --mean_only

The following run with a **small MLP** achieves 97.04% CL1 and 90.73% CL3-ent.

.. code-block:: console 

    $ python3 train_perm_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=64 --n_iter=2500 --lr=0.001 --use_adam --clip_grad_norm=1.0 --num_kl_samples=1 --mlp_arch=100,100 --net_act=relu --mnet_only --mean_only --during_acc_criterion=85 --padding=0


Shared Posterior with VCL
^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **large MLP** achieves 89.72% CL1 and 85.40% CL3-ent.

.. code-block:: console 

    $ python3 train_perm_bbb.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=128 --n_iter=2000 --lr=0.0001 --use_adam --clip_grad_norm=1 --train_sample_size=10 --prior_variance=0.1 --kl_scale=1e-05 --use_prev_post_as_prior --mnet_only

Elastic Weight Consolidation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **large MLP** achieves 94.73% CL1 and 81.12% CL3-ent.

.. code-block:: console 

    $ python3 train_perm_ewc.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --batch_size=32 --n_iter=5001 --lr=0.0001 --use_adam --clip_grad_norm=100 --prior_variance=0.1 --ewc_gamma=1.0 --ewc_lambda=0.01 --n_fisher=200

Permuted MNIST-100
------------------

The following **PR-Dirac-SR** run with a **large MLP** achieves 95.90% CL1 and 70.08% CL3-ent.

.. code-block:: console 

    $ python3 train_perm_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --chmlp_chunk_size=29000 --beta=100.0 --cl_scenario=3 --split_head_cl3 --num_tasks=100 --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=-1 --calc_hnet_reg_targets_online --hnet_reg_batch_size=8 --mlp_arch=1000,1000 --net_act=relu --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=16 --chunk_emb_size=16 --hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=0.1 --full_test_interval=25 --mean_only --padding=2

The following **PR-BbB-SR** run with a **large MLP** achieves 96.84% CL1 and 85.84% CL3-ent.

.. code-block:: console 

    $ python3 train_perm_bbb.py --momentum=-1 --chmlp_chunk_size=55000 --beta=1000.0 --cl_scenario=3 --split_head_cl3 --num_tasks=100 --regularizer=rkl --batch_size=128 --n_iter=5000 --lr=0.0001 --use_adam --clip_grad_norm=100.0 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.01 --calc_hnet_reg_targets_online --hnet_reg_batch_size=8 --mlp_arch=1000,1000 --net_act=relu --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=32 --chunk_emb_size=32 --hnet_net_act=sigmoid --std_normal_temb=1.0 --std_normal_emb=1.0 --keep_orig_init --full_test_interval=25 --store_final_model --padding=2


Miscellaneous
-------------

Description of Tensorboard labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**TODO outdated**

- ``acc_task_given``: Accuracy on data if task identity is given (i.e., the correct embedding is provided)
- ``acc_task_inferred``: Accuracy on data if task identity is inferred. I.e., we use per sample the task embedding that leads to the lowest entropy of the predictive distribution
- ``cl{1,2,3}_accuracy``: The accuracy of the current CL scenario. For **CL1**, this accuracy is identical to ``acc_task_given``. For **CL2** and **CL3**, this accuracy is identical to ``acc_task_inferred``
- ``task_inference_acc``: Task inference accuracy, i.e., how often the correct task embedding was selected
- ``hnet_out_forgetting``: Euclidean distance of current hypernet output to hypernet output right after training on the corresponding task
- ``in_ents``: Average entropy of the predictive distribution on in-distribution samples (note, test samples from a task are assumed to be in-distribution)
- ``out_ents``: Average entropy of the predictive distribution on out-of-distribution samples (i.e., test samples from other tasks are considered out-of-distribution)

Observations
^^^^^^^^^^^^

- Optimizer *Adadelta* seems to not perform well and *Adagrad* appears to be only slightly better. The best optimizers seem to be *SGD*, *Adam* and *RMSprop*.
