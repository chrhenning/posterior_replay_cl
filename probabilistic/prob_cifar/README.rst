CIFAR-10 experiments for continual learning with probabilistic hypernetworks
============================================================================

SplitCIFAR-10
-------------

This section focuses on a SplitCIFAR experiment that contains 5 tasks of CIFAR-10 splits with 2 classes each. These experiments always require the arguments ``--num_tasks=5 --num_classes_per_task=2``.

Fine-Tuning
^^^^^^^^^^^

The following run with **Resnet-32** achieves 96.59% CL1-during and 60.25% CL1-final.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --beta=0 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --epochs=80 --lr=0.0005 --use_adam --clip_grad_norm=-1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --mnet_only --mean_only

Deterministic Task-specific Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **Resnet-32** achieves 93.77% CL1 and 54.74% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --chmlp_chunk_size=7000 --beta=50 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --epochs=40 --lr=0.0005 --use_adam --clip_grad_norm=-1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=16 --chunk_emb_size=16 --hnet_net_act=sigmoid --std_normal_temb=0.01 --std_normal_emb=1.0 --mean_only

The following run with a **WRN-28-10** achieves 95.75% CL1 and 57.50% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --chmlp_chunk_size=700000 --beta=10.0 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=128 --epochs=40 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --calc_hnet_reg_targets_online --hnet_reg_batch_size=1 --net_type=wrn --wrn_block_depth=4 --wrn_widening_factor=10 --wrn_use_fc_bias --no_bias --hnet_type=chunked_hmlp --hmlp_arch=50,50,50,50 --cond_emb_size=32 --chunk_emb_size=32 --use_cond_chunk_embs --std_normal_temb=1.0 --std_normal_emb=0.1 --store_final_model --mean_only

Task-specific Posterior with BbB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **Resnet-32** achieves 95.49% CL1 and 62.16% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --chmlp_chunk_size=9000 --beta=25 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=mse --batch_size=32 --epochs=60 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.0001 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=16 --chunk_emb_size=16 --std_normal_temb=1.0 --std_normal_emb=0.01 --hyper_gauss_init

The following run with a **WRN-28-10** achieves 92.24% CL1 and 50.23% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --keep_orig_init --chmlp_chunk_size=50000 --beta=0.1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=fkl --batch_size=32 --epochs=60 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=1 --prior_variance=1.0 --kl_scale=1e-05 --calc_hnet_reg_targets_online --hnet_reg_batch_size=1 --net_type=wrn --wrn_block_depth=4 --wrn_widening_factor=10 --wrn_use_fc_bias --no_bias --hnet_type=chunked_hmlp --hmlp_arch=250,500,1000 --cond_emb_size=32 --chunk_emb_size=32 --use_cond_chunk_embs --std_normal_temb=0.1 --std_normal_emb=0.1 --hyper_gauss_init

Task-specific Posterior with Radial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **Resnet-32** achieves 94.67% CL1 and 52.89% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --regularizer=mse --momentum=-1 --chmlp_chunk_size=9000 --beta=10 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --epochs=60 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --prior_variance=1.0 --kl_scale=0.0001 --radial_bnn --num_kl_samples=1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=32 --chunk_emb_size=32 --std_normal_temb=0.01 --std_normal_emb=1.0

Task-specific Posterior with SSGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **Resnet-32** achieves 92.83% CL1 and 51.95% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_ssge.py --momentum=-1 --rbf_kernel_width=1.0 --hh_chmlp_chunk_size=4000 --imp_chmlp_chunk_size=40000 --beta=100 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=64 --epochs=20 --lr=0.0005 --use_adam --clip_grad_norm=100.0 --train_sample_size=10 --prior_variance=1.0 --kl_scale=1e-05 --num_kl_samples=1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --imp_hnet_type=chunked_hmlp --imp_hmlp_arch=10,10,10,10 --imp_chunk_emb_size=32 --imp_hnet_net_act=relu --hh_hnet_type=chunked_hmlp --hh_hmlp_arch=100,100 --hh_cond_emb_size=32 --hh_chunk_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=0.1 --std_normal_emb=0.1 --latent_dim=8 --latent_std=1.0 --full_support_perturbation=0.0002 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=5

Elastic Weight Consolidation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Empirical Fisher failed to properly capture curvature, therefore we couldn't find any multihead runs.

The following run with a **Resnet-32 and growing softmax** achieves 20.40% CL3.

.. code-block:: console 

    $ python3 train_resnet_ewc.py --momentum=-1 --cl_scenario=3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --epochs=40 --lr=0.001 --use_adam --clip_grad_norm=100.0 --prior_variance=1.0 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --during_acc_criterion=90,70,60,-1 --ewc_gamma=1.0 --ewc_lambda=0.01

The following run with a **multihead Resnet-32 with Dirac posterior** achieves 82.50% CL1 and 25.46% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_ewc.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --det_multi_head --batch_size=32 --epochs=40 --lr=0.001 --use_adam --clip_grad_norm=-1 --prior_variance=1.0 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --val_sample_size=1 --ewc_gamma=1.0 --ewc_lambda=1000.0 --n_fisher=200

Variational Continual Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **Resnet-32** achieves 61.09% CL1 and 15.97% CL3-ent.

.. code-block:: console

    $ python3 train_resnet_bbb.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --epochs=40 --lr=0.005 --use_adam --clip_grad_norm=100.0 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.0001 --use_prev_post_as_prior --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --mnet_only

The following run with **Resnet-32 and growing softmax** achieves 19.84% CL3.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --cl_scenario=3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --epochs=80 --lr=0.005 --use_adam --clip_grad_norm=100.0 --train_sample_size=1 --prior_variance=1.0 --kl_scale=0.01 --use_prev_post_as_prior --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --keep_orig_init --mnet_only

Experience Replay
^^^^^^^^^^^^^^^^^

The following run with a **Resnet-32** achieves 41.38% CL3.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --cl_scenario=3 --num_tasks=5 --num_classes_per_task=2 --batch_size=8 --epochs=80 --lr=1e-05 --use_adam --clip_grad_norm=-1 --num_kl_samples=0 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --coreset_size=100 --per_task_coreset --coreset_reg=100.0 --coreset_batch_size=32 --coresets_for_experience_replay --mnet_only --mean_only

The following run with a **Resnet-32** achieves 45.50% CL3.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --cl_scenario=3 --num_tasks=5 --num_classes_per_task=2 --batch_size=64 --epochs=40 --lr=5e-05 --use_adam --clip_grad_norm=-1 --num_kl_samples=0 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --coreset_size=100 --per_task_coreset --coreset_reg=100.0 --coreset_batch_size=-1 --coresets_for_experience_replay --mnet_only --mean_only

Training separate Gaussian main networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this control, we train a separate main network via BbB for each task.

The following run with a **Resnet-32** achieves 96.06% CL1 and 61.35% CL3-ent.

.. code-block:: console

    $ python3 train_resnet_bbb.py --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --epochs=80 --lr=0.01 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.01 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --keep_orig_init --mnet_only

Training separate deterministic main networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can use the code to train seperate deterministic main networks. The option ``--mean_only`` ensures that the Gaussian main network becomes a normal main network. The option ``--main_only`` ensures that we train without a hypernetwork. Via the option ``--train_from_scratch`` we are able to train separate networks.

Hence, this control can be viewed as training an ensemble of size 1 per task.

The following run with  a **Resnet-32** achieves 95.42% CL1 and 58.67% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --beta=0 --train_from_scratch --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --epochs=40 --lr=0.0005 --use_adam --clip_grad_norm=-1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --mnet_only --mean_only


Task-specific Posterior with BbB using coreset-fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **Resnet-32** achieves 92.48% CL1 and 64.76% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --chmlp_chunk_size=9000 --beta=25 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=mse --batch_size=32 --epochs=60 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.0001 --coreset_size=100 --per_task_coreset --final_coresets_finetune --final_coresets_kl_scale=-1 --final_coresets_n_iter=-1 --final_coresets_epochs=-1 --final_coresets_balance=0.5 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=16 --chunk_emb_size=16 --std_normal_temb=1.0 --std_normal_emb=0.01 --hyper_gauss_init

The following run with a **Resnet-32** achieves 92.34% CL1 and 63.45% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --keep_orig_init --chmlp_chunk_size=9000 --beta=25 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=mse --batch_size=32 --epochs=60 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.0001 --coreset_size=100 --per_task_coreset --final_coresets_finetune --final_coresets_kl_scale=-1 --final_coresets_n_iter=-1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=16 --chunk_emb_size=16 --std_normal_temb=1.0 --std_normal_emb=0.01 --hyper_gauss_init

The following run with a **Resnet-32** achieves 94.68% CL1 and 68.07% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --keep_orig_init --chmlp_chunk_size=9000 --beta=25 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=mse --batch_size=32 --epochs=60 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.0001 --coreset_size=500 --per_task_coreset --final_coresets_finetune --final_coresets_use_random_labels --final_coresets_kl_scale=0.001 --final_coresets_n_iter=-1 --final_coresets_epochs=10 --final_coresets_balance=0.8 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=16 --chunk_emb_size=16 --std_normal_temb=1.0 --std_normal_emb=0.01 --hyper_gauss_init

The following run with a **Resnet-32** achieves 91.87% CL1 and 67.78% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --keep_orig_init --chmlp_chunk_size=9000 --beta=25 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --regularizer=mse --batch_size=32 --epochs=60 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.0001 --coreset_size=500 --per_task_coreset --final_coresets_finetune --final_coresets_use_random_labels --final_coresets_kl_scale=-1 --final_coresets_n_iter=-1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch=100,100 --cond_emb_size=16 --chunk_emb_size=16 --std_normal_temb=1.0 --std_normal_emb=0.01 --hyper_gauss_init

SplitCIFAR-100
--------------

This section focuses on a SplitCIFAR experiment that contains 10 tasks of CIFAR-100 splits with 10 classes each. These experiments always require the arguments ``--num_tasks=10 --num_classes_per_task=10 --skip_tasks=1``.


Fine-Tuning
^^^^^^^^^^^

The following run with a **Resnet-32** achieves 87.85% CL1-during and 18.63% CL1-final.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --lambda_lr_scheduler --beta=0 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --batch_size=32 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=-1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=hmlp --val_set_size=100 --mnet_only --mean_only

The following run with a **Resnet-18** achieves 91.16% CL1-during and 25.91% CL1-final.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --lambda_lr_scheduler --beta=0 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --batch_size=32 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=100.0 --plateau_lr_scheduler --net_type=iresnet --iresnet_use_fc_bias --iresnet_channel_sizes=64,64,128,256,512 --iresnet_blocks_per_group=2,2,2,2 --iresnet_projection_shortcut --no_bias --hnet_type=hmlp --val_set_size=100 --mnet_only --mean_only

Training separate deterministic main networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
The following run with a **Resnet-32** achieves 85.24% CL1 and 42.28% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --lambda_lr_scheduler --train_from_scratch --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --batch_size=32 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=-1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=hmlp --val_set_size=100 --mnet_only --mean_only

The following run with a **Resnet-18** achieves 89.52% CL1 and 50.80% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --lambda_lr_scheduler --train_from_scratch --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --batch_size=32 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=-1 --plateau_lr_scheduler --net_type=iresnet --iresnet_use_fc_bias --iresnet_channel_sizes=64,64,128,256,512 --iresnet_blocks_per_group=2,2,2,2 --iresnet_projection_shortcut --no_bias --hnet_type=hmlp --val_set_size=100 --mnet_only --mean_only

Training separate posteriors via BbB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **Resnet-18** achieves 82.73% CL1 and 38.86% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --train_from_scratch --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --batch_size=64 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=100.0 --train_sample_size=10 --prior_variance=10.0 --kl_scale=0.001 --net_type=iresnet --iresnet_use_fc_bias --iresnet_channel_sizes=64,64,128,256,512 --iresnet_blocks_per_group=2,2,2,2 --iresnet_projection_shortcut --no_bias --hnet_type=hmlp --keep_orig_init --val_set_size=100 --mnet_only

Deterministic Task-specific Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **Resnet-32** achieves 78.58% CL1 and 34.59% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --lambda_lr_scheduler --chmlp_chunk_size=7000 --beta=10 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --batch_size=32 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=100.0 --plateau_lr_scheduler --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=32 --chunk_emb_size=16 --hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=0.01 --val_set_size=100 --mean_only

The following run with a **Resnet-32 and much bigger hypernetwork** achieves 85.43% CL1 and 41.08% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --lambda_lr_scheduler --chmlp_chunk_size=10000 --beta=0.01 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --batch_size=32 --epochs=200 --lr=0.0001 --use_adam --clip_grad_norm=100.0 --plateau_lr_scheduler --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch=250,500,1000 --cond_emb_size=32 --chunk_emb_size=32 --hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=1.0 --val_set_size=100 --mean_only

The following run with a **Resnet-32, much bigger hypernetwork and stochastic regularization** achieves 82.58% CL1 and 39.12% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --lambda_lr_scheduler --chmlp_chunk_size=150000 --beta=1 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --batch_size=64 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=-1 --hnet_reg_batch_size=1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=16 --chunk_emb_size=32 --hnet_net_act=sigmoid --std_normal_temb=1.0 --std_normal_emb=0.01 --val_set_size=100 --mean_only

The following run with a **Resnet-18** achieves 85.16% CL1 and 40.35% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --lambda_lr_scheduler --chmlp_chunk_size=150000 --beta=100 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --batch_size=64 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=-1 --net_type=iresnet --iresnet_use_fc_bias --iresnet_channel_sizes=64,64,128,256,512 --iresnet_blocks_per_group=2,2,2,2 --iresnet_projection_shortcut --no_bias --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=16 --chunk_emb_size=16 --hnet_net_act=relu --std_normal_temb=0.01 --std_normal_emb=0.01 --val_set_size=100 --mean_only

The following run with a **Resnet-18 and stochastic regularization** achieves 84.57% CL1 and 40.68% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --lambda_lr_scheduler --chmlp_chunk_size=150000 --beta=50 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --batch_size=32 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=-1 --plateau_lr_scheduler --hnet_reg_batch_size=1 --net_type=iresnet --iresnet_use_fc_bias --iresnet_channel_sizes=64,64,128,256,512 --iresnet_blocks_per_group=2,2,2,2 --iresnet_projection_shortcut --no_bias --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=16 --chunk_emb_size=16 --hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=0.01 --val_set_size=100 --mean_only

Task-specific Posterior with BbB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **Resnet-32, much bigger hypernetwork and stochastic regularization** achieves 85.39% CL1 and 41.14% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --lambda_lr_scheduler --chmlp_chunk_size=200000 --beta=25 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --regularizer=mse --batch_size=32 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=0.005 --hnet_reg_batch_size=1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch=100,100,100,100 --cond_emb_size=16 --chunk_emb_size=32 --hnet_net_act=sigmoid --std_normal_temb=1.0 --std_normal_emb=0.01 --hyper_gauss_init --val_set_size=100

The following run with a **Resnet-18** achieves 84.78% CL1 and 42.36% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --keep_orig_init --chmlp_chunk_size=300000 --beta=50 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --regularizer=w2 --batch_size=32 --epochs=120 --lr=0.0005 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=1e-05 --net_type=iresnet --iresnet_use_fc_bias --iresnet_channel_sizes=64,64,128,256,512 --iresnet_blocks_per_group=2,2,2,2 --iresnet_projection_shortcut --no_bias --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=16 --chunk_emb_size=32 --hnet_net_act=relu --std_normal_temb=1.0 --std_normal_emb=0.01 --hyper_gauss_init --val_set_size=100 --during_acc_criterion=75

The following run with a **Resnet-18 and stochastic regularization** achieves 86.56% CL1 and 45.22% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --lambda_lr_scheduler --chmlp_chunk_size=300000 --beta=5 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --regularizer=w2 --batch_size=64 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --kl_scale=1e-06 --hnet_reg_batch_size=1 --net_type=iresnet --iresnet_use_fc_bias --iresnet_channel_sizes=64,64,128,256,512 --iresnet_blocks_per_group=2,2,2,2 --iresnet_projection_shortcut --no_bias --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=32 --chunk_emb_size=32 --hnet_net_act=sigmoid --std_normal_temb=1.0 --std_normal_emb=1.0 --hyper_gauss_init --val_set_size=100

The following run with a **Resnet-18 and stochastic regularization** achieves 86.16% CL1 and 43.31% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_bbb.py --momentum=-1 --lambda_lr_scheduler --chmlp_chunk_size=300000 --beta=10 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --regularizer=w2 --batch_size=64 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=100.0 --plateau_lr_scheduler --train_sample_size=10 --prior_variance=1.0 --kl_scale=1e-05 --hnet_reg_batch_size=1 --net_type=iresnet --iresnet_use_fc_bias --iresnet_channel_sizes=64,64,128,256,512 --iresnet_blocks_per_group=2,2,2,2 --iresnet_projection_shortcut --no_bias --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=16 --chunk_emb_size=32 --hnet_net_act=sigmoid --std_normal_temb=1.0 --std_normal_emb=1.0 --hyper_gauss_init --val_set_size=100

EWC
^^^

The following run with a **multihead Resnet-32 with Dirac posterior** achieves 63.36% CL1 and 14.20% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_ewc.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --det_multi_head --batch_size=32 --epochs=200 --lr=0.0005 --use_adam --clip_grad_norm=-1 --prior_variance=1.0 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --val_set_size=100 --ewc_gamma=1.0 --ewc_lambda=1000.0 --n_fisher=200 

The following run with a **multihead Resnet-18 with Dirac posterior** achieves 66.83% CL1 and 16.96% CL3-ent.

.. code-block:: console 

    $ python3 train_resnet_ewc.py --momentum=-1 --cl_scenario=3 --split_head_cl3 --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --det_multi_head --batch_size=32 --epochs=80 --lr=0.0005 --use_adam --clip_grad_norm=-1 --prior_variance=1.0 --net_type=iresnet --iresnet_use_fc_bias --iresnet_channel_sizes=64,64,128,256,512 --iresnet_blocks_per_group=2,2,2,2 --iresnet_projection_shortcut --no_bias --val_set_size=100 --ewc_gamma=1.0 --ewc_lambda=100000.0 --n_fisher=200
