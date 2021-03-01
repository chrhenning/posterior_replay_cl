Continual Learning of Regression Tasks in Bayesian Neural Networks
==================================================================

Here, we aim to solve 1D regression problems in a continual learning setting using Bayesian Neural Networks.

The possible sets of tasks to be solved are defined in the function :func:`probabilistic.regression.train_utils.generate_tasks`. A set of tasks can be selected via the CLI option ``--used_task_set``. We used task set 3 ``--used_task_set=3`` for all figures in the paper, as polynomials nicely connect at the boundaries such that even prior-focused methods have a chance of learning them.

Note, not all reported runs (which all represent hpconfigs that led to good polynomial fits and good task inference) in this README have been properly tested for seed robustness.

Validation vs test set
----------------------

In these experiments, we abuse the typical interpretation and standardized ML terminology when referring to a *test* or *validation* set. Usually, we measure an error on a test set to assess the training success and therefore the generalization capabilities of our learning algorithm (being totally agnostic to the fact that we don't know the domain of the test set, e.g., how close it is to the training set). When evaluating Bayesian predictions, we are interesting in accomplishing low training errors while having high predictive uncertainty outside the domain of the training data.

In a regression task, the error is measured in terms of MSE, which is allowed to be high outside the training domain. Therefore, we could evaluate the predictive power inside the training domain using the training set directly. However, since we have total control over the data generation and complete knowledge of the input domains, we decided to create a dedicated **validation set** whose samples all **lie inside the training domain**. In contrast, we **allow samples in the test set to be outside the training domain** in order to generate plots that visualize the transition from low to high uncertainty ranges.

**In summary, the only use of the test set to visualize the predictive uncertainty even beyond the training domain. The validation set on the other hand has no influence on the training (in contrast to its typical usage in the ML field) but is used to determine predictive precision when the task identity is given. The predictive uncertainty on the validation set is used to infer the task if the task identity is unknown.** 

Variance of the likelihood function
-----------------------------------

Note, the variance of the likelihood is an important parameter that should depend on the dataset. Keep in mind, that the std of the likelihood has the same units as the dataset outputs/targets. Depending on the range of these outputs, a *sensible* standard deviation should be chosen via the option ``--ll_dist_std``.

Task Set 1
----------

Here, we consider the polynomial regression task that has already been considered in `von Oswald et al. <https://arxiv.org/abs/1906.00695>`__. Note, unfortunately, the hpsearch for most methods has been run with the default value ``--ll_dist_std=0.1``, even though the correct model (according to the training data's aleatoric uncertainty) requires ``--ll_dist_std=0.05``. However, since lower values of ``ll_dist_std`` usually simplify the training, it should be easier to find hyperparameter configurations when correcting for this mistake. **Note, results from this task set have not been reported in the paper, but the observed trends between posterior-replay methods are identical.**

Per-task diagonal Gaussian posterior approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The training script :mod:`probabilistic.regression.train_bbb` learns an approximation of the posterior parameter distributions per task using the `Bayes by Backprop <https://arxiv.org/abs/1505.05424>`__ (BbB) algorithm. Since BbB is not a Continual Learning algorithm and we are looking at a sequence of tasks, we use a hypernetwork to store the posterior its parameters. Hence, a task embedding is used to extract the posterior its parameters from the hypernetwork. The hypernetwork is protected from forgetting by an L2-regularizer (or an actual divergence/distance between Gaussian distributions; see option ``--regularizer``) on hypernet outputs for previous task-embeddings.

Find out more by running:

.. code-block:: console

    $ python3 train_bbb.py --help

The option ``--loglevel_info`` avoids visual clutter on the console. Plots are written to Tensorboard. If desired, plots can be explicitly shown using the option ``--show_plots``.

Here are a few more good runs. **Note**, that option ``--ll_dist_std`` had been tempered with, because we couldn't find good polynomial fits using higher values.

.. code-block:: console

    $ python3 train_bbb.py --loglevel_info --beta=50 --regularizer=rkl --n_iter=10001 --lr=0.001 --train_sample_size=10 --ll_dist_std=0.01 --local_reparam_trick --net_act=relu --hnet_type=hmlp --hmlp_arch=10 --hnet_net_act=relu --cond_emb_size=8 --disable_lrt_test --used_task_set=1

Below is a run with a sigmoid non-linearity.

.. code-block:: console

    $ python3 train_bbb.py --loglevel_info --beta=50 --regularizer=w2 --n_iter=5001 --lr=0.005 --adam_beta1=0.9 --train_sample_size=1 --ll_dist_std=0.01 --local_reparam_trick --net_act=sigmoid --hnet_type=hmlp --hmlp_arch= --hnet_net_act=sigmoid --cond_emb_size=2 --keep_orig_init --hyper_gauss_init --disable_lrt_test --used_task_set=1

Per-task explicit parametric Radial posterior approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The training script :mod:`probabilistic.regression.train_bbb` with option ``--radial_bnn`` learns an approximation of the posterior parameter distributions per task using a `Radial posterior approximation <https://arxiv.org/pdf/1907.00865.pdf>`__. The CL procedure is similar as above described for BbB.

Example runs can be found below.

.. code-block:: console

    $ python3 train_bbb.py --beta=0.1 --regularizer=mse --n_iter=5001 --lr=0.0001 --clip_grad_norm=1.0 --train_sample_size=1 --prior_variance=1.0 --ll_dist_std=0.1 --radial_bnn --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --hnet_type=hmlp --hmlp_arch="50,50" --cond_emb_size=32 --hnet_net_act=relu --std_normal_temb=1.0 --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_bbb.py --beta=0.05 --regularizer=mse --n_iter=5001 --lr=0.0005 --clip_grad_norm=1.0 --train_sample_size=100 --prior_variance=10.0 --ll_dist_std=0.1 --radial_bnn --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --hnet_type=hmlp --hmlp_arch="25,25" --cond_emb_size=2 --hnet_net_act=relu --std_normal_temb=1.0 --keep_orig_init --use_cuda --used_task_set=1

Per-task implicit posterior via AVB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we use training script :mod:`probabilistic.regression.train_avb` **without** the option ``--prior_focused``. Therewith, we learn a posterior via AVB per task. All posteriors are consolidated within a shared hyper-hypernetwork.

The runs below use a hypernetwork architecture that is **dimension-preserving**.

.. code-block:: console

    $ python3 train_avb.py --beta=1.0 --n_iter=10001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="100,100" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="141,141" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=0.1 --latent_dim=141 --latent_std=1.0 --dis_batch_size=10 --num_dis_steps=1 --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_avb.py --beta=1.0 --n_iter=10001 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="141,141" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=0.1 --latent_dim=141 --latent_std=1.0 --full_support_perturbation=-1 --dis_batch_size=10 --num_dis_steps=1 --use_batchstats --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_avb.py --beta=0.01 --n_iter=10001 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="100,100" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="141,141" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=0.001 --latent_dim=141 --latent_std=1.0 --full_support_perturbation=-1 --dis_batch_size=1 --num_dis_steps=1 --use_batchstats --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_avb.py --beta=0.001 --n_iter=15001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="141,141" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=0.1 --hyper_fan_init --latent_dim=141 --latent_std=1.0 --full_support_perturbation=-1 --dis_batch_size=10 --num_dis_steps=1 --use_cuda --used_task_set=1

The runs below use an **inflating hypernetwork** architecture that obtains a full-support implicit distribution via **noise perturbations** using option ``--full_support_perturbation``.

.. code-block:: console

    $ python3 train_avb.py --beta=10.0 --n_iter=15001 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="10,10" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=0.1 --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.02 --dis_batch_size=10 --num_dis_steps=1 --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_avb.py --beta=0.001 --n_iter=15001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=0.1 --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.02 --dis_batch_size=10 --num_dis_steps=1 --use_batchstats --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_avb.py --beta=0.001 --n_iter=15001 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=1 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=0.1 --latent_dim=2 --latent_std=1.0 --full_support_perturbation=0.02 --dis_batch_size=10 --num_dis_steps=1 --use_batchstats --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_avb.py --beta=10.0 --n_iter=15001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="100,100" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=0.1 --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.02 --dis_batch_size=1 --num_dis_steps=1 --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_avb.py --beta=10.0 --n_iter=15001 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=0.1 --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.02 --dis_batch_size=10 --num_dis_steps=1 --use_batchstats --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_avb.py --beta=0.01 --n_iter=15001 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=0.1 --latent_dim=2 --latent_std=1.0 --full_support_perturbation=0.02 --dis_batch_size=1 --num_dis_steps=1 --use_batchstats --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_avb.py --beta=0.01 --n_iter=10001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="100,100" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="10,10" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=0.1 --latent_dim=2 --latent_std=1.0 --full_support_perturbation=0.02 --dis_batch_size=1 --num_dis_steps=1 --use_cuda --used_task_set=1

Per-task implicit posterior via SSGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we use training script :mod:`probabilistic.regression.train_ssge` **without** the option ``--prior_focused``. Therewith, we learn a posterior via SSGE per task. All posteriors are consolidated within a shared hyper-hypernetwork.

The runs below use a hypernetwork architecture that is **dimension-preserving**.

.. code-block:: console

    $ python3 train_ssge.py --beta=1.0 --n_iter=20001 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="141,141" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=sigmoid --std_normal_temb=1.0 --hyper_fan_init --latent_dim=141 --latent_std=1.0 --full_support_perturbation=-1 --rbf_kernel_width=1.0 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=10 --use_cuda --used_task_set=1

The runs below use an **inflating hypernetwork** architecture that obtains a full-support implicit distribution via **noise perturbations** using option ``--full_support_perturbation``.

.. code-block:: console

    $ python3 train_ssge.py --beta=0.1 --n_iter=20001 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="10,10" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --hyper_fan_init --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.02 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=100 --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_ssge.py --rbf_kernel_width=1.0 --beta=0.01 --n_iter=20001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10,10,10" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=sigmoid --std_normal_temb=1.0 --hyper_fan_init --latent_dim=2 --latent_std=1.0 --full_support_perturbation=0.02 --heuristic_kernel --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=0.9 --ssge_sample_size=100 --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_ssge.py --rbf_kernel_width=1.0 --beta=0.1 --n_iter=10001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --latent_dim=2 --latent_std=1.0 --full_support_perturbation=0.0002 --heuristic_kernel --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=0.9 --ssge_sample_size=10 --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_ssge.py --beta=0.01 --n_iter=5001 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --latent_dim=2 --latent_std=1.0 --full_support_perturbation=0.0002 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=10 --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_ssge.py --beta=0.1 --n_iter=20001 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=1 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --hyper_fan_init --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.0002 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=0.9 --ssge_sample_size=100 --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_ssge.py --rbf_kernel_width=1.0 --beta=0.01 --n_iter=10001 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=1.0 --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.0002 --heuristic_kernel --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=100 --use_cuda --used_task_set=1

(runs below are good but not perfect)

.. code-block:: console

    $ python3 train_ssge.py --beta=1.0 --n_iter=5001 --lr=0.001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="10,10" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.02 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=0.9 --ssge_sample_size=10 --use_cuda --used_task_set=1

.. code-block:: console

    $ python3 train_ssge.py --beta=1.0 --n_iter=20001 --lr=0.01 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.1 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10,10,10" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="10,10" --hh_cond_emb_size=32 --hh_hnet_net_act=sigmoid --std_normal_temb=1.0 --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.0002 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=10 --use_cuda --used_task_set=1

Task Set 3
----------

The task set ``--used_task_set=3`` consists of 2 cubic and 1 quadratic polynomial that are arranged such that a single-head prior-focused CL method has an easier time connecting the solutions of consecutive tasks.

Task-specific Posterior with BbB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ python3 train_bbb.py --beta=10.0 --regularizer=fkl --n_iter=5001 --lr=0.0005 --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --local_reparam_trick --mlp_arch="10,10" --net_act=relu --hnet_type=hmlp --hmlp_arch="10,10" --cond_emb_size=8 --hnet_net_act=relu --std_normal_temb=1.0 --keep_orig_init --used_task_set=3 --use_cuda --disable_lrt_test

.. code-block:: console

    $ python3 train_bbb.py --beta=0.001 --regularizer=mse --n_iter=5001 --lr=0.0005 --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --local_reparam_trick --mlp_arch="10,10" --net_act=relu --hnet_type=hmlp --hmlp_arch="50,50" --cond_emb_size=2 --hnet_net_act=relu --std_normal_temb=1.0 --keep_orig_init --hyper_gauss_init --used_task_set=3 --use_cuda --disable_lrt_test

Task-specific Posterior using Radial approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ python3 train_bbb.py --beta=0.1 --regularizer=mse --batch_size=32 --n_iter=5001 --lr=0.0005 --clip_grad_norm=1.0 --train_sample_size=100 --prior_variance=1.0 --ll_dist_std=0.05 --radial_bnn --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --hnet_type=hmlp --hmlp_arch="100,100" --cond_emb_size=32 --hnet_net_act=relu --std_normal_temb=0.1 --keep_orig_init --hyper_gauss_init --used_task_set=3 --use_cuda

The following call is not very seed robust.

.. code-block:: console

    $ python3 train_bbb.py --beta=1.0 --regularizer=mse --batch_size=8 --n_iter=5001 --lr=0.001 --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --radial_bnn --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --hnet_type=hmlp --hmlp_arch="100,100" --cond_emb_size=8 --hnet_net_act=relu --std_normal_temb=0.1 --keep_orig_init --hyper_gauss_init --used_task_set=3 --use_cuda

Task-specific Posterior with AVB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The runs below use a hypernetwork architecture that is **dimension-preserving**.

.. code-block:: console

    $ python3 train_avb.py --dis_batch_size=32 --beta=1.0 --batch_size=32 --n_iter=20001 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="" --dis_net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="141,141" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --latent_dim=141 --latent_std=1.0 --full_support_perturbation=-1 --num_dis_steps=5 --num_ac_samples=100

.. code-block:: console

    $ python3 train_avb.py --dis_batch_size=32 --beta=0.01 --batch_size=8 --n_iter=20001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="100,100" --dis_net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="141,141" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --latent_dim=141 --latent_std=1.0 --full_support_perturbation=-1 --num_dis_steps=1 --num_ac_samples=100

.. code-block:: console

    $ python3 train_avb.py --dis_batch_size=32 --beta=0.05 --batch_size=8 --n_iter=10001 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="141,141" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="10,10" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --latent_dim=141 --latent_std=1.0 --full_support_perturbation=-1 --num_dis_steps=1 --num_ac_samples=100

The runs below use an **inflating hypernetwork** architecture that obtains a full-support implicit distribution via **noise perturbations** using option ``--full_support_perturbation``.

.. code-block:: console

    $ python3 train_avb.py --beta=0.5 --batch_size=32 --n_iter=20001 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=1.0 --hyper_fan_init --used_task_set=3 --use_cuda --latent_dim=2 --latent_std=1.0 --full_support_perturbation=0.02 --dis_batch_size=32 --num_dis_steps=1 --use_batchstats --num_ac_samples=100

.. code-block:: console

    $ python3 train_avb.py --dis_batch_size=32 --beta=1.0 --batch_size=8 --n_iter=10001 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --dis_net_type=mlp --dis_mlp_arch="10,10" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="10,10" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --latent_dim=2 --latent_std=1.0 --full_support_perturbation=0.02 --num_dis_steps=1 --num_ac_samples=100

The following runs use a sigmoid non-linearity (we were only able to find good fits in this case with AVB).

.. code-block:: console

    $ python3 train_avb.py --beta=0.5 --batch_size=32 --n_iter=10001 --lr=0.0001 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=sigmoid --dis_net_type=mlp --dis_mlp_arch="" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --hyper_fan_init --used_task_set=3 --use_cuda --latent_dim=2 --latent_std=1.0 --full_support_perturbation=0.0002 --dis_batch_size=32 --num_dis_steps=5 --use_batchstats --num_ac_samples=100

.. code-block:: console

    $ python3 train_avb.py --beta=0.05 --batch_size=32 --n_iter=10001 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=sigmoid --dis_net_type=mlp --dis_mlp_arch="" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="10,10" --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=1.0 --hyper_fan_init --used_task_set=3 --use_cuda --latent_dim=2 --latent_std=1.0 --full_support_perturbation=0.02 --dis_batch_size=32 --num_dis_steps=5 --use_batchstats --num_ac_samples=100

Prior-focused Continual Learning with AVB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ python3 train_avb.py --dis_batch_size=32 --batch_size=8 --n_iter=20001 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1000.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=sigmoid --dis_net_type=mlp --dis_mlp_arch="" --dis_net_act=sigmoid --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --latent_dim=32 --latent_std=1.0 --prior_focused --full_support_perturbation=0.02 --num_dis_steps=5 --num_ac_samples=100

.. code-block:: console

    $ python3 train_avb.py --dis_batch_size=1 --batch_size=8 --n_iter=5001 --lr=0.0005 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=0.001 --num_kl_samples=1 --mlp_arch="10,10" --net_act=sigmoid --dis_net_type=mlp --dis_mlp_arch="100,100" --dis_net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --latent_dim=32 --latent_std=1.0 --prior_focused --full_support_perturbation=0.02 --num_dis_steps=1 --num_ac_samples=100

Task-specific Posterior with SSGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The runs below use a hypernetwork architecture that is **dimension-preserving**.

.. code-block:: console

    $ python3 train_ssge.py --beta=0.5 --batch_size=32 --n_iter=20001 --lr=0.0005 --use_adam --clip_grad_norm=1.0 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="141,141" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --latent_dim=141 --latent_std=1.0 --full_support_perturbation=-1 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=0.9 --ssge_sample_size=10

.. code-block:: console

    $ python3 train_ssge.py --rbf_kernel_width=1.0 --beta=0.1 --batch_size=32 --n_iter=20001 --lr=0.0005 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="141,141" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=2 --hh_hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --latent_dim=141 --latent_std=1.0 --full_support_perturbation=-1 --heuristic_kernel --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=10

The runs below use an **inflating hypernetwork** architecture that obtains a full-support implicit distribution via **noise perturbations** using option ``--full_support_perturbation``.

.. code-block:: console

    $ python3 train_ssge.py --beta=1.0 --batch_size=32 --n_iter=20001 --lr=0.001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=relu --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=2 --hh_hnet_net_act=sigmoid --std_normal_temb=1.0 --hyper_fan_init --used_task_set=3 --use_cuda --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.0002 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=10

.. code-block:: console

    $ python3 train_ssge.py --rbf_kernel_width=1.0 --beta=0.01 --batch_size=32 --n_iter=10001 --lr=0.0005 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="100,100" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.0002 --heuristic_kernel --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=10

.. code-block:: console

    $ python3 train_ssge.py --rbf_kernel_width=1.0 --beta=0.01 --batch_size=32 --n_iter=20001 --lr=0.0001 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=1.0 --num_kl_samples=1 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=sigmoid --hh_hnet_type=hmlp --hh_hmlp_arch="10,10" --hh_cond_emb_size=32 --hh_hnet_net_act=relu --std_normal_temb=1.0 --hyper_fan_init --used_task_set=3 --use_cuda --latent_dim=32 --latent_std=1.0 --full_support_perturbation=0.02 --heuristic_kernel --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=10

Prior-focused Continual Learning with SSGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ python3 train_ssge.py --batch_size=32 --n_iter=5001 --lr=0.0005 --use_adam --clip_grad_norm=-1 --train_sample_size=1 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=0.1 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="100,100" --imp_hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --latent_dim=2 --latent_std=1.0 --prior_focused --full_support_perturbation=0.02 --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=1.0 --ssge_sample_size=10

.. code-block:: console

    $ python3 train_ssge.py --rbf_kernel_width=1.0 --batch_size=8 --n_iter=20001 --lr=0.0005 --use_adam --clip_grad_norm=-1 --train_sample_size=10 --prior_variance=1.0 --ll_dist_std=0.05 --kl_scale=0.1 --num_kl_samples=10 --mlp_arch="10,10" --net_act=relu --imp_hnet_type=hmlp --imp_hmlp_arch="10,10" --imp_hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --latent_dim=32 --latent_std=1.0 --prior_focused --full_support_perturbation=0.02 --heuristic_kernel --num_ssge_eigenvals=-1 --thr_ssge_eigenvals=0.9 --ssge_sample_size=100

Hypernetwork-protected Deterministic Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ python3 train_bbb.py --beta=0.1 --regularizer=mse --n_iter=10001 --lr=0.001 --clip_grad_norm=1.0 --train_sample_size=1 --ll_dist_std=0.05 --mlp_arch="10,10" --net_act=relu --hnet_type=hmlp --hmlp_arch="50,50" --cond_emb_size=8 --hnet_net_act=relu --std_normal_temb=1.0 --used_task_set=3 --use_cuda --disable_lrt_test --mean_only

Elastic Weight-Consolidation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ python3 train_ewc.py --n_iter=10001 --lr=0.0005 --use_adam --clip_grad_norm=-1 --prior_variance=1.0 --ll_dist_std=0.05 --mlp_arch="10,10" --net_act=relu --used_task_set=3 --ewc_gamma=1.0 --ewc_lambda=1.0 --n_fisher=-1

Variational Continual Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ python3 train_bbb.py --batch_size=32 --n_iter=10001 --lr=0.001 --clip_grad_norm=1.0 --train_sample_size=100 --prior_variance=1.0 --ll_dist_std=0.05 --use_prev_post_as_prior --mlp_arch="10,10" --net_act=relu --keep_orig_init --used_task_set=3 --use_cuda --mnet_only

Remarks and observations
------------------------

  - The option ``--mnet_only`` only makes sense when training on a single task or from scratch, as there is no protection from forgetting.
  - When using the local reparametrization trick, one should consider activating option ``--disable_lrt_test``. Otherwise, the uncertainties are much more wiggly.
  - ReLUs lead to much higher uncertainties than sigmoidal non-linearities, but the regression fits always look piecewise linear and not nice and smooth.
  - AVB and SSGE seem to have (in general) an easier time to learn polynomials properly while inferring task identity correctly than BbB and Radial.
