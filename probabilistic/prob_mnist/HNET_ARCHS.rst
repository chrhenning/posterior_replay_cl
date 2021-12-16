Hypernet architectures that adhere the compression ratio
========================================================

SplitMNIST
----------

These architectures are for the task setup ``--num_tasks=5 --num_classes_per_task=2``.

Small MLP (100-100)
^^^^^^^^^^^^^^^^^^^

This architecture ``--net_type=mlp --mlp_arch=100,100`` has 89,610 weights.

  - ``''`` -> 1300
  - ``'100,100'`` -> 650
  - ``'10,10,10,10'`` -> 8000
  - ``'100,200,100'`` -> 300

Example call:

.. code-block:: console

    $ python3 train_split_ssge.py --num_tasks=5 --num_classes_per_task=2 --net_type=mlp --mlp_arch=100,100 --imp_hnet_type=chunked_hmlp --latent_dim=32 --imp_chunk_emb_size=32 --hh_hnet_type=chunked_hmlp --hh_cond_emb_size=32 --hh_chunk_emb_size=32 --hh_hmlp_arch='' --hh_chmlp_chunk_size=1300 --imp_hmlp_arch='100,200,100' --imp_chmlp_chunk_size=300

Gaussian Small MLP (100-100)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This architecture ``--net_type=mlp --mlp_arch=100,100`` has 179,220 weights.

  - ``''`` -> 2500
  - ``'100,100'`` -> 1500
  - ``'10,10,10,10'`` -> 16000
  - ``'100,200,100'`` -> 1250

Example call:

.. code-block:: console

    $ python3 train_split_bbb.py --num_tasks=5 --num_classes_per_task=2 --net_type=mlp --mlp_arch=100,100 --hnet_type=chunked_hmlp --cond_emb_size=32 --chunk_emb_size=32 --hmlp_arch='10,10,10,10' --chmlp_chunk_size=16000


Default MLP (400-400)
^^^^^^^^^^^^^^^^^^^^^

**TODO**

Small LeNet
^^^^^^^^^^^

Gaussian Small LeNet
^^^^^^^^^^^^^^^^^^^^

The default small lenet architecture ``--net_type=lenet --lenet_type=mnist_small`` with 43680 weights.

  - ``''`` -> 600
  - ``'50,50'`` -> 700
  - ``'10,10,10,10'`` -> 3750
  - ``'50,75,100'`` -> 200

Example call:

.. code-block:: console

    $ python3 train_split_bbb.py --num_tasks=5 --num_classes_per_task=2 --net_type=lenet --lenet_type=mnist_small --hnet_type=chunked_hmlp --cond_emb_size=32 --chunk_emb_size=32 --hmlp_arch='10,10,10,10' --chmlp_chunk_size=8000

Large LeNet
^^^^^^^^^^^

**TODO**

Resnet-32
^^^^^^^^^

The default resnet architecture ``--net_type=resnet --resnet_block_depth=5
--resnet_channel_sizes="16,16,32,64"`` with 462,730 weights.

  - ``''`` -> 7000
  - ``'100,100'`` -> 4250
  - ``'10,10,10,10'`` -> 40000
  - ``'125,250,500'`` -> 500

Example call:

.. code-block:: console

    $ python3 train_split_bbb.py --mean_only --kl_scale=0 --num_tasks=5 --num_classes_per_task=2 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64" --hnet_type=chunked_hmlp --cond_emb_size=32 --chunk_emb_size=32 --hmlp_arch='10,10,10,10' --chmlp_chunk_size=40000

Gaussian Resnet-32
^^^^^^^^^^^^^^^^^^

The default resnet architecture ``--net_type=resnet --resnet_block_depth=5
--resnet_channel_sizes="16,16,32,64"`` with 925,460 weights.

  - ``''`` -> 14000
  - ``'100,100'`` -> 8500
  - ``'10,10,10,10'`` -> 80000
  - ``'125,250,500'`` -> 1450

Example call:

.. code-block:: console

    $ python3 train_split_bbb.py --num_tasks=5 --num_classes_per_task=2
    --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64" --hnet_type=chunked_hmlp --cond_emb_size=32 --chunk_emb_size=32 --hmlp_arch='10,10,10,10' --chmlp_chunk_size=80000


PermtuedMNIST-10
----------------

These architectures are for the task setup ``--num_tasks=10``.

Small MLP (100-100)
^^^^^^^^^^^^^^^^^^^

This architecture ``--mlp_arch=100,100 --padding=0`` has 98,700 weights.

  - ``''`` -> 1400
  - ``'100,100'`` -> 750
  - ``'10,10,10,10'`` -> 8500
  - ``'100,200,100'`` -> 400

Example call:

.. code-block:: console

    $ python3 train_perm_ssge.py --num_tasks=10 --mlp_arch=100,100 --padding=0 --imp_hnet_type=chunked_hmlp --latent_dim=32 --imp_chunk_emb_size=32 --hh_hnet_type=chunked_hmlp --hh_cond_emb_size=32 --hh_chunk_emb_size=32 --hh_hmlp_arch='' --hh_chmlp_chunk_size=750 --imp_hmlp_arch='100,200,100' --imp_chmlp_chunk_size=400

Gaussian Small MLP (100-100)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This architecture ``--mlp_arch=100,100 --padding=0`` has 197,400 weights.

  - ``''`` -> 2900
  - ``'100,100'`` -> 1750
  - ``'10,10,10,10'`` -> 17000
  - ``'100,200,100'`` -> 1250

Example call:

.. code-block:: console

    $ python3 train_perm_bbb.py --num_tasks=10 --mlp_arch=100,100 --padding=0  --hnet_type=chunked_hmlp --cond_emb_size=32 --chunk_emb_size=32 --hmlp_arch='' --chmlp_chunk_size=2900

Default MLP (1000-1000)
^^^^^^^^^^^^^^^^^^^^^^^

This architecture ``--mlp_arch=1000,1000 --padding=2`` has 2,126,100 weights.

**TODO**

Gaussian Default MLP (1000-1000)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This architecture ``--mlp_arch=1000,1000 --padding=2`` has 4,252,200 weights.

**TODO**

PermtuedMNIST-100
-----------------

These architectures are for the task setup ``--num_tasks=100``.

Default MLP (1000-1000)
^^^^^^^^^^^^^^^^^^^^^^^

This architecture ``--mlp_arch=1000,1000 --padding=2`` has 3,027,000 weights.

  - ``''`` -> 45000
  - ``'100,100'`` -> 29000
  - ``'50,50,50,50'`` -> 58000
  - ``'100,250,500'`` -> 5000

**TODO**

Gaussian Default MLP (1000-1000)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This architecture ``--mlp_arch=1000,1000 --padding=2`` has 6,054,000 weights.

  - ``''`` -> 90000
  - ``'100,100'`` -> 55000
  - ``'50,50,50,50'`` -> 110000
  - ``'100,250,500'`` -> 10000

**TODO**