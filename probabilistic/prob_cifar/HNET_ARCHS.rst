Hypernet architectures that adhere the compression ratio
========================================================

SplitCIFAR-10/100
-----------------

These architectures are for the task setup ``--num_tasks=6 --num_classes_per_task=10``.

Resnet-32
^^^^^^^^^

The default resnet architecture ``--net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64"`` with 468,540 weights.

  - ``''`` -> 7000
  - ``'100,100'`` -> 4000
  - ``'10,10,10,10'`` -> 40000
  - ``'125,250,500'`` -> 500

Example call:

.. code-block:: console

    $ python3 train_resnet_ssge.py --num_tasks=6 --num_classes_per_task=10 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64" --imp_hnet_type=chunked_hmlp --latent_dim=32 --imp_chunk_emb_size=32 --hh_hnet_type=chunked_hmlp --hh_cond_emb_size=32 --hh_chunk_emb_size=32 --hh_hmlp_arch='' --hh_chmlp_chunk_size=7000 --imp_hmlp_arch='125,250,500' --imp_chmlp_chunk_size=500

Gaussian Resnet-32
^^^^^^^^^^^^^^^^^^

The default resnet architecture ``--net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64"`` with 937,080 weights.

  - ``''`` -> 14000
  - ``'100,100'`` -> 9000
  - ``'10,10,10,10'`` -> 90000
  - ``'125,250,500'`` -> 1500

Example call:

.. code-block:: console

    $ python3 train_resnet_bbb.py --num_tasks=6 --num_classes_per_task=10 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64" --hnet_type=chunked_hmlp --cond_emb_size=32 --chunk_emb_size=32 --hmlp_arch='10,10,10,10' --chmlp_chunk_size=90000

WRN-28-10
^^^^^^^^^

The WRN architecture ``--net_type=wrn --wrn_block_depth=4 --wrn_widening_factor=10 --wrn_use_fc_bias --no_bias`` with 36,511,244 weights.

  - ``''`` -> 500000
  - ``'50,50,50,50'`` -> 700000
  - ``'250,500,1000'`` -> 30000

Example call:

.. code-block:: console

    $ python3 train_resnet_ssge.py --num_tasks=6 --num_classes_per_task=10 --net_type=wrn --wrn_block_depth=4 --wrn_widening_factor=10 --wrn_use_fc_bias --no_bias --imp_hnet_type=chunked_hmlp --latent_dim=32 --imp_chunk_emb_size=32 --hh_hnet_type=chunked_hmlp --hh_cond_emb_size=32 --hh_chunk_emb_size=32 --hh_hmlp_arch='' --hh_chmlp_chunk_size=500000 --imp_hmlp_arch='250,500,1000' --imp_chmlp_chunk_size=30000

Gaussian WRN-28-10
^^^^^^^^^^^^^^^^^^

The WRN architecture ``--net_type=wrn --wrn_block_depth=4 --wrn_widening_factor=10 --wrn_use_fc_bias --no_bias`` with 73,022,488 weights.

  - ``''`` -> 1000000
  - ``'50,50,50,50'`` -> 1000000
  - ``'250,500,1000'`` -> 50000

Example call:

.. code-block:: console

    $ python3 train_resnet_bbb.py --num_tasks=6 --num_classes_per_task=10 --net_type=wrn --wrn_block_depth=4 --wrn_widening_factor=10 --wrn_use_fc_bias --no_bias --hnet_type=chunked_hmlp --cond_emb_size=32 --chunk_emb_size=32 --hmlp_arch='50,50,50,50' --chmlp_chunk_size=1000000

SplitCIFAR-10
-------------

These architectures are for the task setup ``--num_tasks=5 --num_classes_per_task=2``.

Resnet-32
^^^^^^^^^

The default resnet architecture ``--net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64"`` with 465,290 weights.

  - ``''`` -> 7000
  - ``'100,100'`` -> 4000
  - ``'10,10,10,10'`` -> 40000
  - ``'125,250,500'`` -> 500

Example call:

.. code-block:: console

    $ python3 train_resnet_ssge.py --num_tasks=5 --num_classes_per_task=2 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64" --imp_hnet_type=chunked_hmlp --latent_dim=32 --imp_chunk_emb_size=32 --hh_hnet_type=chunked_hmlp --hh_cond_emb_size=32 --hh_chunk_emb_size=32 --hh_hmlp_arch='' --hh_chmlp_chunk_size=7000 --imp_hmlp_arch='125,250,500' --imp_chmlp_chunk_size=500

The Structured HNET ``--imp_hnet_type=structured_hmlp`` uses 8 internal hypernetworks with the following output sizes

  - 1: 480 (1 chunk, 0.1% of total hnet output)
  - 2: 2352 (1 chunk, 0.5% of total hnet output)
  - 3: 2352 (9 chunks, 4.5% of total hnet output)
  - 4: 4704 (1 chunk, 1% of total hnet output)
  - 5: 9312 (9 chunks, 18% of total hnet output)
  - 6: 18624 (1 chunk, 4% of total hnet output)
  - 7: 37056 (9 chunks, 71.7% of total hnet output)
  - 8: 650 (1 chunk, 0.14% of total hnet output)

Here are possible architecture choices:

  - ``--imp_hnet_type=structured_hmlp --latent_dim=3 --imp_chunk_emb_size="0,0,3,0,3,0,3,0" --imp_hmlp_arch=""``
  - ``--imp_hnet_type=structured_hmlp --latent_dim=8 --imp_chunk_emb_size="0,0,8,0,8,0,8,0" --imp_hmlp_arch="32,3;32,3;32,6;64,32,3;64,32,6;64,32,3;64,32,6;32,3"``

Example call:

.. code-block:: console

    $ python3 train_resnet_ssge.py --num_tasks=5 --num_classes_per_task=2 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64" --imp_hnet_type=structured_hmlp --latent_dim=3 --imp_chunk_emb_size="0,0,3,0,3,0,3,0" --hh_hnet_type=chunked_hmlp --hh_cond_emb_size=32 --hh_chunk_emb_size=32 --hh_hmlp_arch='' --hh_chmlp_chunk_size=7000 --imp_hmlp_arch=''

When using Structured HNET ``--imp_hnet_type=structured_hmlp`` with option ``--imp_shmlp_gcd_chunking`` uses 6 internal hypernetworks with the following output sizes

  - 1: 480 (1 chunk, 0.1% of total hnet output)
  - 2: 2352 (1 chunk, 0.5% of total hnet output)
  - 3: 2352 (11 chunks, 5.6% of total hnet output)
  - 4: 4656 (22 chunks, 22% of total hnet output)
  - 5: 9264 (36 chunks, 71.7% of total hnet output)
  - 6: 650 (1 chunk, 0.1% of total hnet output)

Here are possible architecture choices:

  - ``--imp_hnet_type=structured_hmlp --imp_shmlp_gcd_chunking --latent_dim=8 --imp_chunk_emb_size="0,0,16,16,16,0" --imp_hmlp_arch=""``
  - ``--imp_hnet_type=structured_hmlp --imp_shmlp_gcd_chunking --latent_dim=32 --imp_chunk_emb_size="0,0,32,32,32,0" --imp_hmlp_arch="32,8;32,8;64,32,24;64,32,24;64,32,24;32,8"``

Gaussian Resnet-32
^^^^^^^^^^^^^^^^^^

The default resnet architecture ``--net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64"`` with 930,580 weights.

  - ``''`` -> 14000
  - ``'100,100'`` -> 9000
  - ``'10,10,10,10'`` -> 80000
  - ``'125,250,500'`` -> 1450

Example call:

.. code-block:: console

    $ python3 train_resnet_bbb.py --num_tasks=5 --num_classes_per_task=2 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64" --hnet_type=chunked_hmlp --cond_emb_size=32 --chunk_emb_size=32 --hmlp_arch='10,10,10,10' --chmlp_chunk_size=80000

The Structured HNET ``--imp_hnet_type=structured_hmlp`` uses 16 internal hypernetworks (12 when using ``--shmlp_gcd_chunking``). Here are possible architecture choices:

  - ``--hnet_type=structured_hmlp --cond_emb_size=3 --chunk_emb_size="0,0,3,0,3,0,3,0,0,0,3,0,3,0,3,0" --hmlp_arch=""``
  - ``--hnet_type=structured_hmlp --cond_emb_size=8 --chunk_emb_size="0,0,8,0,8,0,8,0,0,0,8,0,8,0,8,0" --hmlp_arch="32,3;32,3;32,6;64,32,3;64,32,6;64,32,3;64,32,6;32,3;32,3;32,3;32,6;64,32,3;64,32,6;64,32,3;64,32,6;32,3"``
  - ``--hnet_type=structured_hmlp --shmlp_gcd_chunking --cond_emb_size=8 --chunk_emb_size="0,0,16,16,16,0,0,0,16,16,16,0" --hmlp_arch=""``
  - ``--hnet_type=structured_hmlp --shmlp_gcd_chunking --cond_emb_size=32 --chunk_emb_size="0,0,32,32,32,0,0,0,32,32,32,0" --hmlp_arch="32,8;32,8;64,32,24;64,32,24;64,32,24;32,8;32,8;32,8;64,32,24;64,32,24;64,32,24;32,8"``

Example call:

.. code-block:: console

    $ python3 train_resnet_bbb.py --num_tasks=5 --num_classes_per_task=2 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64" --hnet_type=structured_hmlp --cond_emb_size=3 --chunk_emb_size="0,0,3,0,3,0,3,0,0,0,3,0,3,0,3,0" --hmlp_arch=""

WRN-28-10
^^^^^^^^^

The WRN architecture ``--net_type=wrn --wrn_block_depth=4 --wrn_widening_factor=10 --wrn_use_fc_bias --no_bias`` with 36,479,194 weights.

  - ``''`` -> 500000
  - ``'50,50,50,50'`` -> 700000
  - ``'250,500,1000'`` -> 30000

Example call:

.. code-block:: console

    $ python3 train_resnet_ssge.py --num_tasks=5 --num_classes_per_task=2 --net_type=wrn --wrn_block_depth=4 --wrn_widening_factor=10 --wrn_use_fc_bias --no_bias --imp_hnet_type=chunked_hmlp --latent_dim=32 --imp_chunk_emb_size=32 --hh_hnet_type=chunked_hmlp --hh_cond_emb_size=32 --hh_chunk_emb_size=32 --hh_hmlp_arch='' --hh_chmlp_chunk_size=500000 --imp_hmlp_arch='250,500,1000' --imp_chmlp_chunk_size=30000

The Structured HNET ``--imp_hnet_type=structured_hmlp`` uses 9 internal hypernetworks with the following output sizes

  - 1: 464 (1 chunk, 0% of total hnet output)
  - 2: 23360 (1 chunk, 0.1% of total hnet output)
  - 3: 230720 (7 chunks, 4.4% of total hnet output)
  - 4: 461440 (15 chunks, 19% of total hnet output)
  - 5: 1844480 (15 chunks, 75.8% of total hnet output)
  - 6: 2560 (1 chunk, 0% of total hnet output)
  - 7: 51200 (1 chunk, 0.1% of total hnet output)
  - 8: 204800 (1 chunk, 0.6% of total hnet output)
  - 9: 6410 (1 chunk, 0% of total hnet output)

Here are possible architecture choices:

  - ``--imp_hnet_type=structured_hmlp --latent_dim=8 --imp_chunk_emb_size="0,0,2,4,4,0,0,0,0" --imp_hmlp_arch=""``
  - ``--imp_hnet_type=structured_hmlp --latent_dim=32 --imp_chunk_emb_size="0,0,32,32,32,0,0,0,0" --imp_hmlp_arch="32,8;32,8;64,32,10;64,32,12;64,32,12;32,8;32,8;32,8;32,8"``

Example call:

.. code-block:: console

    $ python3 train_resnet_ssge.py --num_tasks=5 --num_classes_per_task=2 --net_type=wrn --wrn_block_depth=4 --wrn_widening_factor=10 --wrn_use_fc_bias --no_bias --imp_hnet_type=structured_hmlp --latent_dim=8 --imp_chunk_emb_size="0,0,2,4,4,0,0,0,0" --hh_hnet_type=chunked_hmlp --hh_cond_emb_size=32 --hh_chunk_emb_size=32 --hh_hmlp_arch='' --hh_chmlp_chunk_size=500000 --imp_hmlp_arch=''

When using Structured HNET ``--imp_hnet_type=structured_hmlp`` with option ``--imp_shmlp_gcd_chunking`` uses 8 internal hypernetworks with the following output sizes

  - 1: 464 (1 chunk, 0% of total hnet output)
  - 2: 23360 (71 chunk, 4.5% of total hnet output)
  - 3: 461440 (15 chunks, 19% of total hnet output)
  - 4: 1844480 (15 chunks, 75.8% of total hnet output)
  - 5: 2560 (1 chunk, 0% of total hnet output)
  - 6: 51200 (1 chunk, 0.1% of total hnet output)
  - 7: 204800 (1 chunk, 0.6% of total hnet output)
  - 8: 6410 (1 chunk, 0% of total hnet output)

Here are possible architecture choices:

  - ``--imp_hnet_type=structured_hmlp --imp_shmlp_gcd_chunking --latent_dim=5 --imp_chunk_emb_size="0,16,8,8,0,0,0,0" --imp_hmlp_arch=""``
  - ``--imp_hnet_type=structured_hmlp --imp_shmlp_gcd_chunking --latent_dim=32 --imp_chunk_emb_size="0,32,32,32,0,0,0,0" --imp_hmlp_arch="32,5;64,32,21;64,32,13;64,32,13;32,5;32,5;32,5;32,5"``

Gaussian WRN-28-10
^^^^^^^^^^^^^^^^^^

The WRN architecture ``--net_type=wrn --wrn_block_depth=4 --wrn_widening_factor=10 --wrn_use_fc_bias --no_bias`` with 73,022,488 weights.

  - ``''`` -> 1000000
  - ``'50,50,50,50'`` -> 1000000
  - ``'250,500,1000'`` -> 50000

Example call:

.. code-block:: console

    $ python3 train_resnet_bbb.py --num_tasks=5 --num_classes_per_task=2 --net_type=wrn --wrn_block_depth=4 --wrn_widening_factor=10 --wrn_use_fc_bias --no_bias --hnet_type=chunked_hmlp --cond_emb_size=32 --chunk_emb_size=32 --hmlp_arch='50,50,50,50' --chmlp_chunk_size=1000000

The Structured HNET ``--imp_hnet_type=structured_hmlp`` uses 18 internal hypernetworks (16 when using ``--shmlp_gcd_chunking``). Here are possible architecture choices:

  - ``--hnet_type=structured_hmlp --cond_emb_size=8 --chunk_emb_size="0,0,2,4,4,0,0,0,0,0,0,2,4,4,0,0,0,0" --hmlp_arch=""``
  - ``--hnet_type=structured_hmlp --cond_emb_size=32 --chunk_emb_size="0,0,32,32,32,0,0,0,0,0,0,32,32,32,0,0,0,0" --hmlp_arch="32,8;32,8;64,32,10;64,32,12;64,32,12;32,8;32,8;32,8;32,8;32,8;32,8;64,32,10;64,32,12;64,32,12;32,8;32,8;32,8;32,8"``
  - ``--hnet_type=structured_hmlp --shmlp_gcd_chunking --cond_emb_size=5 --chunk_emb_size="0,16,8,8,0,0,0,0,0,16,8,8,0,0,0,0" --hmlp_arch=""``
  - ``--hnet_type=structured_hmlp --shmlp_gcd_chunking --cond_emb_size=32 --chunk_emb_size="0,32,32,32,0,0,0,0,0,32,32,32,0,0,0,0" --hmlp_arch="32,5;64,32,21;64,32,13;64,32,13;32,5;32,5;32,5;32,5;32,5;64,32,21;64,32,13;64,32,13;32,5;32,5;32,5;32,5"``

SplitCIFAR-100
-----------------

These architectures are for the task setup ``--num_tasks=10 --num_classes_per_task=10 --skip_tasks=1``.

Resnet-18
^^^^^^^^^

The resnet architecture ``--net_type=iresnet --iresnet_use_fc_bias --no_bias --iresnet_projection_shortcut`` with 11,220,032 weights.

  - ``''`` -> 150000
  - ``'500,500'`` -> 20000
  - ``'100,100,100,100'`` -> 100000
  - ``'250,500,1000'`` -> 10000

Example call:

.. code-block:: console

    $ python3 train_resnet_ssge.py --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --net_type=iresnet --iresnet_use_fc_bias --no_bias --iresnet_projection_shortcut --latent_dim=32 --imp_chunk_emb_size=32 --hh_hnet_type=chunked_hmlp --hh_cond_emb_size=32 --hh_chunk_emb_size=32 --hh_hmlp_arch='' --hh_chmlp_chunk_size=150000 --imp_hmlp_arch='250,500,1000' --imp_chmlp_chunk_size=10000

Gaussian Resnet-18
^^^^^^^^^^^^^^^^^^

The resnet architecture ``--net_type=iresnet --iresnet_use_fc_bias --no_bias --iresnet_projection_shortcut`` with 22,440,064 weights.

  - ``''`` -> 300000
  - ``'500,500'`` -> 40000
  - ``'100,100,100,100'`` -> 200000
  - ``'250,500,1000'`` -> 20000

.. code-block:: console

    $ python3 train_resnet_bbb.py --num_tasks=10 --num_classes_per_task=10 --skip_tasks=1 --net_type=iresnet --iresnet_use_fc_bias --no_bias --iresnet_projection_shortcut --hnet_type=chunked_hmlp --cond_emb_size=32 --chunk_emb_size=32 --hmlp_arch='100,100,100,100' --chmlp_chunk_size=200000

Resnet-32
^^^^^^^^^

The default resnet architecture ``--net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64"`` with 471,140 weights.

  - ``''`` -> 7000
  - ``'100,100'`` -> 4000
  - ``'10,10,10,10'`` -> 40000
  - ``'125,250,500'`` -> 500

Gaussian Resnet-32
^^^^^^^^^^^^^^^^^^

The default resnet architecture ``--net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes="16,16,32,64"`` with 942,280 weights.

  - ``''`` -> 14000
  - ``'100,100'`` -> 9000
  - ``'10,10,10,10'`` -> 80000
  - ``'125,250,500'`` -> 1450