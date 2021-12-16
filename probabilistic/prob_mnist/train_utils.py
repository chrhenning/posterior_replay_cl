#!/usr/bin/env python3
# Copyright 2019 Christian Henning
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
# title          :probabilistic/prob_mnist/train_utils.py
# author         :ch
# contact        :henningc@ethz.ch
# created        :09/06/2019
# version        :1.0
# python_version :3.6.8
"""
Training utilities
------------------

A collection of helper functions for training scripts in this subpackage.
"""
from argparse import Namespace
import numpy as np
import os
from time import time
import torch
import torch.nn.functional as F
from warnings import warn

from data.dataset import Dataset
from data.special.gmm_data import GMMData
from data.special.permuted_mnist import PermutedMNISTList, PermutedMNIST
from data.special.split_mnist import SplitMNIST, get_split_mnist_handlers
from data.special.split_cifar import SplitCIFAR10Data, SplitCIFAR100Data
from data.special.split_cifar import get_split_cifar_handlers
from probabilistic import GaussianBNNWrapper
from probabilistic import prob_utils as putils
from probabilistic.prob_cifar import hpsearch_config_zenke_bbb as hpzenkebbb
from probabilistic.prob_cifar import hpsearch_config_resnet_bbb as hpresnetbbb
from probabilistic.prob_cifar import train_utils as pcutils
from probabilistic.prob_gmm import train_utils as pgutils
from probabilistic.prob_mnist import hpsearch_config_perm_bbb as hppermbbb
from probabilistic.prob_mnist import hpsearch_config_split_avb as hpsplitavb
from probabilistic.prob_mnist import hpsearch_config_split_bbb as hpsplitbbb
from probabilistic.prob_mnist import hpsearch_config_split_ewc as hpsplitewc
from probabilistic.prob_mnist import hpsearch_config_split_mt as hpsplitmt
from probabilistic.prob_mnist import hpsearch_config_split_ssge as hpsplitssge
from probabilistic.regression import train_utils as rtu
from utils import misc
from utils import hnet_regularizer as hreg
from utils import torch_ckpts as tckpt

def load_datasets(config, logger, experiment, writer,
                  data_dir='../../datasets'):
    """Create a data handler per task.

    Args:
        config: Command-line arguments.
        logger: Logger object.
        experiment (str): Type of experiment. See argument `experiment` of
            function :func:`probabilistic.prob_mnist.train_bbb.run` or
            :func:`probabilistic.prob_cifar.train_avb.run`.
        writer (tensorboardX.SummaryWriter): Tensorboard logger.
        data_dir (str): From where to load (or to where to download) the
            datasets?

    Returns:
        A list of data handlers.
    """
    assert experiment in ['gmm_bbb', 'gmm_avb', 'gmm_avb_pf',
                          'split_bbb', 'perm_bbb',
                          'cifar_zenke_bbb', 'cifar_resnet_bbb',
                          'split_mnist_avb', 'split_mnist_avb_pf',
                          'perm_mnist_avb', 'perm_mnist_avb_pf',
                          'cifar_zenke_avb', 'cifar_zenke_avb_pf',
                          'cifar_resnet_avb', 'cifar_resnet_avb_pf',
                          'gmm_ssge', 'gmm_ssge_pf',
                          'split_mnist_ssge', 'split_mnist_ssge_pf',
                          'perm_mnist_ssge', 'perm_mnist_ssge_pf',
                          'cifar_resnet_ssge', 'cifar_resnet_ssge_pf',
                          'gmm_ewc', 'split_mnist_ewc', 'perm_mnist_ewc',
                          'cifar_resnet_ewc',
                          'gmm_mt', 'split_mnist_mt', 'perm_mnist_mt',
                          'cifar_resnet_mt']

    if experiment.startswith('gmm'):
        logger.info('Running Gaussian-Mixture-Model experiment.')

        dhandlers = pgutils.generate_datasets(config, logger, writer)

    elif experiment.startswith('cifar'):
        logger.info('Running CIFAR-10/100 experiment.')

        # FIXME `shared` not used in current implementation of this function.
        dhandlers = load_cifar(config, None, logger,
                               data_dir='../../datasets')

    elif experiment.startswith('split'):
        logger.info('Running SplitMNIST experiment.')

        dhandlers = get_split_mnist_handlers(data_dir, use_one_hot=True,
            num_tasks=config.num_tasks,
            num_classes_per_task=config.num_classes_per_task,
            validation_size=config.val_set_size)
        num_digits = config.num_tasks * config.num_classes_per_task
        if num_digits < 10:
            logger.info('Running SplitMNIST experiments only for digits ' +
                        '0 - %d.' % (num_digits-1))

    elif experiment.startswith('perm'):
        logger.info('Running PermutedMNIST experiment.')

        # Note, these two options are not existing for all experiment types.
        assert hasattr(config, 'padding') and \
               hasattr(config, 'trgt_padding') and \
               hasattr(config, 'data_random_seed')

        pd = config.padding * 2
        in_shape = [28 + pd, 28 + pd, 1]
        input_dim = np.prod(in_shape)

        rand = np.random.RandomState(config.data_random_seed)
        permutations = [None] + [rand.permutation(input_dim)
                                 for _ in range(config.num_tasks - 1)]
        # NOTE When using `PermutedMNISTList` rather than `PermutedMNIST`,
        # we have to ensure a proper use of the data handlers ourselves. See
        # the corresponding documentation.
        if not experiment.endswith('mt'):
            dhandlers = PermutedMNISTList(permutations, data_dir,
                padding=config.padding, trgt_padding=config.trgt_padding,
                show_perm_change_msg=False,
                validation_size=config.val_set_size)
        else: # Multitask training. Parallel access to all data handlers needed!
            dhandlers = [PermutedMNIST(data_dir, permutation=p,
                padding=config.padding, trgt_padding=config.trgt_padding,
                validation_size=config.val_set_size) for p in permutations]

    else:
        raise ValueError('Experiment type "%s" unknown.' % experiment)

    # Postprocess training sets if configured.
    _shrink_training_set(config, logger, dhandlers)

    assert len(dhandlers) == config.num_tasks


    return dhandlers

def load_cifar(config, shared, logger, data_dir='../datasets'):
    """Create a data handler per task.

    Note:
        Datasets are generated with targets being 1-hot encoded.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Object for sharing data between functions.
            Contains the type of experiment.
        logger: Logger object.
        data_dir (str): From where to load (or to where to download) the
            datasets?

    Returns:
        (list) A list of data handlers (i.e., objects of class
        :class:`data.dataset.Dataset`.
    """
    augment_data = not config.disable_data_augmentation
    #if shared.experiment == 'zenke':
    #    augment_data = False
    #    # To be comparable to previous results. Note, Zenke et al. didn't
    #    # utilize any data augmentation as far as I know.
    #    logger.warning('Data augmentation disabled for Zenkenet.')
    if augment_data:
        logger.info('Data augmentation will be used.')

    logger.info('Loading CIFAR datasets ...')
    ntasks_to_generate = config.num_tasks
    if hasattr(config, 'skip_tasks'):
        skip_tasks = config.skip_tasks
    else:
        skip_tasks = 0
    if skip_tasks > 0:
        ntasks_to_generate += skip_tasks
        logger.info('Generating %d tasks, but the first %d tasks will be ' \
                    % (ntasks_to_generate, skip_tasks) + 'omitted.')
    if hasattr(config, 'num_classes_per_task'):
        num_classes_per_task = config.num_classes_per_task
    else:
        num_classes_per_task = 10
    if hasattr(config, 'val_set_size'):
        validation_size = config.val_set_size
    else:
        validation_size = 0
    dhandlers = get_split_cifar_handlers(data_dir, use_one_hot=True,
        use_data_augmentation=augment_data, num_tasks=ntasks_to_generate,
        num_classes_per_task=num_classes_per_task,
        validation_size=validation_size)
    dhandlers = dhandlers[skip_tasks:]
    assert(len(dhandlers) == config.num_tasks)

    logger.info('Loaded %d CIFAR task(s) into memory.' % config.num_tasks)

    return dhandlers


def _shrink_training_set(config, logger, data_handlers):
    """Shrink the training set of the given data handlers.

    This function shrinks the training set size for the given data handlers
    according to the command-line attribute ``config.training_set_size``.

    Caution:
        This function doesn't respect all the internal attributes of a
        :class:`data.dataset.Dataset` object and simply modifies the
        private attribute ``_data['train_inds']``. Hence, all the training data
        will be left untouched, just certain samples will not be considered
        training samples anymore.

    Args:
        config (argparse.Namespace): Command-line arguments.
        logger (logging.Logger): Console/file logger.
        data_handlers (list): The data handlers whose training set should be
            shrunk.
    """
    if not hasattr(config, 'training_set_size') or \
            not config.training_set_size > 0:
        return

    # Note, a PermutedMNISTList requires special treatment since there is only
    # one underlying data handler.
    dhs = data_handlers
    if isinstance(data_handlers[0], PermutedMNISTList):
        dhs = [data_handlers[0]._data]

    for dh in dhs:
        assert isinstance(dh, Dataset)
        # This function has only be tested on these dataset handlers so far.
        # And since it modifies internal attributes it should only be used with
        # extra caution.
        if not isinstance(dh, (PermutedMNIST, SplitMNIST, SplitCIFAR10Data,
                               SplitCIFAR100Data, GMMData)):
            raise NotImplementedError('This function has not been tested on ' +
                                      'datasets of type %s.' % type(dh))

    for t, dh in enumerate(dhs):
        n_train = dh.num_train_samples
        train_inds = dh._data['train_inds']
        assert train_inds.size == n_train

        if n_train > config.training_set_size:
            # Ensure that the chosen training set is independent of the
            # configured random seed.
            rand = np.random.RandomState(42)
            new_train_inds = rand.choice(train_inds,
                size=config.training_set_size, replace=False)
            new_train_inds = np.sort(new_train_inds)
            assert new_train_inds.size == np.unique(new_train_inds).size
            dh._data['train_inds'] = new_train_inds

            task_msg = 'task %d' % t
            if isinstance(data_handlers[0], PermutedMNISTList):
                task_msg = 'all tasks'
            logger.warn('The training dataset of %s was reduced to %d.' % \
                        (task_msg, config.training_set_size))

    # Sanity check.
    for dh in data_handlers:
        assert dh.num_train_samples <= config.training_set_size

def remove_coreset_from_training_data(config, logger, data_handlers, shared):
    """Remove the samples contained in the coresets from the training dhandler.

    This function removes the samples present in the coresets from the
    data handlers that are used to train sequentially on all tasks. This is
    only used whenever there is a fine-tuning stage after training in all tasks
    that is performed by training on the coresets. This function thus ensures
    that the data in the coresets is used only once, i.e. after the sequential
    training on the individual tasks.

    Note that, if the number of samples in each task is limited by the
    command line option ``training_set_size``, this value will correspond to the
    sum of the number of samples in the coreset and in the datahandlers.
    Therefore, this function needs to be applied after such a shrinkage of the
    training set has been performed.

    This function is similar to
    :func:`probabilistic.train_utils._shrink_training_set`.

    Caution:
        This function doesn't respect all the internal attributes of a
        :class:`data.dataset.Dataset` object and simply modifies the
        private attribute ``_data['train_inds']``. Hence, all the training data
        will be left untouched, just certain samples will not be considered
        training samples anymore.

    Args:
        (....): See docstring of function
            :func:`probabilistic.train_utils._shrink_training_set`.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions (contains filenames). Notably, it contains the information
            about the coresets.
    """
    # Note, a PermutedMNISTList requires special treatment since there is only
    # one underlying data handler.
    dhs = data_handlers
    if isinstance(data_handlers[0], PermutedMNISTList):
        dhs = [data_handlers[0]._data]

        # How to handle this case correctly? We would need to use the same
        # samples for all coresets in method `update_coreset`.
        raise NotImplementedError()

    for dh in dhs:
        assert isinstance(dh, Dataset)
        # This function has only be tested on these dataset handlers so far.
        # And since it modifies internal attributes it should only be used with
        # extra caution.
        if not isinstance(dh, (PermutedMNIST, SplitMNIST, SplitCIFAR10Data,
                               SplitCIFAR100Data, GMMData)):
            raise NotImplementedError('This function has not been tested on ' +
                                      'datasets of type %s.' % type(dh))

    for t, dh in enumerate(dhs):
        n_train = dh.num_train_samples
        train_inds = dh._data['train_inds']
        assert train_inds.size == n_train

        # Get the indices of the samples that are in the coreset.
        task_sample_idx = np.where(shared.task_ident == t)[0]
        coreset_inds = shared.sample_ids[task_sample_idx]

        # Determine the new training ids, without the indices in the coreset.
        new_train_inds_mask = ~np.isin(train_inds, coreset_inds)
        new_train_inds = train_inds[new_train_inds_mask]

        if new_train_inds.size == 0:
            raise ValueError('The size of the training set is too small to ' +
                             'remove the coresets for the sequential training '+
                             'of tasks. Please chose a larger ' +
                             '"training_set_size" or a smaller "coreset_size".')
        assert (new_train_inds.size + coreset_inds.size) == train_inds.size
        dh._data['train_inds'] = new_train_inds

        task_msg = 'task %d' % t
        if isinstance(data_handlers[0], PermutedMNISTList):
            task_msg = 'all tasks'
        logger.debug('The coresets were removed from the training dataset ' +
                    'of %s.' % task_msg)

def generate_gauss_networks(config, logger, data_handlers, device, experiment,
                            no_mnet_weights=None, create_hnet=True):
    """Create main network and potentially the corresponding hypernetwork.

    This function internally uses the function
    :func:`probabilistic.regression.generate_gauss_networks`.

    Args:
        config: Command-line arguments.
        logger: Console (and file) logger.
        data_handlers: List of data handlers, one for each task. Needed to
            extract the number of inputs/outputs of the main network. And to
            infer the number of tasks.
        device: Torch device.
        experiment: Type of experiment. See argument `experiment` of method
            :func:`probabilistic.prob_mnist.train_bbb.run`.
        no_mnet_weights (bool, optional): Whether the main network should not
            have trainable weights. If left unspecified, then the main network
            will only have trainable weights if ``create_hnet`` is ``False``.
        create_hnet (bool): Whether a hypernetwork should be constructed.

    Returns:
        (tuple): Tuple containing:

        - **mnet**: Main network instance.
        - **hnet** (optional): Hypernetwork instance. This return value is
          ``None`` if no hypernetwork should be constructed.
    """
    num_tasks = len(data_handlers)

    # Sanity check!
    for i in range(1, num_tasks):
        assert(np.prod(data_handlers[i].in_shape) == \
               np.prod(data_handlers[0].in_shape))
        assert(data_handlers[i].num_classes == data_handlers[0].num_classes)

    # Note, in CL3 it is actually considered one head, as the softmax is
    # computed across all output neurons in https://arxiv.org/abs/1904.07734.
    # However, due to the modularity of the hypernet approach, we can infer
    # task identity and correct output class without having to compute the
    # softmax over all output neurons.
    num_heads = 1 if config.cl_scenario == 2 else num_tasks

    if experiment.startswith('gmm'):
        net_type = 'mlp'
        in_shape = data_handlers[0].in_shape

    elif experiment.startswith('cifar'):
        in_shape = [32, 32, 3]
        if experiment.startswith('cifar_zenke'):
            net_type = 'zenke'
        else:
            net_type = config.net_type
    else:
        if hasattr(config, 'net_type'):
            net_type = config.net_type
        else:
            net_type = 'mlp'

        assert len(data_handlers[0].in_shape) == 3 # MNIST
        in_shape = data_handlers[0].in_shape
        # Note, that padding is currently only applied when transforming the
        # image to a torch tensor.
        if isinstance(data_handlers[0], PermutedMNIST):
            assert len(data_handlers[0].torch_in_shape) == 3 # MNIST
            in_shape = data_handlers[0].torch_in_shape

    if net_type == 'mlp':
        if len(in_shape) > 1:
            n_x = np.prod(in_shape)
            in_shape = [n_x]
    else:
        assert len(in_shape) == 3
        assert net_type in ['lenet', 'resnet', 'wrn', 'iresnet', 'zenke']

    out_shape = [data_handlers[0].num_classes * num_heads]

    mnet, hnet = rtu.generate_gauss_networks(config, logger, data_handlers,
        device, no_mnet_weights=no_mnet_weights, create_hnet=create_hnet,
        in_shape=in_shape, out_shape=out_shape, net_type=net_type,
        non_gaussian=config.mean_only)

    return mnet, hnet

def compute_acc(task_id, data, mnet, hnet, device, config, shared,
                split_type='test', return_dataset=False, return_entropies=False,
                return_confidence=False, return_agreement=False,
                return_pred_labels=False, return_labels=False,
                return_samples=False, deterministic_sampling=False,
                disable_lrt=False, in_samples=None, out_samples=None,
                num_w_samples=None, w_samples=None, normal_post=None,
                alphas=None):
    """Compute the accuracy over a specified dataset split.

    Note, this function does not explicitly execute the code within a
    ``torch.no_grad()`` context. This needs to be handled from the outside if
    desired.

    The number of weight samples evaluated by this method is determined by the
    argument ``config.val_sample_size``, even for the training split!

    The batch size used by this method is determined by the argument
    ``config.val_batch_size``, even for the training split!

    We expect the networks to be in the correct state (usually the ``eval``
    state).

    The ``task_id`` is used only to select the hypernet embedding (and the
    correct output units depending on the CL scenario).

    Args:
        (....): See docstring of function
            :func:`probabilistic.prob_mnist.train_bbb.train`. Note, ``hnet`` can
            be passed as ``None``. In this case, no weights are passed to the
            ``forward`` method of the main network.
        split_type: The name of the dataset split that should be used:

            - "test": The test set will be used.
            - "val": The validation set will be used. If not available, the test
                set will be used.
            - "train": The training set will be used.

        return_dataset: If ``True``, the attributes ``inputs`` and ``targets``
            will be added to the ``return_vals`` Namespace (see return values).
            Both fields will be filled with numpy arrays.
        return_entropies: If ``True``, the attribute ``entropies`` will be added
            to the ``return_vals`` Namespace (see return values). This field
            will contain the entropy of the predictive distribution for each
            sample. The field will be filled with a numpy array.
            Entropies are normalized by the maximum entropy of the corresponding
            categorical distribution.
        return_confidence: If ``True``, the attribute ``confidence`` will be
            added to the ``return_vals`` Namespace (see return values). This
            field will contain the confidence of the predictive distribution for
            each sample, i.e., the maximum probability. The field will be filled
            with a numpy array.
        return_agreement: If ``True``, the attribute ``agreement`` will be
            added to the ``return_vals`` Namespace (see return values). This
            field will contain the model agreement score for each sample, i.e.,
            the mean over labels for the standard deviations across individual
            model predictions. The field will be filled with a numpy array.

            Note:
                Better agreement means lower score (less deviation across
                models).
        return_pred_labels: If ``True``, the attribute ``pred_labels`` will be
            added to the ``return_vals`` Namespace (see return values). This
            field will contain the predicted label (argmax of the predictive
            distribution) for each sample. The field will be filled with a numpy
            array.
        return_labels: If ``True``, the attribute ``labels`` will be added
            to the ``return_vals`` Namespace (see return values). This field
            will contain the true label (argmax of 1-hot vectors) for each
            sample. The field will be filled with a numpy array.
        return_samples: If ``True``, the attribute ``samples`` will be added
            to the ``return_vals`` Namespace (see return values). This field
            will contain all weight samples used.
            The field will be filled with a numpy array.
        deterministic_sampling (bool): If ``True``, the sampling of weights
            (which incorporates the reparametrization trick) will be done
            deterministically by generating a :class:`torch.Generator` object
            with a fixed random seed, that is used to sample the noise for the
            reparametrization trick.
        disable_lrt (bool): Disable the local-reparametrization trick in the
            forward pass of the main network (if it uses it).
        in_samples (numpy.ndarray, optional): If provided, argument
            ``split_type`` will be ignored. Instead of taking samples from
            ``data``, the input samples provided here are used.
        out_samples (numpy.ndarray, optional): The output samples corresponding
            to ``in_samples``. If ``in_samples`` is provided but ``out_samples``
            is not provided, the returned `accuracy` will be ``None``.
        num_w_samples (int, optional): The number of weight samples used to
            compute an MC estimate of the predictive distribution. If not
            specified, ``config.val_sample_size`` is used. Note, ``1`` is
            enforced for deterministic networks.
        w_samples (list, optional): A list of PyTorch tensors, that can
            be used as weight samples for the main network (and can be
            passed as network weights to the ``forward`` method). If
            provided, those samples are used instead of drawing samples
            from the approximate posterior.
        normal_post (tuple, optional): See docstring of function
            :func:`probabilistic.regression.train_utils.compute_mse`
        alphas (torch.tensor, optional): The coefficients to be used to
            construct the weighted superposition of the embeddings. Only
            relevant if option ``config.supsup_task_inference`` is active.

    Returns:
        (tuple): Tuple containing:

        - **accuracy**: Overall accuracy on dataset split.
        - **return_vals**: A namespace object that contains several attributes,
          depending on the arguments passed. It will always contains:

              - `w_mean`: The current mean values of all synapses in the
                    main network.
              - `w_std`: The current standard deviations of all synapses in
                    the main network (might be ``None`` if deterministic
                    networks are used.
    """
    assert in_samples is not None or split_type in ['test', 'val', 'train']
    assert out_samples is None or in_samples is not None
    assert normal_post is None or hnet is None

    if hasattr(config, 'supsup_task_inference') and \
            config.supsup_task_inference and alphas is not None:

        # Compute the average task embedding, simply computed as the weighted
        # mean of all task-specific embeddings (i.e. weighting via alphas).
        if config.use_cond_chunk_embs or hnet is None:
            raise NotImplementedError()
        task_embeddings = torch.stack( \
            hnet.conditional_params[:shared.num_trained])

        supsup_embedding = torch.mv(task_embeddings.T, alphas)[None, :] # [1,32]

    generator=None
    if deterministic_sampling:
        # FIXME The generator usage is not well documented. I seem to have the
        # following options:
        # >>> generator = torch.Generator(device=device)
        # >>> torch.normal(torch.zeros_like(mean), 1., generator=generator)
        # In which case the random numbers should be generated on the GPU, but
        # I need to create this huge tensor full of zeros for nothing.
        # On the other hand, I can do the following:
        # >>> generator = torch.Generator() # Leave on CPU
        # >>> torch.normal(0., 1., mean.size(), generator=generator).to(device)
        # Though, I consider this option as worse, since epsilon is generated on
        # CPU and then send to the GPU.
        generator = torch.Generator(device=device)
        # Note, PyTorch recommends using large random seeds:
        # https://tinyurl.com/yx7fwrry
        # Note, we want the test noise to be independent of the user set
        # random seed (which should only influence the training via different
        # weight init and different mini-batch assembly). The testing should
        # always behave deterministically.
        generator.manual_seed(2147483647)

    return_vals = Namespace()

    allowed_outputs = out_units_of_task(config, data, task_id,
                                        shared.num_trained)

    ST = shared.softmax_temp[task_id]
    if not hasattr(config, 'calibrate_temp') or not config.calibrate_temp:
        assert ST == 1.

    if in_samples is not None:
        X = in_samples
        T = out_samples
    elif split_type == 'train':
        X = data.get_train_inputs()
        T = data.get_train_outputs()
    elif split_type == 'test' or data.num_val_samples == 0:
        X = data.get_test_inputs()
        T = data.get_test_outputs()
    else:
        X = data.get_val_inputs()
        T = data.get_val_outputs()

    num_samples = X.shape[0]

    gauss_main = isinstance(mnet, GaussianBNNWrapper)

    if T is not None:
        T = fit_targets_to_softmax(config, shared, device, data, task_id, T)

    if return_dataset:
        return_vals.inputs = X
        return_vals.targets = T

    labels = None
    if T is not None:
        labels = np.argmax(T, axis=1)
    if return_labels:
        return_vals.labels = labels

    X = data.input_to_torch_tensor(X, device)
    #if T is not None:
    #    T = data.output_to_torch_tensor(T, device)

    if hnet is None:
        hnet_out = None
    else:
        if hasattr(config, 'supsup_task_inference') and \
                config.supsup_task_inference and alphas is not None:
            hnet_out = hnet.forward(cond_input=supsup_embedding)
        else:
            hnet_out = hnet.forward(cond_id=task_id)

    if normal_post is not None:
        w_mean = normal_post[0]
        w_std = normal_post[1]
    elif not gauss_main:
        w_mean = hnet_out
        w_std = None
    else:
        w_mean, w_rho = mnet.extract_mean_and_rho(weights=hnet_out)
        w_std = putils.decode_diag_gauss(w_rho, logvar_enc=mnet.logvar_encoding)

    return_vals.w_mean = w_mean
    return_vals.w_std = w_std

    # There is no weight sampling when using a deterministic main network.
    if w_samples is not None:
        num_w_samples = len(w_samples)
    elif num_w_samples is None:
        if gauss_main or normal_post is not None:
            num_w_samples = config.val_sample_size
        else:
            num_w_samples = 1
    else:
        if not gauss_main and normal_post is None and num_w_samples > 1:
            warn('Cannot draw multiple weight samples for deterministic ' +
                 'network')
            num_w_samples = 1

    if return_samples:
        num_w = mnet.num_params
        if gauss_main:
            num_w = num_w // 2
        return_vals.samples = torch.empty((num_w_samples, num_w))

    if hasattr(config, 'non_growing_sf_cl3') and config.cl_scenario == 3 \
            and config.non_growing_sf_cl3:
        softmax_width = config.num_tasks * data.num_classes
    elif config.cl_scenario == 3 and not config.split_head_cl3:
        softmax_width = len(allowed_outputs)
    else:
        softmax_width = data.num_classes
    softmax_outputs = torch.empty((num_w_samples, X.shape[0], softmax_width))

    kwargs = mnet_kwargs(config, task_id, mnet)

    for j in range(num_w_samples):
        # Make sure, that we provide the same weight sample for all test samples
        # (i.e., across mini-batches).
        # Note, in case of the local reparam trick, we anyway have a
        # different weight sample per (x,y) pair in the mini-batch.
        if w_samples is not None:
            W = w_samples[j]
            emean = None
            erho = None
        elif normal_post is not None: # Sample weights from a Gaussian posterior.
            W = []
            for ii, pmean in enumerate(normal_post[0]):
                pstd = normal_post[1][ii]
                W.append(torch.normal(pmean, pstd, generator=generator))
            emean = None
            erho = None
        elif not gauss_main:
            W = w_mean
        elif config.local_reparam_trick and not disable_lrt:
            W = None
            emean = w_mean
            erho = w_rho
        else:
            W = putils.sample_diag_gauss(w_mean, w_std, generator=generator,
                is_radial=config.radial_bnn)
            emean = None
            erho = None

        curr_bs = config.val_batch_size
        n_processed = 0

        while n_processed < num_samples:
            if n_processed + curr_bs > num_samples:
                curr_bs = num_samples - n_processed
            n_processed += curr_bs

            sind = n_processed - curr_bs
            eind = n_processed

            if not gauss_main:
                Y = mnet.forward(X[sind:eind, :], weights=W, **kwargs)
            else:
                Y = mnet.forward(X[sind:eind, :], weights=None, mean_only=False,
                    extracted_mean=emean, extracted_rho=erho, sample=W,
                    disable_lrt=disable_lrt, rand_state=generator, **kwargs)
            if allowed_outputs is not None:
                Y = Y[:, allowed_outputs]

            softmax_outputs[j, sind:eind, :] = F.softmax(Y / ST, dim=1)

        if return_samples:
            if W is None:
                return_vals.samples = None
            else:
                return_vals.samples[j, :] = torch.cat([p.detach().flatten() \
                    for p in W]).cpu().numpy()

    # Predictive distribution per sample.
    pred_dists = softmax_outputs.mean(dim=0)

    pred_labels = torch.argmax(pred_dists, dim=1).detach().cpu().numpy()
    # Note, that for CL3 (without split heads) `labels` are already absolute,
    # not relative to the head (see post-processing of targets `T` above).
    if labels is not None:
        accuracy = 100. * np.sum(pred_labels == labels) / num_samples
    else:
        accuracy = None

    if return_pred_labels:
        assert pred_labels.size == X.shape[0]
        return_vals.pred_labels = pred_labels

    if return_entropies:
        # We use the "maximum" trick to improve numerical stability.
        #return_vals.entropies = - np.sum(pred_dists * \
        #                                 np.log(np.maximum(pred_dists, 1e-5)),
        #                                 axis=1)
        return_vals.entropies = - torch.sum(pred_dists * torch.log( \
                torch.where(pred_dists > 1e-5, pred_dists, \
                            1e-5 * torch.ones_like(pred_dists))), axis=1)

        assert return_vals.entropies.numel() == X.shape[0]

        # Normalize by maximum entropy.
        max_ent = - np.log(1.0 / data.num_classes)
        return_vals.entropies /= max_ent

        # FIXME A bit ugly, but we need to backprop through entropies in the
        # case of task selection with the entropy gradient.
        if alphas is None:
            return_vals.entropies = return_vals.entropies.detach().cpu().numpy()

    if return_confidence:
        return_vals.confidence, _ = torch.max(pred_dists, dim=1)
        assert return_vals.confidence.numel() == X.shape[0]

        return_vals.confidence = return_vals.confidence.detach().cpu().numpy()

    if return_agreement:
        # We use `unbiased=False` to be consistent with the numpy default.
        return_vals.agreement = softmax_outputs.std(axis=0,
                                                    unbiased=False).mean(axis=1)
        assert return_vals.agreement.numel() == X.shape[0]

        return_vals.agreement = return_vals.agreement.detach().cpu().numpy()

    return accuracy, return_vals

def out_units_of_task(config, data, task_id, num_trained):
    """Based on the current CL scenario, compute the indices of all output
    neurons of the main network that belong to the given task.

    Args:
        config: Command-line arguments.
        data: A dataset handler (it is assumed that all handlers have the same
            number of classes!).
        task_id: The ID of the task for which the output units should be
            inferred (note, first task has ID 0).
        num_trained (int): Necessary when training unconditioned ``mnet`` with
            growing softmax. The number of already trained tasks.

    Returns:
        A list of integers, each denoting the index of an output neuron of the
        main network belonging to this task (the list contains consecutive
        indices).

        If the whole output belongs to each task (**CL2**), then ``None`` is
        returned.
    """
    if hasattr(config, 'multi_head'):
        assert not hasattr(config, 'cl_scenario')
        if config.multi_head:
            assert len(data.out_shape) == 1
            n_y = data.out_shape[0]
            allowed_outputs = list(range(task_id * n_y, (task_id+1) * n_y))
        else:
            allowed_outputs = None

    elif config.cl_scenario == 1:
        allowed_outputs = list(range(task_id * data.num_classes,
                                     (task_id+1) * data.num_classes))
    elif config.cl_scenario == 2:
        allowed_outputs = None # all outputs
    else:
        if config.split_head_cl3:
            allowed_outputs = list(range(task_id * data.num_classes,
                                         (task_id+1) * data.num_classes))
        elif hasattr(config, 'non_growing_sf_cl3') \
                and config.non_growing_sf_cl3:
            # Full softmax over all tasks.
            #allowed_outputs = list(range(0, config.num_tasks*data.num_classes))
            allowed_outputs = None
        else:
            # Growing head - all output neurons existing so far.
            allowed_outputs = list(range(0, (task_id+1) * data.num_classes))
            ### Important
            # If the model is not task-specific (for instance the output of a
            # hypernet), where another instance already inferred the task-
            # identity by choosing the  model, then I have to always consider
            # all outputs trained so far!
            # FIXME Does that condition cover all cases?
            if hasattr(config, 'ewc_lambda') or config.mnet_only:
                allowed_outputs = list(range(0, num_trained * data.num_classes))

    return allowed_outputs

def save_summary_dict(config, shared, experiment, summary_fn=None):
    """Write a text file in the result folder that gives a quick
    overview over the results achieved so far.

    Args:
        (....): See docstring of method :func:`setup_summary_dict`.
        summary_fn (str, optional): The name of the summary file.
    """
    assert experiment in ['gmm_bbb', 'gmm_avb', 'gmm_avb_pf',
                          'split_bbb', 'perm_bbb',
                          'cifar_zenke_bbb', 'cifar_resnet_bbb',
                          'split_mnist_avb', 'split_mnist_avb_pf',
                          'perm_mnist_avb', 'perm_mnist_avb_pf',
                          'cifar_zenke_avb', 'cifar_zenke_avb_pf',
                          'cifar_resnet_avb', 'cifar_resnet_avb_pf',
                          'gmm_ssge', 'gmm_ssge_pf',
                          'split_mnist_ssge', 'split_mnist_ssge_pf',
                          'perm_mnist_ssge', 'perm_mnist_ssge_pf',
                          'cifar_resnet_ssge', 'cifar_resnet_ssge_pf',
                          'gmm_ewc', 'split_mnist_ewc', 'perm_mnist_ewc',
                          'cifar_resnet_ewc',
                          'gmm_mt', 'split_mnist_mt', 'perm_mnist_mt',
                          'cifar_resnet_mt']

    # "setup_summary_dict" must be called first.
    assert(hasattr(shared, 'summary'))

    if summary_fn == None:
        if experiment == 'gmm_bbb':
            # Note, I just used the same filename in the hpconfig.
            summary_fn = hpsplitbbb._SUMMARY_FILENAME
        elif experiment == 'split_bbb':
            summary_fn = hpsplitbbb._SUMMARY_FILENAME
        elif experiment == 'perm_bbb':
            summary_fn = hppermbbb._SUMMARY_FILENAME
        elif experiment == 'cifar_zenke_bbb':
            summary_fn = hpzenkebbb._SUMMARY_FILENAME
        elif experiment == 'cifar_resnet_bbb':
            summary_fn = hpresnetbbb._SUMMARY_FILENAME
        else:
            # Note, I made sure that all that incorporate AVB use the same
            # filename.
            if 'avb' in experiment:
                summary_fn = hpsplitavb._SUMMARY_FILENAME
            elif 'ssge' in experiment:
                summary_fn = hpsplitssge._SUMMARY_FILENAME
            elif 'ewc' in experiment:
                summary_fn = hpsplitewc._SUMMARY_FILENAME
            elif 'mt' in experiment:
                summary_fn = hpsplitmt._SUMMARY_FILENAME

    with open(os.path.join(config.out_dir, summary_fn), 'w') as f:
        for k, v in shared.summary.items():
            if isinstance(v, list):
                f.write('%s %s\n' % (k, misc.list_to_str(v)))
            elif isinstance(v, float):
                f.write('%s %f\n' % (k, v))
            else:
                f.write('%s %d\n' % (k, v))

def mnet_kwargs(config, task_id, mnet):
    """Keyword arguments that should be passed to the main network.

    In particular, if batch normalization is used in the main network, then
    this function makes sure that the main network chooses the correct set
    if batch statistics in case they are checkpointed (i.e., multiple are
    available).

    Args:
        config (argparse.Namespace): Command-line arguments.
        task_id (int): ID of task to be solved by main network.
        mnet: The main network.

    Returns:
        (dict): A set of keyword arguments that should be passed to the main
        network.
    """
    # We need to tell the main network, which batch statistics to use, in case
    # batchnorm is used and we checkpoint the batchnorm stats.
    mnet_kwargs = {}
    if not config.train_from_scratch and mnet.batchnorm_layers is not None:
        if hasattr(config, 'bn_distill_stats') and config.bn_distill_stats:
            raise NotImplementedError()
        elif not config.bn_no_running_stats and \
                not config.bn_no_stats_checkpointing:
            # Specify current task as condition to select correct
            # running stats.
            mnet_kwargs['condition'] = task_id

    return mnet_kwargs

def checkpoint_bn_stats(config, task_id, mnet):
    """Checkpoint the batchnorm stats in the main network (if needed).

    Args:
        config (argparse.Namespace): Command-line arguments.
        task_id (int): Current task ID.
        mnet: Main network.
    """
    if not config.train_from_scratch and mnet.batchnorm_layers is not None:
        if (not hasattr(config, 'bn_distill_stats') or \
                    not config.bn_distill_stats) and \
                not config.bn_no_running_stats and \
                not config.bn_no_stats_checkpointing:
            # Checkpoint the current running statistics (that have been
            # estimated while training the current task).
            for bn_layer in mnet.batchnorm_layers:
                assert bn_layer.num_stats == task_id+1
                bn_layer.checkpoint_stats()

def apply_lr_schedulers(config, shared, logger, task_id, data, mnet, hnet,
                        device, train_iter, iter_per_epoch, plateau_scheduler,
                        lambda_scheduler, hhnet=None, method='bbb'):
    """Apply learning rate schedulers.

    This function applied the LR schedulers according to CLI options
    ``plateau_lr_scheduler`` and ``lambda_lr_scheduler``.

    Note:
        If no validation set is available, the function will fall back to the
        test set when computing a performance metric for the plateau scheduler.

    Args:
        (....): See docstring of function
            :func:`probabilistic.prob_mnist.train_bbb.train`. Note, ``hnet`` can
            be passed as ``None``.
        train_iter (int): Current training iteration.
        iter_per_epoch (int): See return value of function
            :func:`utils.sim_utils.calc_train_iter`.
        plateau_scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): The
            plateau scheduler or ``None`` if not needed.
        lambda_scheduler (torch.optim.lr_scheduler.LambdaLR): The lambda
            LR scheduler or ``None`` if not needed.
        hhnet (optional): If ``method`` is ``'avb'``. then the hyper-
            hypernetwork might be required to estimate uncertainty on old
            coreset samples.
        method (str): Possible choices: ``'bbb'`` and ``'avb'``. Determines
            how uncertainty is computed, if needed.
    """
    assert method in ['bbb', 'avb', 'ssge']
    assert hhnet is None or method != 'bbb'

    if config.plateau_lr_scheduler:
        if 'regression' in shared.experiment_type:
            raise NotImplementedError()

        assert iter_per_epoch != -1
        if train_iter % iter_per_epoch == 0 and train_iter > 0:
            curr_epoch = train_iter // iter_per_epoch

            split_type = 'test' if data.num_val_samples == 0 else 'validation'
            logger.info('Computing %s accuracy for plateau LR ' % (split_type) +
                        'scheduler (epoch %d) ...' % curr_epoch)
            # We need a validation quantity for the plateau LR scheduler.

            pcutils.set_train_mode(False, mnet, hnet, hhnet, None)

            with torch.no_grad():
                if method == 'bbb':
                    disable_lrt_test = config.disable_lrt_test \
                        if hasattr(config, 'disable_lrt_test') else False
                    test_acc, _ = compute_acc(task_id, data, mnet, hnet, device,
                        config, shared, split_type='val',
                        deterministic_sampling=True,
                        disable_lrt=disable_lrt_test)
                else:
                    test_acc, _ = pcutils.compute_acc(task_id, data, mnet, hnet,
                        hhnet, device, config, shared, split_type='val',
                        deterministic_sampling=True)

            pcutils.set_train_mode(True, mnet, hnet, hhnet, None)

            plateau_scheduler.step(test_acc)

            logger.info('Plateau LR scheduler uses acc %f%% (epoch %d).' %
                        (test_acc, curr_epoch))

    if config.lambda_lr_scheduler:
        assert iter_per_epoch != -1
        if train_iter % iter_per_epoch == 0 and train_iter > 0:
            curr_epoch = train_iter // iter_per_epoch
            logger.info('Applying Lambda LR scheduler (epoch %d).'
                        % curr_epoch)
            lambda_scheduler.step()

def calc_kl_scale(config, num_train_iter, train_iter, logger,
                  final_finetune=False):
    """Calculate the loss scaling of the prior-matching term.

    Either a user-specified config is chosen or a burn-in schedule is followed.

    Args:
        config (argparse.Namespace): Command-line arguments.
        num_train_iter (int): See return value of function
            :func:`utils.sim_utils.calc_train_iter`.
        train_iter (int): Current training iteration.
        logger: Console (and file) logger.
        final_finetune (bool, optional): Whether the kl scale to be looked at
            is for the final fine-tuning stage with coresets.
    """
    kl_scale = config.kl_scale
    if final_finetune and config.final_coresets_kl_scale != -1:
            kl_scale = config.final_coresets_kl_scale

    if config.kl_schedule == 0:
        return kl_scale
    else:
        if kl_scale == 1:
            warn('Option "kl_schedule" has no effect if "kl_scale" is ' +
                 'set to 1.')
            return kl_scale
        #elif kl_scale > 1:
        #    raise ValueError('Option "kl_schedule" is not allowed if ' +
        #                     '"kl_scale" is greater than 1.')

        kl_schedule = np.abs(config.kl_schedule)

        kl_scales = np.linspace(kl_scale, 1, kl_schedule)
        # If the `kl_schedule` is negative, we anneal the KL influence.
        if config.kl_schedule < 0:
            kl_scales = np.flip(kl_scales, axis=0)
        intervals = np.linspace(0, num_train_iter, kl_schedule+1,
                                dtype=np.int)
        # Choose the index of the smallest interval such that the upper
        # boundary is bigger than "train_iter" (implying that the lower
        # boundary is smaller than "train_iter").
        cur_interval = np.argmax(train_iter < intervals) - 1
        assert cur_interval >= 0

        prev_interval = np.argmax((train_iter-1) < intervals) - 1
        if prev_interval >= 0 and prev_interval < cur_interval:
            logger.debug('KL scaling is changed from %f to %f.' % \
                         (kl_scales[prev_interval], kl_scales[cur_interval]))

        return kl_scales[cur_interval]

def checkpoint_nets(config, shared, task_id, mnet, hnet, hhnet=None, dis=None):
    """Checkpoint the main and (if existing) the current hypernet.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions (contains filenames).
        task_id (int): On which task have the networks been trained last.
        mnet: The main network.
        hnet: The hypernetwork. May be ``None``.
        hhnet (optional): The hyper-hypernetwork.
        dis (optional): The discriminator network.
    """
    assert hasattr(shared, 'summary')

    if 'regression' in shared.experiment_type:
        perf_key = 'during_mse_task_%d' % task_id
        perf_val = shared.summary['aa_mse_during'][task_id]
        perf_score = -perf_val
    else:
        # Note, we use the "task-given" during accuracy on purpose, independent
        # of the CL scenario, since the during accuracies for "task-inferred"
        # scenarios don't make a lot of sense (they depend on the number of
        # tasks observed so far).
        perf_key = 'during_acc_task_%d' % task_id
        perf_val = shared.summary['acc_task_given_during'][task_id]
        perf_score = perf_val

    ts = time()

    tckpt.save_checkpoint({'state_dict': mnet.state_dict(),
                           perf_key: perf_val},
        shared.ckpt_mnet_fn % task_id, perf_score, train_iter=None,
        timestamp=ts)
    if hnet is not None:
        tckpt.save_checkpoint({'state_dict': hnet.state_dict(),
                               perf_key: perf_val},
            shared.ckpt_hnet_fn % task_id, perf_score, train_iter=None,
            timestamp=ts)
    if hhnet is not None:
        tckpt.save_checkpoint({'state_dict': hhnet.state_dict(),
                               perf_key: perf_val},
            shared.ckpt_hhnet_fn % task_id, perf_score, train_iter=None,
            timestamp=ts)
    if dis is not None:
        tckpt.save_checkpoint({'state_dict': dis.state_dict(),
                               perf_key: perf_val},
            shared.ckpt_dis_fn % task_id, perf_score, train_iter=None,
            timestamp=ts)

def load_networks(shared, task_id, device, logger, mnet, hnet, hhnet=None,
                  dis=None):
    """Load checkpointed networks.

    Args:
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions (contains filenames).
        task_id (int): On which task have the networks been trained last.
        device: PyTorch device.
        logger: Console (and file) logger.
        mnet: The main network.
        hnet: The hypernetwork. May be ``None``.
        hhnet (optional): The hyper-hypernetwork.
        dis (optional): The discriminator network.

    Returns:
        (float): The performance score of the checkpoint.
    """
    ckpt_dict, score = tckpt.load_checkpoint(shared.ckpt_mnet_fn % task_id,
        mnet, device=device, ret_performance_score=True)
    if hnet is not None:
        tckpt.load_checkpoint(shared.ckpt_hnet_fn % task_id, hnet,
                              device=device)
    if hhnet is not None:
        tckpt.load_checkpoint(shared.ckpt_hhnet_fn % task_id, hhnet,
                              device=device)
    if dis is not None:
        tckpt.load_checkpoint(shared.ckpt_dis_fn % task_id, dis,
                              device=device)

    logger.debug('Loaded network(s) for task %d from checkpoint,' % task_id + \
                 'that has a performance score of %f.' % score)

    return score


def calc_regged_out_inds(config, task_id, data):
    """Calculate the indices of the output heads of all previous tasks.

    This function computes the indices of the output heads for all task IDs
    smaller than ``task_id``. This information is important to one wants to use
    the option ``inds_of_out_heads`` of function
    :func:`utils.hnet_regularizer.calc_fix_target_reg`.

    Args:
        config (argparse.Namespace): Command-line arguments.
        task_id (int): On which task the network is currently trained.
        data: The data loader of the current task.
            
            Note:
                This function assumes the same output shape for all data
                handlers (i.e., all tasks).

    Returns:
        (list): List of lists of integers. ``None`` if no multihead setup is
        used.
    """
    regged_outputs = None
    # Note, CL2 only has one output head.
    if config.cl_scenario != 2:
        regged_outputs = []
        for i in range(task_id):
            regged_outputs.append(out_units_of_task(config, data, i, None))

    return regged_outputs

def calc_reg_target(config, task_id, hnet, mnet=None):
    """Calculate the targets for the regularizer.

    Note, this function simply invokes the function
    :func:`utils.hnet_regularizer.get_current_targets`, except if the
    command-line argument ``regularizer`` exists (i.e., the output of the
    hypernetwork can be interpreted as Gaussian distribution).

    Args:
        config (argparse.Namespace): Command-line arguments.
        task_id (int): On which task the network is currently trained. Targets
            for tasks ``0, ..., task_id-1`` are computed.
        hnet: The hypernetwork.
        mnet (optional): The main network. Only needs to be provided if
            command-line argument ``regularizer`` is provided and
            ``regularizer != 'mse'``.

    Returns:
        (tuple): Tuple containing:

        - **targets**: The targets returned from function
          :func:`utils.hnet_regularizer.get_current_targets`.
        - **target_means** (optional): Is returned as ``None`` except if
          ``mnet`` is provided and command-line argument
          ``regularizer != 'mse'``. Will contain the target means of the
          Gaussian distributions that come out of the hypernetwork for old task
          embeddings.
        - **target_logvars** (optional): Same as return value ``target_means``,
          but containing the log-variances.
    """
    targets = hreg.get_current_targets(task_id, hnet)
    target_means = None
    target_logvars = None

    if hasattr(config, 'regularizer') and config.regularizer != 'mse':
        assert isinstance(mnet, GaussianBNNWrapper)

        # Required to test different regularizers than the default one.
        target_means   = [None] * task_id
        target_logvars = [None] * task_id
        for i in range(task_id):
            target_means[i], rho = \
                mnet.extract_mean_and_rho(weights=targets[i])
            _, target_logvars[i] = putils.decode_diag_gauss(rho, \
                logvar_enc=mnet.logvar_encoding, return_logvar=True)

            # Important, targets have to be detached from the graph. We don't
            # want to backprop through them.
            target_means[i] = [p.detach().clone() for p in target_means[i]]
            target_logvars[i] = [p.detach().clone() for p in target_logvars[i]]

    return targets, target_means, target_logvars

def calc_gauss_reg_all_tasks(config, task_id, mnet, hnet, target_means=None,
                   target_logvars=None, prev_theta=None, prev_task_embs=None,
                   batch_size=None):
    """Calculate a hypernet regularization for a Gaussian weight posteriors.

    This function is simply a wrapper around :func:`calc_gauss_reg` to take
    care of the regularization across several tasks, and allow regularizing
    only on a subset of randomly selected tasks.

    Args:
        (....): See docstring of function :func:`calc_gauss_reg`.
        target_means (list): Per-task mean of the regularization target.
        target_logvars (list): Per-task log-variance of the regularization
            target.
        batch_size (int, optional): If specified, only a random subset of
            previous tasks is regularized. If the given number is bigger than
            the number of previous tasks, all previous tasks are regularized.

            Note:
                A ``batch_size`` smaller or equal to zero will be ignored
                rather than throwing an error.

    Return:
        (float): The regularization loss across the number of specified tasks.
    """
    # Number of tasks to be regularized.
    num_regs = task_id
    ids_to_reg = list(range(num_regs))
    if batch_size is not None and batch_size > 0:
        if num_regs > batch_size:
            ids_to_reg = np.random.choice(num_regs, size=batch_size,
                                          replace=False).tolist()
            num_regs = batch_size

    target_mean = None
    target_logvar = None
    reg = 0
    for i in ids_to_reg:
        if prev_theta is None:
            target_mean = target_means[i]
            target_logvar = target_logvars[i]
        reg += calc_gauss_reg(config, i, mnet, hnet,
                                   target_mean=target_mean,
                                   target_logvar=target_logvar,
                                   prev_theta=prev_theta,
                                   prev_task_embs=prev_task_embs)

    return reg / num_regs


def calc_gauss_reg(config, task_id, mnet, hnet, target_mean=None,
                   target_logvar=None, current_mean=None, current_logvar=None,
                   prev_theta=None, prev_task_embs=None):
    """Calculate a hypernet regularization for a Gaussian weight posterior.

    If the output of the hypernetwork is a Gaussian distribution, then we can
    compute an analytic distance/divergence between past and current hypernet
    outputs and use this as a regularization target.

    Note:
        This function can only be called if command-line argument
        ``regularizer != 'mse'``.

    Args:
        config (argparse.Namespace): Command-line arguments.
        task_id (int): The task ID for which the regularization should be
            computed.
        hnet: The hypernetwork.
        mnet: The main network.
        target_mean: Mean of the regularization target.
        target_logvar: Log-variance of the regularization target.
        current_mean (optional): The mean value currently outputted by the
            hypernet ``hnet`` for task ID ``task_id``. If specified, this method
            won't compute them, i.e., ``mnet`` and ``hnet`` won't be used.
        current_logvar (optional): Similar to ``current_mean``, but
            corresponding to the log-variance outputs.
        prev_theta (optional): If given, ``prev_task_embs`` but not
            ``target_mean`` and not ``target_logvar`` has to be specified.
            ``prev_theta`` is expected to be the internal hypernet weights
            "theta" prior to learning the current task. Hence, it can be
            used to compute the targets on the fly (which is more memory
            efficient (constant memory), but more computationally demanding).
            The computed targets will be detached from the computational graph.
            Independent of the current hypernet mode, the targets are computed
            in "eval" mode.
        prev_task_embs (optional): ``prev_task_embs`` are the task embeddings
            (or conditional parameters) of the hypernetwork for task ``task_id``
            prior to learning the current task (which might have changed those
            parameters).
            See docstring of ``prev_theta`` for more details.

    Return:
        Distance/divergence between past and current ``hnet`` output for task
        ID ``task_id``.
    """
    if not hasattr(config, 'regularizer') or \
            not config.regularizer in ['rkl', 'fkl', 'w2']:
        raise ValueError('Method only applicable when regularizing Gaussian ' +
                         'hypernet outputs.')

    assert (current_mean is None and current_logvar is None) or \
        (current_mean is not None and current_logvar is not None)
    assert(target_mean is None or target_logvar is not None)
    assert(prev_theta is None or prev_task_embs is not None)
    assert(target_mean is not None or prev_theta is not None and \
           target_mean is None or prev_theta is None)

    # FIXME Improve function to allow masking in the multi-head case.

    if prev_theta is not None:
        # Compute targets in eval mode!
        hnet_mode = hnet.training
        hnet.eval()

        # Compute target on the fly using previous hnet.
        with torch.no_grad():
            hnet_out = hnet.forward(cond_id=task_id, weights={
                    'uncond_weights': prev_theta,
                    'cond_weights': prev_task_embs
                })

            target_mean, rho = mnet.extract_mean_and_rho(weights=hnet_out)
            _, target_logvar = putils.decode_diag_gauss(rho, \
                logvar_enc=mnet.logvar_encoding, return_logvar=True)

        target_mean = [p.detach().clone() for p in target_mean]
        target_logvar = [p.detach().clone() for p in target_logvar]

        hnet.train(mode=hnet_mode)

    if current_mean is None:
        hnet_out = hnet.forward(cond_id=task_id)

        w_mean, w_rho = mnet.extract_mean_and_rho( \
            weights=hnet_out)
        _, w_logvar = putils.decode_diag_gauss(w_rho, \
            logvar_enc=mnet.logvar_encoding, return_logvar=True)
    else:
        w_mean = current_mean
        w_logvar = current_logvar

    if config.regularizer == 'rkl': # Reverse KL
        loss_reg = putils.kl_diag_gaussians(w_mean, w_logvar,
                                            target_mean, target_logvar)
    elif config.regularizer == 'fkl': # Forward KL
        loss_reg = putils.kl_diag_gaussians(target_mean, target_logvar,
                                            w_mean, w_logvar)
    elif config.regularizer == 'w2': # Wasserstein-2
        loss_reg = putils.square_wasserstein_2(w_mean, w_logvar,
                                               target_mean, target_logvar)

    return loss_reg

def calc_batch_uncertainty(config, shared, task_id, inputs, mnet, hnet, data,
                           num_w_samples, mnet_weights=None,
                           allowed_outputs=None, disable_lrt=False):
    """Compute the per-sample uncertainties for a given batch of inputs.

    Note:
        This function currently assumes a Gaussian weight posterior.

    Note:
        This function is executed inside a ``torch.no_grad()`` context.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared: Miscellaneous data shared among training functions (softmax
            temperature is stored in here).
        task_id (int): In case a hypernet ``hnet`` is given, the ``task_id`` is
            used to load the corresponding main network ``mnet`` weights.
        inputs (torch.Tensor): A batch of main network ``mnet`` inputs.
        mnet: The main network.
        hnet (optional): The hypernetwork, can be ``None``.
        data: Dataset loader. Needed to determine the number of classes.
        num_w_samples (int): The number of weight samples that should be drawn
            to estimate predictive uncertainty.
        mnet_weights (tuple, optional): To save computation, one can pass
            weights for the main network here, if they have been computed
            prior to calling this methods.

            Example:

                If the main network is an instance of class
                :class:`probabilistic.gauss_mnet_interface.GaussianBNNWrapper`,
                then ``mnet_weights`` should be a tuple of means and rhos of
                the Gaussian weight posterior
                (``mnet_weights=(w_mean, w_rho)``).
        allowed_outputs (tuple, optional): The indices of the neurons belonging
            to outputs head ``task_id``. Only needs to be specified in a
            multi-head setting.
        disable_lrt (bool): See option ``disable_lrt`` of function
            :func:`compute_acc`.

    Returns:
        (numpy.ndarray): The entropy of the estimated predictive distribution
        per input sample.
    """
    assert data.classification
    assert config.cl_scenario == 2 or allowed_outputs is not None
    assert isinstance(mnet, GaussianBNNWrapper) or config.mean_only

    # FIXME We calibrate the temperature after training on a task. This function
    # is currently only used to track batch uncertainty during training or
    # choose coreset samples that have maximum uncertainty on a single model
    # (note, relative order of uncertain samples doesn't change due to
    # calibration for a single model). Hence, the function is invoked before
    # the temperature is optimized.
    # Therefore, I throw an assertion if we use the function in the future for
    # other purposes, just in case the programmer is unaware.
    assert shared.softmax_temp[task_id] == 1.
    ST = shared.softmax_temp[task_id]

    with torch.no_grad():
        if mnet_weights is None:
            if hnet is None:
                hnet_out = None
            else:
                hnet_out = hnet.forward(cond_id=task_id)

            if config.mean_only:
                w_mean = hnet_out
            else:
                w_mean, w_rho = mnet.extract_mean_and_rho(weights=hnet_out)
        else:
            if config.mean_only:
                assert len(mnet_weights) == 1
                w_mean = mnet_weights[0]
            else:
                assert len(mnet_weights) == 2
                assert isinstance(mnet, GaussianBNNWrapper)
                w_mean, w_rho = mnet_weights

        if allowed_outputs is not None:
            num_outs = len(allowed_outputs)
        else:
            num_outs = data.num_classes
        softmax_outputs = np.empty((num_w_samples, inputs.shape[0], num_outs))

        kwargs = mnet_kwargs(config, task_id, mnet)

        for j in range(num_w_samples):
            if config.mean_only:
                Y = mnet.forward(inputs, weights=w_mean, **kwargs)
            else:
                Y = mnet.forward(inputs, weights=None, mean_only=False,
                    extracted_mean=w_mean, extracted_rho=w_rho, sample=None,
                    disable_lrt=disable_lrt, rand_state=None, **kwargs)
            if allowed_outputs is not None:
                Y = Y[:, allowed_outputs]

            softmax_outputs[j, :, :] = F.softmax(Y / ST, dim=1).detach(). \
                cpu().numpy()

        # Predictive distribution per sample.
        pred_dists = softmax_outputs.mean(axis=0)

        # We use the "maximum" trick to improve numerical stability.
        entropies = - np.sum(pred_dists * np.log(np.maximum(pred_dists, 1e-5)),
                             axis=1)
        assert entropies.size == inputs.shape[0]

        # Normalize by maximum entropy.
        max_ent = - np.log(1.0 / data.num_classes)

        return entropies / max_ent

def update_coreset(config, shared, task_id, data, mnet, hnet, device, logger,
                   allowed_outputs, hhnet=None, method='bbb'):
    """Update or create a new coreset.

    This function extracts a random subset of the training set from the data
    handler ``data`` and stores it as a coreset. If command-line option
    ``per_task_coreset`` was set, then the coreset size will be increased
    by ``coreset_size`` samples from ``data``. Otherwise, the
    ``coreset_size // (task_id+1)`` samples with the highest uncertainty
    (under the current model, which is trained on ``data``) are removed from
    the coreset (which always has size ``coreset_size``) and replaced by random
    training samples from ``data``.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Coresets will be stored and maintained
            here.
        task_id (int): In case a hypernet ``hnet`` is given, the ``task_id`` is
            used to load the corresponding main network ``mnet`` weights.
        data: Dataset loader. New data will be added to the coreset from the
            training set of this data loader.
        mnet: The main network.
        hnet (optional): The hypernetwork, can be ``None``.
        device: PyTorch device.
        logger: Console (and file) logger.
        allowed_outputs (tuple, optional): The indices of the neurons belonging
            to outputs head ``task_id``. Only needs to be specified in a
            multi-head setting.
        hhnet (optional): If ``method`` is ``'avb'``. then the hyper-
            hypernetwork might be required to estimate uncertainty on old
            coreset samples.
        method (str): Possible choices: ``'bbb'`` and ``'avb'``. Determines
            how uncertainty is computed, if needed.
    """
    assert method in ['bbb', 'avb']
    assert hhnet is None or method != 'bbb'

    if config.coreset_size == -1:
        return

    if config.per_task_coreset or not hasattr(shared, 'coreset'):
        num_new_samples = config.coreset_size
    else:
        # How many samples to be replaced.
        num_replace = config.coreset_size // (task_id+1)
        num_new_samples = num_replace

    # Pick random samples from the training set as new coreset.
    batch = data.next_train_batch(num_new_samples, return_ids=True)
    new_inputs = data.input_to_torch_tensor(batch[0], device,
                                            mode='train')
    new_targets = data.output_to_torch_tensor(batch[1], device,
                                              mode='train')
    #_, new_labels = torch.max(new_targets, 1)
    #new_labels = new_labels.detach().cpu().numpy()

    if config.per_task_coreset or not hasattr(shared, 'coreset'):

        # Add samples to existing coreset.
        if hasattr(shared, 'coreset'):
            assert np.all(np.equal(list(shared.coreset.shape[1:]),
                                   list(new_inputs.shape[1:])))
            shared.coreset = torch.cat([shared.coreset, new_inputs], dim=0)
            shared.coreset_targets = torch.cat([shared.coreset_targets,
                                                new_targets], dim=0)
            #shared.coreset_labels = np.concatenate([shared.coreset_labels,
            #                                        new_labels])
            shared.task_ident = np.concatenate([shared.task_ident,
                np.ones(num_new_samples) * task_id])
            shared.sample_ids = np.concatenate([shared.sample_ids, batch[2]])
        else:
            shared.coreset = new_inputs
            shared.coreset_targets = new_targets
            #shared.coreset_labels = new_labels
            shared.task_ident = np.ones(num_new_samples) * task_id
            shared.sample_ids = batch[2]

        logger.debug('%d training samples from task %d have been added to ' \
                     % (num_new_samples, task_id+1) + 'the coreset.')
    else:
        assert hasattr(shared, 'coreset')

        logger.debug('%d/%d samples in the coreset will be replaced by ' \
                     % (num_replace, config.coreset_size) +
                     'samples from task %d.' % (task_id+1))

        if 'regression' in shared.experiment_type:
            raise NotImplementedError()

        if method == 'bbb':
            ents = calc_batch_uncertainty(config, shared, task_id,
                shared.coreset, mnet, hnet, data, config.val_sample_size,
                mnet_weights=None, allowed_outputs=allowed_outputs,
                disable_lrt=config.disable_lrt_test)
        else:
            ents = pcutils.calc_batch_uncertainty(config, shared, task_id,
                device, shared.coreset, mnet, hnet, hhnet, data,
                config.val_sample_size, hnet_theta=None,
                allowed_outputs=allowed_outputs)

        # We replace those samples in the coreset that achieve high entropy
        # under the current model.
        replace_inds = np.argsort(ents)[-num_replace:]

        assert np.all(np.equal(list(shared.coreset.shape[1:]),
                               list(new_inputs.shape[1:])))
        shared.coreset[replace_inds, :] = new_inputs
        shared.coreset_targets[replace_inds, :] = new_targets
        #shared.coreset_labels[replace_inds] = new_labels
        shared.task_ident[replace_inds] = np.ones(num_replace) * task_id
        shared.sample_ids[replace_inds] = batch[2]

def parse_performance_criterion(config, shared, logger):
    """Parse the CLI argument that determines the performance criterion to be
    met for a run to continue.

    Function parses argument ``--during_acc_criterion``.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Coresets will be stored and maintained
            here.
        logger: Console (and file) logger.

    Returns:
        (list): List of minimal during accuracies to attain for the first
            ``config.num_tasks-1``.
    """
    n = config.num_tasks

    during_acc_criterion = [-1] * (n-1)
    if hasattr(config, 'during_acc_criterion') and \
            config.during_acc_criterion != '-1':
        min_daccs = misc.str_to_floats(config.during_acc_criterion)
        if len(min_daccs) == 1:
            during_acc_criterion = [min_daccs[0]] * (n-1)
        elif len(min_daccs) < n-1:
            logger.warn('Too less values in argument "during_acc_criterion". ' +
                        'Will be filled up with "-1" values.')
            during_acc_criterion[:len(min_daccs)] = min_daccs
        elif len(min_daccs) > n-1:
            logger.warn('Too many values in argument "during_acc_criterion". ' +
                        'Only the first %d values are considered.' % (n-1))
            during_acc_criterion = min_daccs[:n]
        else:
            during_acc_criterion = min_daccs

    return during_acc_criterion

def fit_targets_to_softmax(config, shared, device, data, task_id, targets):
    """Fit dataset target vectors to the actual softmax output layer.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Coresets will be stored and maintained
            here.
        device: Pytorch device.
        data (data.Dataset): The dataset of the current task.
        task_id (int): Task ID.
        targets (torch.Tensor or numpy.nd_array): Dataset targets that should be
            adapted.
        allowed_outputs (list): List of integers, containing the indices of the
            network outputs belonging to the current task.

    Returns:
        (torch.Tensor or numpy.ndarray)
    """
    T = targets
    num_samples = T.shape[0]

    if hasattr(config, 'non_growing_sf_cl3') and config.cl_scenario == 3 \
            and config.non_growing_sf_cl3:
        T_tmp = T
        if isinstance(T, torch.Tensor):
            T = torch.zeros((num_samples,
                config.num_tasks * data.num_classes)).to(device)
        else:
            T = np.zeros((num_samples,
                config.num_tasks * data.num_classes))
        sind = task_id * data.num_classes
        T[:,sind:sind+data.num_classes] = T_tmp
    elif config.cl_scenario == 3 and not config.split_head_cl3 and \
            ('ewc' in shared.experiment_type or config.mnet_only):
        # See function `out_units_of_task` for an explanation of this case.
        T_tmp = T
        if isinstance(T, torch.Tensor):
            T = torch.zeros((num_samples,
                shared.num_trained * data.num_classes)).to(device)
        else:
            T = np.zeros((num_samples,
                shared.num_trained * data.num_classes))
        sind = task_id * data.num_classes
        T[:,sind:sind+data.num_classes] = T_tmp
    elif config.cl_scenario == 3 and not config.split_head_cl3 and \
            task_id > 0 and T is not None:
        # We preprend zeros to the 1-hot vector according to the number of
        # output units belonging to previous tasks.
        if isinstance(T, torch.Tensor):
            T = torch.cat((torch.zeros((num_samples,
                task_id * data.num_classes)).to(device), T), dim=1)
        else:
            T = np.concatenate((np.zeros((num_samples,
                task_id * data.num_classes)), T), axis=1)

    return T

if __name__ == '__main__':
    pass


