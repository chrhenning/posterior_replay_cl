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
# @title           :utils/hnet_regularizer.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :06/05/2019
# @version         :1.0
# @python_version  :3.6.8
"""
Hypernetwork Regularization
---------------------------

We summarize our own regularizers in this module. These regularizer ensure that
the output of a hypernetwork don't change.
"""

import torch
import numpy as np
from warnings import warn

from hnets import HyperNetInterface
from utils.module_wrappers import CLHyperNetInterface

def get_current_targets(task_id, hnet):
    r"""For all :math:`j < \text{task\_id}`, compute the output of the
    hypernetwork. This output will be detached from the graph before being added
    to the return list of this function.

    Note, if these targets don't change during training, it would be more memory
    efficient to store the weights :math:`\theta^*` of the hypernetwork (which
    is a fixed amount of memory compared to the variable number of tasks).
    Though, it is more computationally expensive to recompute
    :math:`h(c_j, \theta^*)` for all :math:`j < \text{task\_id}` everytime the
    target is needed.

    Note, this function sets the hypernet temporarily in eval mode. No gradients
    are computed.

    See argument ``targets`` of :func:`calc_fix_target_reg` for a use-case of
    this function.

    Args:
        task_id (int): The ID of the current task.
        hnet: An instance of the hypernetwork before learning a new task
            (i.e., the hypernetwork has the weights :math:`\theta^*` necessary
            to compute the targets).

    Returns:
        An empty list, if ``task_id`` is ``0``. Otherwise, a list of
        ``task_id-1`` targets. These targets can be passed to the function
        :func:`calc_fix_target_reg` while training on the new task.
    """
    # We temporarily switch to eval mode for target computation (e.g., to get
    # rid of training stochasticities such as dropout).
    hnet_mode = hnet.training
    hnet.eval()

    if not isinstance(hnet, HyperNetInterface):
        assert isinstance(hnet, CLHyperNetInterface)
        warn('Use of deprecated hypernetwork interface. This function will ' +
             'discontinue the support of those hypernetworks in the future.',
             DeprecationWarning)

    ret = []

    with torch.no_grad():
        if not isinstance(hnet, HyperNetInterface):
            for j in range(task_id):
                W = hnet.forward(task_id=j)
                # NOTE I am pretty sure the clone here is not necessary, which
                # is why I take it out for the new hnet interface.
                ret.append([d.detach().clone() for d in W])
        elif task_id > 0:
            W = hnet.forward(cond_id=list(range(task_id)),
                             ret_format='sequential')
            ret = [[p.detach() for p in W_tid] for W_tid in W]

    hnet.train(mode=hnet_mode)

    return ret

def calc_jac_reguarizer(hnet, task_id, dTheta, device):
    r"""Compute the CL regularizer, which is a sum over all previous task
    ID's, that enforces that the norm of the matrix product of hypernet
    Jacobian and ``dTheta`` is small.

    I.e., for all :math:`j < \text{task\_id}` minimize the following norm:

    .. math::
        \lVert J_h(c_j, \theta) \Delta \theta \rVert^2

    where :math:`\theta` (and :math:`\Delta\theta`) is assumed to be vectorized.

    This regularizer origins in a first-order Taylor approximation of the
    regularizer:

    .. math::
        \lVert h(c_j, \theta) - h(c_j, \theta + \Delta\theta) \rVert^2

    .. deprecated:: 1.0
        Function is deprecated and currently only compatible with hypernetworks
        implementing the deprecated interface
        :class:`utils.module_wrappers.CLHyperNetInterface`.

    Args:
        (....): See docstring of method :func:`calc_fix_target_reg`.
        device: Current PyTorch device.

    Returns:
        The value of the regularizer.
    """
    assert isinstance(hnet, CLHyperNetInterface)
    assert task_id > 0
    assert hnet.has_theta # We need parameters to be regularized.

    # TODO Adapt to new hnet interface to get rid of deprecation.
    warn('Function is deprecated.', DeprecationWarning)

    reg = 0

    for i in range(task_id): # For all previous tasks.
        W = torch.cat([w.view(-1) for w in hnet.forward(task_id=i)])

        # Problem, autograd.grad() accumulates all gradients with
        # respect to each w in "W". Hence, we don't get a Jacobian but a
        # tensor of the same size as "t", where the partials of all W's
        # with respect to this "t" are summed.
        #J = torch.autograd.grad(W, t,
        #    grad_outputs=torch.ones(W.size()).to(device),
        #    retain_graph=True, create_graph=True, only_inputs=True)[0]
        #
        # Hence, to get the actual jacobian, we have to iterate over all
        # individual outputs of the hypernet.
        for wind, w in enumerate(W):
            tmp = 0
            # FIXME `torch.autograd.grad` can also take a list of tensors as
            # input (and returns a corresponding tuple). Thus, this second for
            # loop is unneccessary. It should be much more efficient to get rid
            # of it (`grad` may also reuse gradients in this case).
            for tind, t in enumerate(hnet.theta):
                partial = torch.autograd.grad(w, t, grad_outputs=None,
                    retain_graph=True, create_graph=True,
                    only_inputs=True)[0]

                # Intuitively, "partial" represents part of a row in the
                # Jacobian (if dTheta would have been linearized to a
                # vector). To compute the matrix vector product (Jacobian
                # times dTheta), we have to sum over the element-wise
                # product of rows from the Jacobian with dTheta.
                tmp += torch.mul(partial, dTheta[tind]).sum()

            # Since we are interested in computing the squared L2 norm of
            # the matrix vector product: Jacobian times dTheta,
            # we have to simply sum the sqaured dot products between all
            # rows of the Jacobian with dTheta.
            reg += torch.pow(tmp, 2)

    # Normalize by the number of tasks.
    return reg / task_id

def calc_value_preserving_reg(hnet, task_id, dTheta):
    r"""This regularizer simply restricts a change in output-mapping for
    previous task embeddings. I.e., for all j < task_id minimize:

    .. math::
        \lVert h(c_j, \theta) - h(c_j, \theta + \Delta\theta) \rVert^2

    .. deprecated:: 1.0
        Function is deprecated and currently only compatible with hypernetworks
        implementing the deprecated interface
        :class:`utils.module_wrappers.CLHyperNetInterface`.

    Args:
        (....): See docstring of method :func:`calc_fix_target_reg`.

    Returns:
        The value of the regularizer.
    """
    assert isinstance(hnet, CLHyperNetInterface)
    assert task_id > 0
    assert hnet.has_theta # We need parameters to be regularized.

    # TODO Adapt to new hnet interface to get rid of deprecation.
    warn('Function is deprecated.', DeprecationWarning)

    reg = 0

    for i in range(task_id): # For all previous tasks.
        W_prev = torch.cat([w.view(-1) for w in hnet.forward(task_id=i)])
        W_new = torch.cat([w.view(-1) for w in hnet.forward(task_id=i,
                                                            dTheta=dTheta)])

        reg += (W_prev - W_new).pow(2).sum()

    return reg / task_id

def calc_fix_target_reg(hnet, task_id, targets=None, dTheta=None, dTembs=None,
                        mnet=None, inds_of_out_heads=None,
                        fisher_estimates=None, prev_theta=None,
                        prev_task_embs=None, batch_size=None, reg_scaling=None):
    r"""This regularizer simply restricts the output-mapping for previous
    task embeddings. I.e., for all :math:`j < \text{task\_id}` minimize:

    .. math::
        \lVert \text{target}_j - h(c_j, \theta + \Delta\theta) \rVert^2

    where :math:`c_j` is the current task embedding for task :math:`j` (and we
    assumed that ``dTheta`` was passed).

    Args:
        hnet: The hypernetwork whose output should be regularized; has to
            implement the interface
            :class:`hnets.hnet_interface.HyperNetInterface`.
        task_id (int): The ID of the current task (the one that is used to
            compute ``dTheta``).
        targets (list): A list of outputs of the hypernetwork. Each list entry
            must have the output shape as returned by the
            :meth:`hnets.hnet_interface.HyperNetInterface.forward` method of the
            ``hnet``. Note, this function doesn't detach targets. If desired,
            that should be done before calling this function.

            Also see :func:`get_current_targets`.
        dTheta (list, optional): The current direction of weight change for the
            internal (unconditional) weights of the hypernetwork evaluated on
            the task-specific loss, i.e., the weight change that would be
            applied to the unconditional parameters :math:`\theta`. This
            regularizer aims to modify this direction, such that the hypernet
            output for embeddings of previous tasks remains unaffected.
            Note, this function does not detach ``dTheta``. It is up to the
            user to decide whether dTheta should be a constant vector or
            might depend on parameters of the hypernet.

            Also see :func:`utils.optim_step.calc_delta_theta`.
        dTembs (list, optional): The current direction of weight change for the
            task embeddings of all tasks that have been learned already.
            See ``dTheta`` for details.
        mnet: Instance of the main network. Has to be provided if
            ``inds_of_out_heads`` are specified.
        inds_of_out_heads: (list, optional): List of lists of integers, denoting
            which output neurons of the main network are used for predictions of
            the corresponding previous tasks.
            This will ensure that only weights of output neurons involved in
            solving a task are regularized.

            If provided, the method
            :meth:`mnets.mnet_interface.MainNetInterface.get_output_weight_mask
            of the main network ``mnet`` is used to determine which hypernetwork
            outputs require regularization.
        fisher_estimates (list, optional): A list of list of tensors, containing
            estimates of the Fisher Information matrix for each weight
            tensor in the main network and each task.
            Note, that :code:`len(fisher_estimates) == task_id`.
            The Fisher estimates are used as importance weights for single
            weights when computing the regularizer.
        prev_theta (list, optional): If given, ``prev_task_embs`` but not
            ``targets`` has to be specified. ``prev_theta`` is expected to be
            the internal unconditional weights :math:`theta` prior to learning
            the current task. Hence, it can be used to compute the targets on
            the fly (which is more memory efficient (constant memory), but more
            computationally demanding).
            The computed targets will be detached from the computational graph.
            Independent of the current hypernet mode, the targets are computed
            in ``eval`` mode.
        prev_task_embs (list, optional): If given, ``prev_theta`` but not
            ``targets`` has to be specified. ``prev_task_embs`` are the task
            embeddings (conditional parameters) of the hypernetwork.
            See docstring of ``prev_theta`` for more details.
        batch_size (int, optional): If specified, only a random subset of
            previous tasks is regularized. If the given number is bigger than
            the number of previous tasks, all previous tasks are regularized.

            Note:
                A ``batch_size`` smaller or equal to zero will be ignored
                rather than throwing an error.
        reg_scaling (list, optional): If specified, the regulariation terms for
            the different tasks are scaled arcording to the entries of this
            list.

    Returns:
        The value of the regularizer.
    """
    # FIXME The current function interface is in its terminology very close to
    # continual learning of separate tasks. The naming of variables could be
    # brought closer to the new hypernet interface in the future.
    if isinstance(hnet, CLHyperNetInterface):
        warn('Use of deprecated hypernetwork interface. This function will ' +
             'discontinue the support of those hypernetworks in the future.',
             DeprecationWarning)
        return _deprecated_calc_fix_target_reg(hnet, task_id, targets, dTheta,
            dTembs, mnet, inds_of_out_heads, fisher_estimates, prev_theta,
            prev_task_embs, batch_size, reg_scaling)

    assert isinstance(hnet, HyperNetInterface)
    assert task_id > 0
    # FIXME We currently assume the hypernet has all parameters internally.
    # Alternatively, we could allow the parameters to be passed to us, that we
    # will then pass to the forward method.
    assert hnet.unconditional_params is not None and \
        len(hnet.unconditional_params) > 0
    assert targets is None or len(targets) == task_id
    assert inds_of_out_heads is None or mnet is not None
    assert inds_of_out_heads is None or len(inds_of_out_heads) >= task_id
    assert targets is None or (prev_theta is None and prev_task_embs is None)
    assert prev_theta is None or prev_task_embs is not None
    #assert prev_task_embs is None or len(prev_task_embs) >= task_id
    assert dTembs is None or len(dTembs) >= task_id
    assert reg_scaling is None or len(reg_scaling) >= task_id

    # Number of tasks to be regularized.
    num_regs = task_id
    ids_to_reg = list(range(num_regs))
    if batch_size is not None and batch_size > 0:
        if num_regs > batch_size:
            ids_to_reg = np.random.choice(num_regs, size=batch_size,
                                          replace=False).tolist()
            num_regs = batch_size

    # FIXME Assuming all unconditional parameters are internal.
    assert len(hnet.unconditional_params) == \
        len(hnet.unconditional_param_shapes)

    weights = dict()
    uncond_params = hnet.unconditional_params
    if dTheta is not None:
        uncond_params = hnet.add_to_uncond_params(dTheta, params=uncond_params)
    weights['uncond_weights'] = uncond_params

    if dTembs is not None:
        # FIXME That's a very unintutive solution for the user. The problem is,
        # that the whole function terminology is based on the old hypernet
        # interface. The new hypernet interface doesn't have the concept of
        # task embedding.
        # The problem is, the hypernet might not just have conditional input
        # embeddings, but also other conditional weights.
        # If it would just be conditional input embeddings, we could just add
        # `dTembs[i]` to the corresponding embedding and use the hypernet
        # forward argument `cond_input`, rather than passing conditional
        # parameters.
        # Here, we now assume all conditional parameters have been passed, which
        # is unrealistic. We leave the problem open for a future implementation
        # of this function.
        assert hnet.conditional_params is not None and \
            len(hnet.conditional_params) == len(hnet.conditional_param_shapes) \
            and len(hnet.conditional_params) == len(dTembs)
        weights['cond_weights'] = hnet.add_to_uncond_params(dTembs,
            params=hnet.conditional_params)

    if targets is None:
        prev_weights = dict()
        prev_weights['uncond_weights'] = prev_theta
        # FIXME We just assume that `prev_task_embs` are all conditional
        # weights.
        prev_weights['cond_weights'] = prev_task_embs

    reg = 0

    for i in ids_to_reg:
        weights_predicted = hnet.forward(cond_id=i, weights=weights)

        if targets is not None:
            target = targets[i]
        else:
            # Compute targets in eval mode!
            hnet_mode = hnet.training
            hnet.eval()

            # Compute target on the fly using previous hnet.
            with torch.no_grad():
                target = hnet.forward(cond_id=i, weights=prev_weights)
            target = [d.detach().clone() for d in target]

            hnet.train(mode=hnet_mode)

        if inds_of_out_heads is not None:
            # Regularize all weights of the main network except for the weights
            # belonging to output heads of the target network other than the
            # current one (defined by task id).
            W_target = flatten_and_remove_out_heads(mnet, target,
                                                    inds_of_out_heads[i])
            W_predicted = flatten_and_remove_out_heads(mnet, weights_predicted,
                                                       inds_of_out_heads[i])
        else:
            # Regularize all weights of the main network.
            W_target = torch.cat([w.view(-1) for w in target])
            W_predicted = torch.cat([w.view(-1) for w in weights_predicted])

        if fisher_estimates is not None:
            _assert_shape_equality(weights_predicted, fisher_estimates[i])

            if inds_of_out_heads is not None:
                FI = flatten_and_remove_out_heads(mnet, fisher_estimates[i],
                                                  inds_of_out_heads[i])
            else:
                FI = torch.cat([w.view(-1) for w in fisher_estimates[i]])

            reg_i = (FI * (W_target - W_predicted).pow(2)).sum()
        else:
            reg_i = (W_target - W_predicted).pow(2).sum()

        if reg_scaling is not None:
            reg += reg_scaling[i] * reg_i
        else:
            reg += reg_i

    return reg / num_regs

def _assert_shape_equality(list1, list2):
    """Ensure that 2 lists of tensors have the same shape."""
    assert(len(list1) == len(list2))
    for i in range(len(list1)):
        assert(np.all(np.equal(list(list1[i].shape), list(list2[i].shape))))

def flatten_and_remove_out_heads(mnet, weights, allowed_outputs):
    """Flatten a list of target network tensors to a single vector, such that
    output neurons that belong to other than the current output head are
    dropped.

    Note, this method assumes that the main network has a fully-connected output
    layer.

    Args:
        mnet: Main network instance.
        weights: A list of weight tensors of the main network (must adhere the
            corresponding weight shapes).
        allowed_outputs: List of integers, denoting which output neurons of
            the fully-connected output layer belong to the current head.

    Returns:
        The flattened weights with those output weights not belonging to the
        current head being removed.
    """
    if hasattr(mnet, 'get_output_weight_mask'):
        out_masks = mnet.get_output_weight_mask(out_inds=allowed_outputs)

        if mnet.hyper_shapes_learned_ref is None and \
                len(weights) != len(mnet.param_shapes):
            raise NotImplementedError('Proper masking cannot be performed if ' +
                'if attribute "hyper_shapes_learned_ref" is not implemented.')

        new_weights = []
        for i, w in enumerate(weights):
            if len(weights) == len(mnet.param_shapes):
                w_ind = i
            else:
                assert len(weights) == len(mnet.hyper_shapes_learned_ref)
                w_ind = mnet.hyper_shapes_learned_ref[i]

            if out_masks[w_ind] is None:
                new_weights.append(w.flatten())
            else:
                new_weights.append(w[out_masks[w_ind]].flatten())

        return torch.cat(new_weights)
    else:
        warn('Legacy support for deprecated code. Please upgrade the used ' +
             'network interface.', DeprecationWarning)
        # FIXME the method `get_output_weight_mask` and attribute `mask_fc_out`
        # did not exist in a previous version of the main network interface,
        # which is why we need to ensure downwards compatibility.
        # Previously, it was assumed sufficient for masking if `has_fc_out` was
        # set to True.
        assert(mnet.has_fc_out)
        assert(not hasattr(mnet, 'mask_fc_out') or \
               (mnet.has_fc_out and mnet.mask_fc_out))

        obias_ind = len(weights)-1 if mnet.has_bias else -1
        oweights_ind = len(weights)-2 if mnet.has_bias else len(weights)-1

        ret = []
        for i, w in enumerate(weights):
            if i == obias_ind: # Output bias
                ret.append(w[allowed_outputs])
            elif i == oweights_ind: # Output weights
                ret.append(w[allowed_outputs, :].view(-1))
            else:
                ret.append(w.view(-1))

        return torch.cat(ret)

def _deprecated_calc_fix_target_reg(hnet, task_id, targets, dTheta, dTembs,
                                    mnet, inds_of_out_heads, fisher_estimates,
                                    prev_theta, prev_task_embs, batch_size,
                                    reg_scaling):
    """Compute hypernetwork regularizer using hypernetwork with deprecated
    interface :class:`utils.module_wrappers.CLHyperNetInterface`."""
    assert(task_id > 0)
    assert(hnet.has_theta) # We need parameters to be regularized.
    assert(targets is None or len(targets) == task_id)
    assert(inds_of_out_heads is None or mnet is not None)
    assert(inds_of_out_heads is None or len(inds_of_out_heads) >= task_id)
    assert(targets is None or (prev_theta is None and prev_task_embs is None))
    assert(prev_theta is None or prev_task_embs is not None)
    assert(prev_task_embs is None or len(prev_task_embs) >= task_id)
    assert(dTembs is None or len(dTembs) >= task_id)
    assert(reg_scaling is None or len(reg_scaling) >= task_id)

    # Number of tasks to be regularized.
    num_regs = task_id
    ids_to_reg = list(range(num_regs))
    if batch_size is not None and batch_size > 0:
        if num_regs > batch_size:
            ids_to_reg = np.random.choice(num_regs, size=batch_size,
                                          replace=False).tolist()
            num_regs = batch_size

    reg = 0

    for i in ids_to_reg:
        if dTembs is None:
            weights_predicted = hnet.forward(task_id=i, dTheta=dTheta)
        else:
            temb = hnet.get_task_emb(i) + dTembs[i]
            weights_predicted = hnet.forward(dTheta=dTheta, task_emb=temb)

        if targets is not None:
            target = targets[i]
        else:
            # Compute targets in eval mode!
            hnet_mode = hnet.training
            hnet.eval()

            # Compute target on the fly using previous hnet.
            with torch.no_grad():
                target = hnet.forward(theta=prev_theta,
                                      task_emb=prev_task_embs[i])
            target = [d.detach().clone() for d in target]

            hnet.train(mode=hnet_mode)

        if inds_of_out_heads is not None:
            # Regularize all weights of the main network except for the weights
            # belonging to output heads of the target network other than the
            # current one (defined by task id).
            W_target = flatten_and_remove_out_heads(mnet, target,
                                                    inds_of_out_heads[i])
            W_predicted = flatten_and_remove_out_heads(mnet, weights_predicted,
                                                       inds_of_out_heads[i])
        else:
            # Regularize all weights of the main network.
            W_target = torch.cat([w.view(-1) for w in target])
            W_predicted = torch.cat([w.view(-1) for w in weights_predicted])

        if fisher_estimates is not None:
            _assert_shape_equality(weights_predicted, fisher_estimates[i])
            if inds_of_out_heads is not None:
                FI = flatten_and_remove_out_heads(mnet, fisher_estimates[i],
                                                  inds_of_out_heads[i])
            else:
                FI = torch.cat([w.view(-1) for w in fisher_estimates[i]])

            reg_i = (FI * (W_target - W_predicted).pow(2)).sum()
        else:
            reg_i = (W_target - W_predicted).pow(2).sum()

        if reg_scaling is not None:
            reg += reg_scaling[i] * reg_i
        else:
            reg += reg_i

    return reg / num_regs

if __name__ == '__main__':
    pass


