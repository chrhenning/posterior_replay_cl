Task Inference in Continual Learning via Predictive Uncertainty
===============================================================

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

In this subpackage, we explore ideas regarding continual learning via Bayesian Neural Networks (BNNs). In particular, we use hypernetworks to obtain task-specific approximate parameter posterior distributions.

General Remarks
---------------

Elastic Weight Consolidation
----------------------------

In regression tasks, EWC can be applied either to single-head or multi-head networks. For multi-head networks, the task identity needs to be explicitly provided during inference. In order to be able to infer the task identity automatically, we consider the predictive uncertainty from the posterior constructed from the Fisher matrix over all shared parameters and the task-specific Fisher matrix from the output head. Therewith, we get an uncertainty estimate per output head which we can utilize for task inference.

For classification, we provide 5 operation modes:

  - ``--cl_scenario=1``: Multi-head network where task identity will be provided during inference
  - ``--cl_scenario=2``: Single-head network, thus task identity does not need to be provided. Hence, this scenario requires all tasks to have the same number of classes and that input domains are disjoint.
  - ``--cl_scenario=3``: Joint, growing softmax over all tasks learned so far. Disadvantage: softmax requires normalization, that is dependent on the number of classes. Hence, if posterior is formed for the current task, it considers a model that requires a different normalization than the models with larger softmax that are considered on future tasks. Therefore, EWC's performance is expected to be poor here.
  - ``--cl_scenario=3 --non_growing_sf_cl3``: Joint softmax over all tasks is learned. The softmax has its final size from the beginning. Thus, there is no model misspecification and EWC should work better than in the previous case. However, this means that the number of tasks has to be known a priori, which is a violation of CL desiderata.
  - ``--cl_scenario=3 --split_head_cl3``: A multi-head network is trained and an individual posterior per task is formed (similar to the regression case) using the Fisher matrices acquired during training. Thos posteriors are used during inference to infer the task identity (and therewith the output head to be chosen).
