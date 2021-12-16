#!/usr/bin/env python3
# Copyright 2020 Christian Henning
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
# @title          :probabilistic/prob_cifar/train_args.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/30/2020
# @version        :1.0
# @python_version :3.6.9
"""
Helper functions for command-line argument definition and parsing
-----------------------------------------------------------------
"""

def miscellaneous_args(agroup, show_no_hhnet=False):
    """This is a helper function of the function
    :func:`probabilistic.prob_mnist.train_args.parse_cmd_arguments` to add
    arguments to the miscellaneous argument group.

    Args:
        agroup: The argument group returned by method
            :func:`utils.cli_args.miscellaneous_args`.
        no_hhnet (bool): Whether the option `no_hhnet` should be provided.
    """
    if show_no_hhnet:
        agroup.add_argument('--no_hhnet', action='store_true',
                            help='No hyper-hypernet will be used.')
    agroup.add_argument('--no_dis', action='store_true',
                        help='No discriminator will be used for the prior-' +
                             'matching.')

def imp_args(parser, dlatent_dim=8, show_prior_focused=False):
    """This is a helper function of the function
    :func:`probabilistic.prob_mnist.train_args.parse_cmd_arguments` to add
    an argument group for options specific to training an implicit distribution.

    Arguments specified in this function:
        - `latent_dim`
        - `latent_std`
        - `prior_focused`
        - `full_support_perturbation`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        dlatent_dim: Default value of option `latent_dim`.
        show_prior_focused (bool): Whether option `prior-focused` should be
            shown.
    """
    vigroup = parser.add_argument_group('Implicit Distribution options')
    vigroup.add_argument('--latent_dim', type=int, metavar='N',
                        default=dlatent_dim,
                        help='Dimensionality of the latent vector (noise ' +
                             'input to the hypernet). Default: %(default)s.')
    vigroup.add_argument('--latent_std', type=float, default=1.0,
                        help='Standard deviation of the latent space. ' +
                             'Default: %(default)s.')
    if show_prior_focused:
        vigroup.add_argument('--prior_focused', action='store_true',
                            help='If enabled, CL will be performed in a ' +
                                 'prior-manner, i.e., the posterior of the ' +
                                 'most recent task becomes the prior of the ' +
                                 'current task. Note, in this case a single ' +
                                 'implicit distribution yielding weight ' +
                                 'realizations for all tasks simultaneously ' +
                                 'is trained, such that no hyper-' +
                                 'hypernetwork is needed.')
    vigroup.add_argument('--full_support_perturbation', type=float, default=-1.,
                        help='If unequal "-1", then zero-mean noise will be ' +
                             'added to the hypernetwork output with a std ' +
                             'that results from the product of the value ' +
                             'specified here and the value specified via ' +
                             'option "latent_std". Default: %(default)s.')

def avb_args(parser):
    """This is a helper function of the function
    :func:`probabilistic.prob_mnist.train_args.parse_cmd_arguments` to add
    an argument group for options specific to AVB training.

    Arguments specified in this function:
        - `dis_lr`
        - `dis_batch_size`
        - `num_dis_steps`
        - `no_dis_reinit`
        - `use_batchstats`
        - `no_adaptive_contrast`
        - `num_ac_samples`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
    """
    ### AVB options.
    agroup = parser.add_argument_group('AVB options')
    agroup.add_argument('--dis_lr', type=float, default=-1.,
                        help='If not "-1.", then this option will be used as ' +
                             'learning rate for the discriminator ' +
                             '(otherwise, the learning rate specified by ' +
                             '"lr" is used). Default: %(default)s.')
    agroup.add_argument('--dis_batch_size', type=int, metavar='N', default=1,
                        help='Batch size when training discriminator. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--num_dis_steps', type=int, metavar='N', default=1,
                        help='Number of discriminator updates before ' +
                             'updating the hypernetwork. Default: %(default)s.')
    agroup.add_argument('--no_dis_reinit', action='store_true',
                        help='The discriminator is randomly re-initialized ' +
                             'after every task. If this option is activated, ' +
                             'the discriminator will be continously trained.')
    agroup.add_argument('--use_batchstats', action='store_true',
                        help='Input additional batch statistics to the ' +
                             'discriminator as a cheap trick to mitigate ' +
                             'mode collapse.')
    agroup.add_argument('--no_adaptive_contrast', action='store_true',
                        help='The AVB paper proposed a trick to stabilize ' +
                             'prior-matching, called adaptive contrast. ' +
                             'This trick is used whenever applicable (i.e., ' +
                             'prior density can be computed analytically). ' +
                             'This option will deactivate the use of this ' +
                             'trick.')
    agroup.add_argument('--num_ac_samples', type=int,  metavar='N', default=100,
                        help='If the adaptive contrast trick is applied, ' +
                             'then this option determines the number of ' +
                             'weight samples used to determine mean and ' +
                             'variance of the implicit distribution.' +
                             'Default: %(default)s.')

def ssge_args(parser, dssge_sample_size=10, drbf_kernel_width=1.,
              dthr_ssge_eigenvals=1.):
    """This is a helper function of the function
    :func:`probabilistic.prob_mnist.train_args.parse_cmd_arguments` to add
    an argument group for options specific to SSGE training.

    Arguments specified in this function:
        - `heuristic_kernel`
        - `rbf_kernel_width`
        - `num_ssge_eigenvals`
        - `thr_ssge_eigenvals`
        - `ssge_sample_size`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.`.
        dssge_sample_size: Default value of option `ssge_sample_size`.
        drbf_kernel_width: Default value of option `rbf_kernel_width`.
        dthr_ssge_eigenvals: Default value of option `thr_ssge_eigenvals`.
    """
    ### SSGE options.
    sgroup = parser.add_argument_group('SSGE options')
    sgroup.add_argument('--heuristic_kernel', action='store_true',
                        help='If "True" the width of the RBF kernel will ' +
                             'be computed as the median of the pairwise ' +
                             'distance between all samples used for the ' +
                             'gradient estimation. Else, the value ' +
                             '"rbf_kernel_width" will be used.')
    sgroup.add_argument('--rbf_kernel_width', type=float,
                        default=drbf_kernel_width,
                        help='The width of the RBF kernels used during SSGE ' +
                             'gradient estimation. Ignored if the option ' +
                             '"heuristic_kernel" is active. Default: ' +
                             '%(default)s.')
    sgroup.add_argument('--num_ssge_eigenvals', type=int, default=-1,
                        help='The number of eigenvalues to be used when ' +
                             'estimating gradients with SSGE. If "-1" all ' +
                             'eigenvalues will be used. Default: %(default)s.')
    sgroup.add_argument('--thr_ssge_eigenvals', type=float,
                        default=dthr_ssge_eigenvals,
                        help='The threshold of the eigenvalue ratio to '+
                             'to be used for determining the number of ' +
                             'eigenvalues for estimating gradients with SSGE. '+
                             'For a value of "1", all eigenvalues are used. ' +
                             'Default: %(default)s.')
    sgroup.add_argument('--ssge_sample_size', type=int,
                        default=dssge_sample_size,
                        help='How many samples should be used for the ' +
                             'approximation of the score function when using ' +
                             'SSGE. Default: %(default)s.')

def extra_cl_args(agroup):
    """This is a helper function of the function
    :func:`probabilistic.prob_mnist.train_args.parse_cmd_arguments` to add
    arguments to the CL argument group.

    Args:
        agroup: The argument group returned by method
            :func:`utils.cli_args.cl_args`.
    """
    agroup.add_argument('--skip_tasks', type=int, default=0,
                        help='The number of tasks to be skipped. E.g., if ' +
                             '"num_classes_per_task=10" and 1 task should be ' +
                             'skipped, then CIFAR-10 will be skipped. Note, ' +
                             'this option will no affect "num_tasks". ' +
                             'Default: %(default)s.')

if __name__ == '__main__':
    pass


