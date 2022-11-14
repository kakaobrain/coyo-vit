# https://github.com/tensorflow/models/blob/master/official/common/distribute_utils.py

# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for running models in a distributed setting."""

import json
import os
import tensorflow as tf


def _collective_communication(all_reduce_alg):
    """Return a CollectiveCommunication based on all_reduce_alg.

    Args:
      all_reduce_alg: a string specifying which collective communication to pick,
        or None.

    Returns:
      tf.distribute.experimental.CollectiveCommunication object

    Raises:
      ValueError: if `all_reduce_alg` not in [None, "ring", "nccl"]
    """
    collective_communication_options = {
        None: tf.distribute.experimental.CollectiveCommunication.AUTO,
        "ring": tf.distribute.experimental.CollectiveCommunication.RING,
        "nccl": tf.distribute.experimental.CollectiveCommunication.NCCL
    }
    if all_reduce_alg not in collective_communication_options:
        raise ValueError(
            "When used with `multi_worker_mirrored`, valid values for "
            "all_reduce_alg are [`ring`, `nccl`].  Supplied value: {}".format(
                all_reduce_alg))
    return collective_communication_options[all_reduce_alg]


def _mirrored_cross_device_ops(all_reduce_alg, num_packs):
    """Return a CrossDeviceOps based on all_reduce_alg and num_packs.

    Args:
      all_reduce_alg: a string specifying which cross device op to pick, or None.
      num_packs: an integer specifying number of packs for the cross device op.

    Returns:
      tf.distribute.CrossDeviceOps object or None.

    Raises:
      ValueError: if `all_reduce_alg` not in [None, "nccl", "hierarchical_copy"].
    """
    if all_reduce_alg is None:
        return None
    mirrored_all_reduce_options = {
        "nccl": tf.distribute.NcclAllReduce,
        "hierarchical_copy": tf.distribute.HierarchicalCopyAllReduce
    }
    if all_reduce_alg not in mirrored_all_reduce_options:
        raise ValueError(
            "When used with `mirrored`, valid values for all_reduce_alg are "
            "[`nccl`, `hierarchical_copy`].  Supplied value: {}".format(
                all_reduce_alg))
    cross_device_ops_class = mirrored_all_reduce_options[all_reduce_alg]
    return cross_device_ops_class(num_packs=num_packs)


def tpu_initialize(tpu_address):
    """Initializes TPU for TF 2.x training.

    Args:
      tpu_address: string, bns address of master TPU worker.

    Returns:
      A TPUClusterResolver.
    """
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu_address)
    if tpu_address not in ("", "local"):
        tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    return cluster_resolver


def get_distribution_strategy(device="gpu",
                              all_reduce_alg=None,
                              num_packs=1,
                              tpu_address=None,
                              **kwargs):
    """Return a DistributionStrategy for running the model.

    Args:
      device: a string specifying which device to use. Accepted values are
       "cpu", "gpu", "tpu", "gpu_multinode", and "gpu_multinode_async" -- case
        insensitive. "tpu" means to use TPUStrategy using `tpu_address`.
      all_reduce_alg: Optional. Specifies which algorithm to use when performing
        all-reduce. For `MirroredStrategy`, valid values are "nccl" and
        "hierarchical_copy". For `MultiWorkerMirroredStrategy`, valid values are
        "ring" and "nccl".  If None, DistributionStrategy will choose based on
        device topology.
      num_packs: Optional.  Sets the `num_packs` in `tf.distribute.NcclAllReduce`
        or `tf.distribute.HierarchicalCopyAllReduce` for `MirroredStrategy`.
      tpu_address: Optional. String that represents TPU to connect to. Must not be
        None if `distribution_strategy` is set to `tpu`.
      **kwargs: Additional kwargs for internal usages.

    Returns:
      tf.distribute.DistibutionStrategy object.
    Raises:
      ValueError: if `device` is not given string ; or if
        `device` is `tpu` but `tpu_address` is not specified.
    """
    del kwargs
    if not isinstance(device, str):
        msg = ("device must be a string but got: %s." % (device,))
        raise ValueError(msg)

    device = device.lower()

    if device == "tpu":
        # When tpu_address is an empty string, we communicate with local TPUs.
        cluster_resolver = tpu_initialize(tpu_address)
        return tf.distribute.TPUStrategy(cluster_resolver)

    if device == "cpu":
        return tf.distribute.OneDeviceStrategy("device:CPU:0")

    if device == "gpu":
        devices = tf.config.list_logical_devices('GPU')
        devices = [device.name for device in devices if device.device_type == 'GPU']
        num_gpus = len(devices)

        if num_gpus == 0:
            devices = ["device:CPU:0"]
        return tf.distribute.MirroredStrategy(
            devices=devices,
            cross_device_ops=_mirrored_cross_device_ops(all_reduce_alg, num_packs))

    if device == "gpu_multinode":
        return tf.distribute.experimental.MultiWorkerMirroredStrategy(
            communication=_collective_communication(all_reduce_alg))

    if device == "gpu_multinode_async":
        cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        return tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

    raise ValueError("Unrecognized Device Name: %r" %
                     device)


def configure_cluster(worker_hosts=None, task_index=-1):
    """Set multi-worker cluster spec in TF_CONFIG environment variable.

    Args:
      worker_hosts: comma-separated list of worker ip:port pairs.
      task_index: index of the worker.

    Returns:
      Number of workers in the cluster.
    """
    tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
    if tf_config:
        num_workers = (
                len(tf_config["cluster"].get("chief", [])) +
                len(tf_config["cluster"].get("worker", [])))
    elif worker_hosts:
        workers = worker_hosts.split(",")
        num_workers = len(workers)
        if num_workers > 1 and task_index < 0:
            raise ValueError("Must specify task_index when number of workers > 1")
        task_index = 0 if num_workers == 1 else task_index
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": {
                "worker": workers
            },
            "task": {
                "type": "worker",
                "index": task_index
            }
        })
    else:
        num_workers = 1
    return num_workers


def get_strategy_scope(strategy):
    if strategy:
        strategy_scope = strategy.scope()
    else:
        strategy_scope = DummyContextManager()

    return strategy_scope


class DummyContextManager(object):

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
