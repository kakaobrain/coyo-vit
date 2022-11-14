# https://github.com/google/automl/blob/master/efficientnetv2/datasets.py

import functools
import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from . import build_dataset
from .preprocessing import Preprocessor


class TfdsInputLoader(object):
    """Generates TFDataset input_fn from a series of TFRecord files.
    The format of the data required is created by the script at:
        https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
    """

    def __init__(self,
                 is_training,
                 tfds_build_list,
                 preprocess_config=None,
                 mixup_alpha=0.,
                 cutmix_alpha=0.,
                 image_dtype=None,
                 image_size=224,
                 supervised_key='label',
                 normalize_label=False,
                 cache=False,
                 skip_decoding=True,
                 transpose_image=False,
                 debug=False):
        """Create an input from TFRecord files.
        Args:
          is_training: `bool` for whether the input is for training
          tfds_build_list: `list` of dictionaries containing tfds configuration
          preprocess_config: 'dict', containing preprocessing information
          about the order and methods. if None, do not apply any preprocessing method.
          image_dtype: image dtype. If None, use tf.float32.
          image_size: `int` for image size (both width and height).
          supervised_key: `str` indicating label feature key.
          cache: if true, fill the dataset by repeating from its cache.
          transpose_image: Whether to transpose the image. Useful for the "double
            transpose" trick for improved input throughput.
          debug: bool, If true, use deterministic behavior and add orig_image to dataset.
        """
        self.is_training = is_training
        self.tfds_build_list = tfds_build_list
        self.preprocess_config = preprocess_config or []
        self.image_dtype = image_dtype or tf.float32
        self.image_size = image_size
        self.supervised_key = supervised_key
        self.normalize_label = normalize_label
        self.cache = cache
        self.skip_decoding = skip_decoding
        self.transpose_image = transpose_image
        self.debug = debug

        # preprocessor
        self.preprocessor = Preprocessor(
            image_size=self.image_size,
            image_dtype=self.image_dtype,
            is_training=self.is_training
        )

        # for mixup and cutmix operation.
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

        # for input pipeline performance.
        self.file_buffer_size_m = None
        self.shuffle_size_k = 128

        # randomness
        self.shuffle_seed = 1111 if debug else None

        # build dataset
        self.builder_list = self.build

    @property
    def build(self):
        self.decoders = {}
        if self.skip_decoding:
            self.decoders['image'] = tfds.decode.SkipDecoding()

        builder_list = []
        self.info = {
            'num_examples': 0,
            'num_shards': 0,
            'num_classes': None,
        }

        for tfds_build_config in self.tfds_build_list:
            tfds_name = tfds_build_config.tfds_name
            tfds_data_dir = tfds_build_config.tfds_data_dir
            tfds_split = tfds_build_config.tfds_split

            logging.info('use TFDS: %s[%s]', tfds_name, tfds_split)

            if tfds_name:
                builder = tfds.builder(tfds_name, data_dir=tfds_data_dir)
            else:
                builder = tfds.core.builder_from_directory(tfds_data_dir)

            self.info['num_examples'] += builder.info.splits[tfds_split].num_examples
            self.info['num_shards'] += len(builder.info.splits[tfds_split].file_instructions)
            try:
                self.info['num_classes'] = builder.info.features[self.supervised_key].num_classes
            except:
                self.info['num_classes'] = 21841
                logging.warning(f'It\'s for coyo300M')
            builder_list.append([builder, tfds_split])

            logging.info(f'stacking dataset {tfds_name}[{tfds_split}] -> updated info: {self.info}')
        return builder_list

    def set_shapes(self, batch_size, features, labels):
        """Statically set the batch_size dimension."""
        features['image'].set_shape(features['image'].get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])))
        labels['label'].set_shape(labels['label'].get_shape().merge_with(
            tf.TensorShape([batch_size, None])))
        return features, labels

    def _get_null_input(self, data):
        """Returns a null image (all black pixels).
        Args:
          data: element of a dataset, ignored in this method, since it produces
              the same null image regardless of the element.
        Returns:
          a tensor representing a null image.
        """
        del data  # Unused since output is constant regardless of input
        return tf.zeros([self.image_size, self.image_size, 3], self.image_dtype)

    def input_fn(self, params):
        """Input function which provides a single batch for train or eval.
        Args:
          params: `dict` of parameters passed from the `TPUEstimator`.
              `params['batch_size']` is always provided and should be used as the
              effective batch size.
        Returns:
          A `tf.data.Dataset` object.
        """
        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # tf.estimator.tpu.RunConfig for details.
        batch_size = params['batch_size']

        if 'context' in params:
            current_host = params['context'].current_input_fn_deployment()[1]
            num_hosts = params['context'].num_hosts
        else:
            current_host = 0
            num_hosts = 1

        return self._input_fn(batch_size, current_host, num_hosts)

    def preprocess(self, features):
        """The preprocessing function."""
        image = self.preprocessor.preprocess_image(features['image'], self.preprocess_config)
        new_features = {'image': image}
        if self.debug:
            new_features['orig_image'] = features['image']

        label = self.preprocessor.preprocess_label(features, self.info['num_classes'], self.supervised_key,
                                                   self.normalize_label)
        new_label = {'label': label}

        if self.cutmix_alpha > 0.0:
            new_features['cutmix_mask'] = self.preprocessor.make_cutmix_mask(self.cutmix_alpha,
                                                                             self.image_size,
                                                                             self.image_size)

        return new_features, new_label

    def _input_fn(self,
                  batch_size,
                  current_host,
                  num_hosts,
                  tf_data_experimental_slack: bool = False
                  ):
        input_context = tf.distribute.InputContext(
            input_pipeline_id=current_host,  # Worker id
            num_input_pipelines=num_hosts,  # Total number of workers
        )

        read_config = tfds.ReadConfig(
            input_context=input_context,  # auto-shard
            interleave_cycle_length=10,
            interleave_block_length=1,
            shuffle_seed=self.shuffle_seed,  # for shuffle_files
            try_autocache=False,
            skip_prefetch=True,
        )

        ds = build_dataset.read_instruction_to_ds(
            builder_list=self.builder_list,
            shuffle_files=self.is_training and not self.debug,
            decoders=self.decoders,
            read_config=read_config,
        )

        if self.is_training:
            if self.cache:
                ds = ds.cache()
            ds = ds.shuffle(self.shuffle_size_k * 1024, seed=self.shuffle_seed)
        ds = ds.repeat()

        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)

        # Apply Mixup
        if self.is_training and (self.mixup_alpha > 0.0 or self.cutmix_alpha > 0.0):
            ds = ds.map(
                functools.partial(self.preprocessor.make_mix, batch_size, self.mixup_alpha, self.cutmix_alpha),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Assign static batch size dimension
        ds = ds.map(
            functools.partial(self.set_shapes, batch_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def transpose_image(features):
            # NHWC -> HWCN
            features['image'] = tf.transpose(features['image'], [1, 2, 3, 0])
            return features

        if self.transpose_image:
            # https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L135-L146
            ds = ds.map(
                lambda features, labels: (transpose_image(features), labels),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        options = tf.data.Options()
        options.deterministic = self.debug
        # https://www.tensorflow.org/guide/data_performance_analysis#3_are_you_reaching_high_cpu_utilization
        options.threading.max_intra_op_parallelism = 1
        # determined at runtime based on the number of available CPU cores.
        options.threading.private_threadpool_size = 0
        options.autotune.enabled = True
        options.experimental_slack = tf_data_experimental_slack
        ds = ds.with_options(options)
        return ds

    def distribute_dataset_fn(self, global_batch_size):
        """Dataset for tf.distribute.Strategy.distribute_datasets_from_function."""

        def dataset_fn(input_context):
            return self._input_fn(
                input_context.get_per_replica_batch_size(global_batch_size),
                input_context.input_pipeline_id, input_context.num_input_pipelines)

        return dataset_fn
