"""DatasetBuilder function.
(rewritten from tfds.core.dataset_builder)"""

import functools
import os
import tensorflow.compat.v2 as tf
from tensorflow_datasets.core import decode, example_parser
from tensorflow_datasets.core import tfrecords_reader
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.utils import read_config as read_config_lib
from tensorflow_datasets.core.utils import type_utils
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Type, Union


def read_instruction_to_ds(
        builder_list,
        shuffle_files: bool = False,
        decoders: Optional[type_utils.TreeDict[decode.Decoder]] = None,
        read_config: Optional[read_config_lib.ReadConfig] = None,
):
    base_builder, _ = builder_list[0]
    disable_shuffling = base_builder.info.disable_shuffling
    file_format = base_builder.info.file_format

    if isinstance(decoders, decode.PartialDecoding):
        features = decoders.extract_features(base_builder.info.features)
        example_specs = features.get_serialized_info()
        decoders = decoders.decoders
        # Full decoding (all features decoded)
    else:
        features = base_builder.info.features
        example_specs = base_builder._example_specs
        decoders = decoders  # pylint: disable=self-assigning-variable

    decode_fn = functools.partial(features.decode_example, decoders=decoders)
    _parser = example_parser.ExampleParser(example_specs)

    file_instructions = []
    for builder, tfds_split in builder_list:
        # Prepend path to filename
        file_instructions += [
            f.replace(filename=os.path.join(builder._data_dir, f.filename))
            for f in builder.info.splits[tfds_split].file_instructions
        ]

    ds = tfrecords_reader._read_files(
        file_instructions=file_instructions,
        read_config=read_config,
        shuffle_files=shuffle_files,
        disable_shuffling=disable_shuffling,
        file_format=file_format,
    )

    # Parse and decode
    def parse_and_decode(ex: utils.Tensor) -> utils.TreeDict[utils.Tensor]:
        # TODO(pierrot): `parse_example` uses
        # `tf.io.parse_single_example`. It might be faster to use `parse_example`,
        # after batching.
        # https://www.tensorflow.org/api_docs/python/tf/io/parse_example
        ex = _parser.parse_example(ex)
        if decode_fn:
            ex = decode_fn(ex)
        return ex

    # Eventually add the `tfds_id` after the decoding
    if read_config and read_config.add_tfds_id:
        parse_and_decode = functools.partial(
            tfrecords_reader._decode_with_id, decode_fn=parse_and_decode)

    ds = ds.map(
        parse_and_decode,
        num_parallel_calls=read_config.num_parallel_calls_for_decode,
    )
    return ds
