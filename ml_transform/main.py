"""TensorFlow Transform main function."""

from __future__ import absolute_import, division, print_function

import os
import tempfile

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
import numpy as np
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from absl import app, flags

IMAGE_KEY = 'image'
LABEL_KEY = 'label'

TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'

RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
    dataset_schema.from_feature_spec({
        IMAGE_KEY:
        tf.FixedLenFeature([32, 32, 3], tf.float32),
        LABEL_KEY:
        tf.FixedLenFeature([], tf.int64)
    }))


def to_example(data):
  image = data[0]
  label = data[1]

  image = image.astype(np.float)
  image /= 255.0
  label = label.astype(np.int64)

  return tf.train.Example(
      features=tf.train.Features(
          feature={
              IMAGE_KEY:
              tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[image.tobytes()])),
              LABEL_KEY:
              tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
          })).SerializeToString()


def format(data):
  image = data[0]
  label = data[1]

  image = image.astype(np.float)
  image /= 255.0

  label = np.squeeze(label.astype(np.int64))

  return {IMAGE_KEY: image, LABEL_KEY: label}


def preprocessing_fn(inputs):
  return inputs


def transform_tft(train_data, test_data, working_dir):
  with beam.Pipeline() as pipeline:
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      raw_data = (pipeline | 'ReadTrainData' >> beam.Create(train_data) |
                  'CreateTrainData' >> beam.Map(lambda data: format(data)))
      raw_dataset = (raw_data, RAW_DATA_METADATA)
      transformed_dataset, transform_fn = (
          raw_dataset | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
      transformed_data, transformed_metadata = transformed_dataset
      transformed_data_coder = tft.coders.ExampleProtoCoder(
          transformed_metadata.schema)

      _ = (transformed_data |
           'EncodeTrainData' >> beam.Map(transformed_data_coder.encode) |
           'WriteTrainData' >> beam.io.WriteToTFRecord(
               os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)))

      raw_test_data = (pipeline | 'ReadTestData' >> beam.Create(test_data) |
                       'CreateTestData' >> beam.Map(lambda data: format(data)))
      raw_test_dataset = (raw_test_data, RAW_DATA_METADATA)

      transformed_test_dataset = ((raw_test_dataset, transform_fn) |
                                  tft_beam.TransformDataset())
      # Don't need transformed data schema, it's the same as before.
      transformed_test_data, _ = transformed_test_dataset

      _ = (transformed_test_data |
           'EncodeTestData' >> beam.Map(transformed_data_coder.encode) |
           'WriteTestData' >> beam.io.WriteToTFRecord(
               os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE)))

      _ = (transform_fn |
           'WriteTransformFn' >> tft_beam.WriteTransformFn(working_dir))


def transform(train_data, test_data, working_dir):
  """Transform the data and write out as a TFRecord of Example protos."""

  options = PipelineOptions()
  options.view_as(StandardOptions).runner = 'DirectRunner'
  with beam.Pipeline(options=options) as pipeline:
    _ = (pipeline | 'ReadTrainData' >> beam.Create(train_data) |
         'EncodeTrainData' >> beam.Map(lambda data: to_example(data)) |
         'WriteTrainData' >> beam.io.WriteToTFRecord(
             os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)))

    _ = (pipeline | 'ReadTestData' >> beam.Create(test_data) |
         'EncodeTestData' >> beam.Map(lambda data: to_example(data)) |
         'WriteTestData' >> beam.io.WriteToTFRecord(
             os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE)))


def main(_):
  dataset = FLAGS.dataset
  if dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  transform_tft(
      list(zip(x_train, y_train)), list(zip(x_test, y_test)), FLAGS.working_dir)


def define_flags():
  flags.DEFINE_enum(
      name='dataset',
      help='Dataset type.',
      enum_values=['mnist', 'cifar10', 'fmnist'],
      default='cifar10')

  flags.DEFINE_string(name='working_dir', help='Working directory.', default='')


if __name__ == '__main__':
  define_flags()
  FLAGS = flags.FLAGS
  app.run(main)
