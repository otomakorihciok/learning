"""Download machine learning dataset as image."""

from datetime import datetime
import os
import shutil
import sys
import zipfile

from absl import app, flags
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
import numpy as np
from PIL import Image
import tensorflow as tf

keras = tf.keras


def save(data, working_dir):
  Image.fromarray(data).save(
      os.path.join(working_dir,
                   datetime.now().strftime("%Y%m%d-%H%M%S-%f.png")))


def main(_):
  dataset = FLAGS.dataset
  dataset_module = None
  if dataset == 'mnist':
    dataset_module = keras.datasets.mnist
  elif dataset == 'fmnist':
    dataset_module = keras.datasets.fashion_mnist
  elif dataset == 'cifar10':
    dataset_module = keras.datasets.cifar10
  elif dataset == 'cifar100':
    dataset_module = keras.datasets.cifar100
  else:
    raise ValueError('Dataset %s is not supported.' % dataset)

  try:
    os.makedirs(FLAGS.working_dir)
  except:
    pass

  options = PipelineOptions()
  options.view_as(StandardOptions).runner = 'DirectRunner'
  with beam.Pipeline(options=options) as p:
    (x_train, _), (x_test, _) = dataset_module.load_data()
    all_data = np.concatenate((x_train, x_test), axis=0)
    print(len(all_data))

    _ = (p | 'Create source.' >> beam.Create(all_data) |
         'Create image' >> beam.Map(lambda data: save(data, FLAGS.working_dir)))

  with zipfile.ZipFile(
      os.path.join(FLAGS.working_dir, 'dataset.zip'), 'w',
      allowZip64=True) as zip:
    [zip.write(f) for f in tf.gfile.Glob('{}/*.png'.format(FLAGS.working_dir))]

  [os.remove(f) for f in tf.gfile.Glob('{}/*.png'.format(FLAGS.working_dir))]


flags.DEFINE_string(name='working_dir', help='Working directory.', default='')
flags.DEFINE_enum(
    name='dataset',
    help='Dataset type.',
    enum_values=['mnist', 'fmnist', 'cifar10', 'cifar100'],
    default='mnist')

if __name__ == '__main__':
  FLAGS = flags.FLAGS
  app.run(main)
