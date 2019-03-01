"""Download machine learning dataset as image."""

from datetime import datetime
from io import BytesIO
import os
import shutil
import sys
import zipfile

from absl import app, flags
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
import google.auth
from google.cloud import storage
import numpy as np
from PIL import Image
import tensorflow as tf

keras = tf.keras

label_name = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def save_image(data, working_dir):
  Image.fromarray(data).save(
      os.path.join(working_dir,
                   datetime.now().strftime("%Y%m%d-%H%M%S-%f.png")))


def create_csv(data, working_dir, dataset_name='TRAIN'):
  global bucket
  blob_name = os.path.join(working_dir,
                           datetime.now().strftime("%Y%m%d-%H%M%S-%f.png"))
  blob = bucket.blob(blob_name)
  b = BytesIO()
  Image.fromarray(data[0]).save(b, format='png')
  blob.upload_from_string(b.getvalue(), content_type='image/png')
  label = label_name[np.asscalar(data[1])]
  url = 'gs://{}/{}'.format(bucket.name, blob_name)
  return '{},{},{}'.format(dataset_name, url, label)


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

    output_format = FLAGS.output_format
    if output_format == 'image':
      (x_train, _), (x_test, _) = dataset_module.load_data()
      all_data = np.concatenate((x_train, x_test), axis=0)
      if FLAGS.count is not None:
        np.random.shuffle(all_data)
        all_data = all_data[:FLAGS.count]
      print(len(all_data))
      _ = (p | 'Create source.' >> beam.Create(all_data) | 'Create image' >>
           beam.Map(lambda data: save_image(data, FLAGS.working_dir)))

      with zipfile.ZipFile(allowZip64=True) as zipf:
        [
            zipf.write(f)
            for f in tf.gfile.Glob('{}/*.png'.format(FLAGS.working_dir))
        ]

      [
          os.remove(f)
          for f in tf.gfile.Glob('{}/*.png'.format(FLAGS.working_dir))
      ]
    elif output_format == 'csv':
      global bucket
      credentials, project = google.auth.default()
      client = storage.Client(project=project, credentials=credentials)
      bucket = client.get_bucket(FLAGS.bucket)
      (x_train, y_train), (x_test, y_test) = dataset_module.load_data()

      url = 'gs://%s' % FLAGS.bucket
      train_data = zip(x_train, y_train)
      test_data = zip(x_test, y_test)
      if FLAGS.count is not None:
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        train_data = train_data[:FLAGS.count]
        test_data = test_data[:FLAGS.count]
      _ = (p | 'Create train source.' >> beam.Create(train_data) |
           'Create train csv' >>
           beam.Map(lambda data: create_csv(data, FLAGS.working_dir)) |
           'Save train csv' >> beam.io.WriteToText(
               os.path.join(url, 'csv', 'train'), file_name_suffix='.csv'))

      _ = (p | 'Create test source.' >> beam.Create(test_data) |
           'Create test csv' >> beam.Map(lambda data: create_csv(
               data, FLAGS.working_dir, dataset_name='VALIDATION')) |
           'Save test csv' >> beam.io.WriteToText(
               os.path.join(url, 'csv', 'test'), file_name_suffix='.csv'))


flags.DEFINE_string(name='working_dir', help='Working directory.', default='')
flags.DEFINE_enum(
    name='dataset',
    help='Dataset type.',
    enum_values=['mnist', 'fmnist', 'cifar10', 'cifar100'],
    default='mnist')
flags.DEFINE_integer(name='count', help='Number of data.', default=None)
flags.DEFINE_string(name='bucket', help='GCS bucket.', default=None)
flags.DEFINE_enum(
    name='output_format',
    help='Format of output. csv and image is allowed.',
    enum_values=['image', 'csv'],
    default='image')

if __name__ == '__main__':
  FLAGS = flags.FLAGS
  app.run(main)
