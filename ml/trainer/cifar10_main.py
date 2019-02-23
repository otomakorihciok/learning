# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""ResNet model for classifying images from CIFAR-10 dataset.

Support single-host training with one or multiple devices.

ResNet as proposed in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. arXiv:1512.03385

CIFAR-10 as in:
http://www.cs.toronto.edu/~kriz/cifar.html


"""
from __future__ import absolute_import, division, print_function

import argparse
import functools
import itertools
import os

import numpy as np
import six
import tensorflow as tf

from . import cifar10
tf.logging.set_verbosity(tf.logging.INFO)


def get_keras_model(params):
  model = tf.keras.applications.resnet50.ResNet50(
      weights=None, input_shape=(32, 32, 3), classes=10)
  optimizer = tf.train.MomentumOptimizer(
      learning_rate=params['learning_rate'], momentum=params['momentum'])

  model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      metrics=['accuracy'])

  return model


def input_fn(data_dir, subset, batch_size, use_distortion_for_training=True):
  """Create input graph for model.

  Args:
    data_dir: Directory where TFRecords representing the dataset are located.
    subset: one of 'train', 'validate' and 'eval'.
    batch_size: total batch size for training to be divided by the number of
    shards.
    use_distortion_for_training: True to use distortions.
  Returns:
    two lists of tensors for features and labels.
  """
  use_distortion = subset == 'train' and use_distortion_for_training
  dataset = cifar10.Cifar10DataSet(data_dir, subset, use_distortion)
  image_batch, label_batch = dataset.make_batch(batch_size)
  return image_batch, label_batch


def url_serving_input_receiver_fn():
  inputs = tf.placeholder(tf.string, None)
  image = tf.io.read_file(inputs)
  image = tf.image.decode_png(image, channels=3)
  image = tf.cast(image, tf.float32)
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)
  return tf.estimator.export.TensorServingInputReceiver(image,
                                                        {'image_url': inputs})


def main(job_dir, data_dir, use_distortion_for_training, log_device_placement,
         **hparams):
  # Session configuration.
  session_config = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=log_device_placement)

  config = tf.estimator.RunConfig(
      save_checkpoints_steps=1000,
      save_summary_steps=100,
      model_dir=job_dir,
      session_config=session_config)

  # Create estimator.
  train_input_fn = functools.partial(
      input_fn,
      data_dir,
      subset='train',
      batch_size=hparams['train_batch_size'],
      use_distortion_for_training=use_distortion_for_training)

  eval_input_fn = functools.partial(
      input_fn, data_dir, subset='eval', batch_size=hparams['eval_batch_size'])

  num_eval_examples = cifar10.Cifar10DataSet.num_examples_per_epoch('eval')
  if num_eval_examples % hparams['eval_batch_size'] != 0:
    raise ValueError('validation set size must be multiple of eval_batch_size')

  keras_model = get_keras_model(hparams)

  train_steps = hparams['train_steps']
  eval_steps = num_eval_examples // hparams['eval_batch_size']

  exporter = tf.estimator.FinalExporter('cifar10',
                                        url_serving_input_receiver_fn)

  train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn, steps=eval_steps, exporters=[exporter], throttle_secs=0)

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=keras_model, config=config)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      required=True,
      help='The directory where the CIFAR-10 input data is stored.')
  parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='The directory where the model will be stored.')
  parser.add_argument(
      '--num-layers',
      type=int,
      default=44,
      help='The number of layers of the model.')
  parser.add_argument(
      '--train-steps',
      type=int,
      default=80000,
      help='The number of steps to use for training.')
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=128,
      help='Batch size for training.')
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=100,
      help='Batch size for validation.')
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.9,
      help='Momentum for MomentumOptimizer.')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.1,
      help="""\
      This is the inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """)
  parser.add_argument(
      '--use-distortion-for-training',
      type=bool,
      default=True,
      help='If doing image distortion for training.')
  parser.add_argument(
      '--log-device-placement',
      action='store_true',
      default=False,
      help='Whether to log device placement.')
  args = parser.parse_args()

  if (args.num_layers - 2) % 6 != 0:
    raise ValueError('Invalid --num-layers parameter.')

  main(**vars(args))
