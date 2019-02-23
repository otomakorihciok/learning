"""Keras."""

import argparse
import functools
import os

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


def parse(example_proto):
  features = tf.parse_single_example(
      example_proto,
      features={
          'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })
  image = tf.decode_raw(features['image'], tf.uint8)
  image = tf.reshape(image, [32, 32, 3])
  image = tf.cast(image, tf.float32)
  image = tf.div(image, 255.0)
  label = features['label']
  label = tf.cast(label, tf.int32)

  return image, label


def dataset(filenames, batch_size, is_training=True):
  dataset = tf.data.TFRecordDataset(filenames).repeat()
  dataset = dataset.map(parse)
  if is_training:
    dataset = dataset.shuffle(buffer_size=3 * batch_size)

  dataset = dataset.batch(batch_size)
  return dataset


def input_fn(filenames, batch_size, is_training=True):
  dataset = tf.data.TFRecordDataset(filenames).repeat()
  dataset = dataset.map(parse)
  if is_training:
    dataset = dataset.shuffle(buffer_size=3 * batch_size)

  dataset = dataset.batch(batch_size)
  image_batch, label_batch = dataset.make_one_shot_iterator().get_next()
  return {'input_1': image_batch}, label_batch


def run_keras(model, data_dir, job_dir, **params):
  train_dataset = dataset([os.path.join(data_dir, 'train.tfrecords')],
                          params['train_batch_size'])
  eval_dataset = dataset([os.path.join(data_dir, 'eval.tfrecords')],
                         params['eval_batch_size'],
                         is_training=False)

  checkpoint_path = os.path.join(job_dir, 'cp-{epoch:04d}.ckpt')
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
      checkpoint_path,
      verbose=1,
      save_weights_only=True,
      # Save weights, every 5-epochs.
      period=5)

  tb_callback = tf.keras.callbacks.TensorBoard(log_dir=job_dir)

  # Training and evaluation.
  model.fit(
      train_dataset,
      epochs=params['epochs'],
      validation_data=eval_dataset,
      steps_per_epoch=int(params['train_steps'] / params['epochs']),
      validation_steps=100,
      callbacks=[cp_callback])

  # Save model.
  tf.contrib.saved_model.save_keras_model(model,
                                          os.path.join(job_dir, 'saved_model'))


def url_serving_input_receiver_fn():
  inputs = tf.placeholder(tf.string, None)
  image = tf.io.read_file(inputs)
  image = tf.image.decode_png(image, channels=3)
  image = tf.cast(image, tf.float32)
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)
  return tf.estimator.export.TensorServingInputReceiver(image,
                                                        {'image_url': inputs})


def run_estimator(model, data_dir, job_dir, **params):
  session_config = tf.ConfigProto(allow_soft_placement=True)

  config = tf.estimator.RunConfig(
      save_checkpoints_steps=100,
      save_summary_steps=100,
      model_dir=job_dir,
      session_config=session_config)

  # Create estimator.
  train_input_fn = functools.partial(
      input_fn, [os.path.join(data_dir, 'train.tfrecords')],
      params['train_batch_size'])

  eval_input_fn = functools.partial(input_fn,
                                    [os.path.join(data_dir, 'eval.tfrecords')],
                                    params['eval_batch_size'], False)

  # exporter = tf.estimator.FinalExporter('cifar10',
  #                                       url_serving_input_receiver_fn)
  # train_spec = tf.estimator.TrainSpec(
  #     train_input_fn, max_steps=params['train_steps'])
  # eval_spec = tf.estimator.EvalSpec(
  #     eval_input_fn, steps=100, exporters=[exporter], throttle_secs=0)

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=model, config=config)

  #tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  epoch = params['epochs']
  steps_per_epoch = int(params['train_steps'] / epoch)
  for _ in range(epoch):
    estimator.train(train_input_fn, steps=steps_per_epoch)
    estimator.evaluate(eval_input_fn, steps=100)

  estimator.export_savedmodel('cifar10', url_serving_input_receiver_fn)


def main(data_dir, job_dir, **params):
  model = tf.keras.applications.resnet50.ResNet50(
      weights=None, input_shape=(32, 32, 3), classes=10)

  optimizer = tf.train.MomentumOptimizer(
      params['learning_rate'], momentum=params['momentum'])

  model.compile(
      optimizer=optimizer,
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

  model.summary()

  run_estimator(model, data_dir, job_dir, **params)


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
      '--train-steps',
      type=int,
      default=80000,
      help='The number of steps to use for training.')
  parser.add_argument(
      '--epochs',
      type=int,
      default=1,
      help='The number of epochs to use for training.')
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
  args = parser.parse_args()
  main(**vars(args))
