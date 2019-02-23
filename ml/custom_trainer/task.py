"""Custom estimator."""

import tensorflow as tf
from absl.flags import DEFINE_list

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_file', None,
                    'The file pattern of input training data.')
flags.DEFINE_string('eval_file', None,
                    'The file pattern of input evaluation data.')
flags.DEFINE_string('job_dir', None,
                    'The directory where the model will be stored.')
flags.DEFINE_string('export_dir', None,
                    'The directory where the model will be saved.')
flags.DEFINE_integer('epochs', 1, 'The number of epochs to use for training.')
flags.DEFINE_integer('train_batch_size', 128, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 100, 'Batch size for validation.')
flags.DEFINE_integer('n_classes', 10, 'Number of classes.')
flags.DEFINE_float('momentum', 0.9, 'Momentum for MomentumOptimizer.')
flags.DEFINE_float(
    'learning_rate', 0.1,
    'This is the inital learning rate value. The learning rate will decrease'
    'during training. For more details check the model_fn implementation in'
    'this file.')
flags.DEFINE_list('input_shape', [32, 32, 3], 'Input shape of data.')


def model_fn_builder(input_shape, n_classes, init_lr, momentum, train_step):

  def model_fn(features, labels, mode, params):
    head = tf.contrib.estimator.multi_class_head(n_classes=n_classes)
    model = tf.keras.applications.resnet50.ResNet50(
        weights=None, input_shape=tuple(input_shape), classes=n_classes)

    logits = model(features['image'])

    learning_rate = tf.train.polynomial_decay(
        init_lr,
        tf.train.get_or_create_global_step(),
        train_step,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=momentum)
    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=optimizer,
        logits=logits)

  return model_fn


def input_fn_builder(filenames, input_shape, batch_size, is_training=True):
  name_to_features = {
      "image": tf.FixedLenFeature([], tf.string),
      "label": tf.FixedLenFeature([], tf.int64)
  }

  def _decode_record(record, name_to_features):
    example = tf.parse_single_example(record, name_to_features)
    image = example['image']
    label = example['label']

    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, input_shape)
    image = tf.cast(image, tf.float32)
    image = tf.div(image, 255.0)

    label = tf.to_int32(label)
    return {'image': image}, label

  def input_fn():
    dataset = tf.data.TFRecordDataset(filenames)
    if is_training:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(3 * batch_size)
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size))
    return dataset

  return input_fn


def url_serving_input_receiver_fn_builder(input_shape):

  def url_serving_input_receiver_fn():
    inputs = tf.placeholder(tf.string, None)
    image = tf.io.read_file(inputs)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_image_with_crop_or_pad(image, input_shape[0],
                                                   input_shape[1])
    return tf.estimator.export.ServingInputReceiver({'image': image},
                                                    {'image_url': inputs})

  return url_serving_input_receiver_fn


def get_records(filepattern):
  total = 0
  filenames = tf.gfile.Glob(filepattern)
  for filename in filenames:
    iter = tf.io.tf_record_iterator(filename)
    total += sum(1 for _ in iter)

  return total, filenames


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  config = tf.estimator.RunConfig(
      model_dir=FLAGS.job_dir, save_checkpoints_steps=1000)
  train_total_examples, train_examples = get_records(FLAGS.train_file)
  train_steps = int(
      train_total_examples / FLAGS.train_batch_size * FLAGS.epochs)

  input_shape = FLAGS.input_shape
  model_fn = model_fn_builder(input_shape, FLAGS.n_classes, FLAGS.learning_rate,
                              FLAGS.momentum, train_steps)

  estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)

  train_input_fn = input_fn_builder(train_examples, input_shape,
                                    FLAGS.train_batch_size)

  train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)

  eval_total_examples, eval_examples = get_records(FLAGS.eval_file)
  eval_steps = int(eval_total_examples // FLAGS.eval_batch_size)

  eval_input_fn = input_fn_builder(eval_examples, input_shape,
                                   FLAGS.eval_batch_size, False)

  exporter = tf.estimator.FinalExporter(
      FLAGS.export_dir, url_serving_input_receiver_fn_builder(input_shape))
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn, steps=eval_steps, exporters=[exporter])

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  flags.mark_flag_as_required("train_file")
  flags.mark_flag_as_required("eval_file")
  flags.mark_flag_as_required("job_dir")
  tf.app.run()
