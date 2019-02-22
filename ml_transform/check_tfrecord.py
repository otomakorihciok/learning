"""Check TFRecord."""

import os
import struct
import sys

from PIL import Image
import tensorflow as tf
import tensorflow_transform as tft

filename = sys.argv[1]
directory = os.path.dirname(filename)

tf_transform_output = tft.TFTransformOutput(directory)
dataset = tf.data.experimental.make_batched_features_dataset(
    file_pattern=[filename],
    batch_size=100,
    features=tf_transform_output.transformed_feature_spec(),
    reader=tf.data.TFRecordDataset,
    shuffle=True)

dataset = dataset.map(
    lambda x: (tf.cast(x['image'] * 255.0, tf.uint8), x['label']))
batch_images, batch_labels = dataset.make_one_shot_iterator().get_next()

i = 0
with tf.Session() as sess:
  while True:
    try:
      data, label = sess.run([batch_images, batch_labels])
      i += 1
      if i % 100 == 0:
        Image.fromarray(data[0]).save('{}.png'.format(i))
    except tf.errors.OutOfRangeError:
      break

print(i)
