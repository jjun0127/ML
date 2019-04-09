import tensorflow as tf
import numpy as np

x = np.random.sample((10, 2))
dataset = tf.data.Dataset.from_tensor_slices(x)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


with tf.Session() as sess:
    while True:
        try:
            print(sess.run(next_element))
        except tf.errors.OutOfRangeError:
            break
