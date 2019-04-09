import tensorflow as tf
import numpy as np

train, test = tf.keras.datasets.mnist.load_data()
train_x, train_y = train
test_x, test_y = test

train_x = np.reshape(train_x, [-1, 784])
test_x = np.reshape(test_x, [-1, 784])

dict_x = {"image": train_x}

# dataset = tf.data.Dataset.from_tensor_slices(({"image": train_x}, train_y))
# dataset = dataset.shuffle(100000).repeat().batch(10)

# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()


classifier = tf.estimator.LinearClassifier(
    feature_columns=[tf.feature_column.numeric_column("image", shape=[1, 784])],
    n_classes=10
)

batch_size = 100
steps = 10000


def train_input_fn(x, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(({"image": x}, y))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


classifier.train(input_fn=lambda: train_input_fn(train_x, train_y.astype(np.int32), batch_size), steps=steps)


def test_input_fn(x, y, batch_size):
    x = {"image": x}
    if y is None:
        inputs = x
    else:
        inputs = (x, y)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)

    return dataset


result = classifier.evaluate(input_fn=lambda : test_input_fn(test_x, test_y.astype(np.int32), batch_size))

print(result)
