import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(777)  # for reproducibility

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train, test = tf.keras.datasets.mnist.load_data()

nb_classes = 10

train_x, train_y = train
test_x, test_y = test

train_x = np.reshape(train_x, [-1, 784])
test_x = np.reshape(test_x, [-1, 784])



# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.int32, [None, 1])
result_Y = tf.placeholder(tf.int32, [1, None])
Y_one_hot = tf.one_hot(Y, nb_classes)
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=([Y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.cast(result_Y, tf.int64))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # one_hot_train_y = tf.one_hot(train_y, nb_classes).eval()
    # one_hot_test_y = tf.one_hot(test_y, nb_classes).eval()
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(int(train_x.shape[0]) / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = train_x[i*batch_size:(i+1)*batch_size], \
                                 np.reshape(train_y[i*batch_size:(i+1)*batch_size], [-1, 1])
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch


        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))


    print("Learning finished")

    # Test the model using test sets

    # print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
    #       X: test_x, Y: test_y}))

    h, a = sess.run([tf.argmax(hypothesis, 1), accuracy], feed_dict={X: test_x, result_Y: np.reshape(test_y, [1,-1])})
    print(a)

    # Get one and predict
    # r = random.randint(0, mnist.test.num_examples - 1)
    # print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    # print("Prediction: ", sess.run(
    #     tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
    #
    # plt.imshow(
    #     mnist.test.images[r:r + 1].reshape(28, 28),
    #     cmap='Greys',
    #     interpolation='nearest')
    # plt.show()