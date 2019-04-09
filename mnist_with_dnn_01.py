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
train_y = train_y.reshape([-1, 1])
test_x, test_y = test
test_y = test_y.reshape([-1, 1])

train_x = np.reshape(train_x, [-1, 784])
test_x = np.reshape(test_x, [-1, 784])



# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)

W1 = tf.get_variable("W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.zeros([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.zeros([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.get_variable("W4", shape=[512, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.zeros([nb_classes]))

logits = tf.matmul(L3, W4)
hypothesis = tf.nn.relu(logits + b4)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=([Y_one_hot])))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.cast(tf.reshape(Y, [-1]), tf.int64))
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
                                 train_y[i*batch_size:(i+1)*batch_size],
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch


        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))


    print("Learning finished")

    # Test the model using test sets

    # print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
    #       X: test_x, Y: test_y}))

    a = sess.run([accuracy], feed_dict={X: test_x, Y: test_y})
    print(a)

    # Get one and predict
    r = random.randint(0, len(test_x))
    print("Label: ", test_y[r])
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: np.reshape(test_x[r], [1, -1])}))

    plt.imshow(
        test_x[r].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()

