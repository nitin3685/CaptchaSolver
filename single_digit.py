import tensorflow as tf
import pdb

# Import data
import input_data
captcha = input_data.read_data_sets(0)

# Parameters
learning_rate = 1e-3
training_iters = 40000
batch_size = 50
display_step = 1
display_test = 20

width = 140
height = 60

# Network Parameters
n_input = width * height # Captcha data input (img shape: 198*60)
n_classes = 9 # Captcha total classes (1-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(_X, _weights, _biases, _dropout):
    # _X = tf.nn.relu(tf.add(_X, _biases['b1']))

    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, width, height, 1])

    # Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=2)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)

    # Convolution Layer
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])) # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32],     stddev=0.05)), # 5x5 conv, 1 input, 32 outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64],    stddev=0.01)), # 5x5 conv, 32 inputs, 64 outputs
    'wd1': tf.Variable(tf.truncated_normal([35*15*64, 1024],  stddev=0.01)), # fully connected, 7*7*64 inputs, 1024 outputs
    'out': tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.1)) # 1024 inputs, 10 outputs (class prediction)
}

biases = {
    # 'b1':  tf.Variable(tf.constant(-0.05, shape=[width * height])),
    'bc1': tf.Variable(tf.constant(0.01, shape=[32])),
    'bc2': tf.Variable(tf.constant(0.001, shape=[64])),
    'bd1': tf.Variable(tf.constant(0.01, shape=[1024])),
    'out': tf.Variable(tf.constant(0.01, shape=[n_classes]))
}

# Add summary ops to collect data
tf.histogram_summary("wc1",   weights["wc1"])
tf.histogram_summary("wc2",   weights["wc2"])
tf.histogram_summary("wd1",   weights["wd1"])
tf.histogram_summary("w_out", weights["out"])
# tf.histogram_summary("b1",   biases["b1"])
tf.histogram_summary("bc1",   biases["bc1"])
tf.histogram_summary("bc2",   biases["bc2"])
tf.histogram_summary("bd1",   biases["bd1"])
tf.histogram_summary("b_out", biases["out"])
tf.histogram_summary("y", y)

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
with tf.name_scope("CostScope") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    tf.scalar_summary("cost", cost)

# Evaluate model
with tf.name_scope("AccuracyScope") as scope:
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.scalar_summary("accuracy", accuracy)

# Initializing the variables
init = tf.initialize_all_variables()

acc_list = []
loss_list = []

def learn_by_accuracy():
    with tf.Session() as sess:
        # Merge all the summaries and write them out to /tmp/mnist_logs
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./log", sess.graph.as_graph_def(add_shapes=True))
        # Run sesssion
        sess.run(init)
        step = 1

        min_accuracy = 0.15
        test_accuracy = 0.
        while test_accuracy < 0.95:
            batch_xs, batch_ys = captcha.train.next_batch(batch_size)

            current_accuracy = 0
            max_iterations = 0
            while current_accuracy < min_accuracy and max_iterations < 7:
                max_iterations += 1
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                current_accuracy, current_cost = sess.run([accuracy, cost], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print(current_accuracy, current_cost, min_accuracy)

            if min_accuracy < 0.95 and max_iterations < 7:
                min_accuracy += 0.005
            if step % display_step == 0:
                # Calculate batch accuracy and loss
                summary, _, _ = sess.run([merged, accuracy, cost], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                writer.add_summary(summary, step)
            test_accuracy = sess.run(accuracy, feed_dict={x: captcha.test.images[:450], y: captcha.test.labels[:450], keep_prob: 1.})
            print("Testing Accuracy:", test_accuracy)
            step += 1
        print("Optimization Finished!")
        # Calculate accuracy for 256 captcha test images
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: captcha.test.images[:450], y: captcha.test.labels[:450], keep_prob: 1.}))


def learn_by_examples():
    with tf.Session() as sess:
        # Merge all the summaries and write them out to /tmp/mnist_logs
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./log", sess.graph.as_graph_def(add_shapes=True))
        # Run sesssion
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_xs, batch_ys = captcha.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch accuracy and loss
                summary, _, _ = sess.run([merged, accuracy, cost], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                writer.add_summary(summary, step)
            step += 1
        print("Optimization Finished!")
        # Calculate accuracy for 256 captcha test images
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: captcha.test.images[:450], y: captcha.test.labels[:450], keep_prob: 1.}))


print("Optimization Started!")
# Launch the graph
learn_by_examples()
