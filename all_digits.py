# Import data
import tensorflow as tf
import input_data
captcha = input_data.read_data_sets()

# Parameters
learning_rate = 1e-3
training_iters = 40000
batch_size = 100
display_step = 1

width = 140
height = 60

# Network Parameters
n_input = width * height
n_classes = 54 # Captcha total classes (0-9 digits)
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
    _X = tf.nn.relu(tf.add(_X, _biases['b1']))
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

    # Fully connected layer
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd2']), _biases['bd2'])) # Relu activation
    dense2 = tf.nn.dropout(dense2, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense2, _weights['out']), _biases['out'])
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.02)),
    'wd1': tf.Variable(tf.truncated_normal([35*15*64, 1024], stddev=0.02)),
    'wd2': tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.02)),
    'out': tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.constant(-0.05, shape=[width * height])),
    'bc1': tf.Variable(tf.constant(0.01, shape=[32])),
    'bc2': tf.Variable(tf.constant(0.01, shape=[64])),
    'bd1': tf.Variable(tf.truncated_normal([1024], stddev=0.01)),
    'bd2': tf.Variable(tf.truncated_normal([1024], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
pred_digits = tf.split(1, 6, pred)
y_digits = tf.split(1, 6, y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_digits[0], y_digits[0]))
cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_digits[1], y_digits[1]))
cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_digits[2], y_digits[2]))
cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_digits[3], y_digits[3]))
cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_digits[4], y_digits[4]))
cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_digits[5], y_digits[5]))
#cost = tf.nn.l2_loss(pred - y)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(tf.reshape(pred, [-1, 6, 9]),2), tf.argmax(tf.reshape(y, [-1, 6, 9]),2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), 0)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    min_accuracy = 0.15
    test_accuracy = 0.
    while test_accuracy < 0.9:
        batch_xs, batch_ys = captcha.train.next_batch(batch_size)
        current_accuracy = 0
        while current_accuracy < min_accuracy:
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            current_accuracy = sess.run(tf.reduce_min(accuracy), feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print(current_accuracy, end='; ', flush=True)
        # print()
        if min_accuracy < 0.9:
            min_accuracy += 0.005
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{0}".format(acc))
        test_acc_digigts = sess.run(accuracy, feed_dict={x: captcha.test.images[:450], y: captcha.test.labels[:450], keep_prob: 1.})
        test_accuracy = test_acc_digigts.min()
        print("Testing Accuracy:", test_acc_digigts)
        step += 1
    print("Optimization Finished!")
