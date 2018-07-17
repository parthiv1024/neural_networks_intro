# Import MNIST data
import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

import tensorflow as tf

# Set hyper parameters
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

x = tf.placeholder("float", [None, 784]) # MNIST image has dimensions 28 x 28 = 784
y = tf.placeholder("float", [None, 10]) # Digits range from 0 to 9

# Create a model

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
	# Construct a linear model
	model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax activation function

# Add summary ops to collect data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

# More name scopes will clean up graph representations
with tf.name_scope("cost_function") as scope:
	# Minimize error using cross entropy
	cost_function = -tf.reduce_sum(y * tf.log(model))
	# Create a summary to monitor cost function
	tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train") as scope:
	# Gradient Descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initialize all variables
init = tf.initialize_all_variables()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	summary_writer = tf.summary.FileWriter('~/Documents/neural_networks_intro/data/logs', graph_def=sess.graph_def)

	# Training Cycle
	for iteration in range(training_iteration):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples/batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			# Fit training using batch data
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			# Compute the average loss
			avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
			# Write logs for each iteration
			summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
			summary_writer.add_summary(summary_str, iteration*total_batch + i)
		# Display logs per iteration step
		if iteration % display_step == 0:
			print("Iteration:", '%04d' % (iteration+1), "cost=", "{:.9f}".format(avg_cost))
	print("Tuning Completed")

	# Test the model
	predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	# Calculate Accuracy
	accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
	print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))