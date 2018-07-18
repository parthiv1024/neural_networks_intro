from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		# Seed the random number generator
		# This makes sure that the randomly generated numbers are same for each run
		random.seed(1)

		# The neural network has a single neuron, with 3 input connections and 1 output connection
		# We create a 3 x 1 matrix for the weights of the synapses between the input layer and the neuron
		# The range of values for weights is from -1 to 1
		self.synaptic_weights = 2 * random.random((3,1)) - 1

	# The sigmoid function squishes down different values to between 0 and 1 and has an 'S' shaped graph
	# This is called normalization
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	# The derivative of the sigmoid function
	# It is the gradient of the sigmoid curve and shows how confident we are with the existing weight
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	# The neural networks are trained through a process of trial and error
	# Adjusting the synaptic weights each time
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in range(number_of_training_iterations):
			# Pass the training set through the neural network (a single neuron)
			output = self.think(training_set_inputs)

			# Calculate the error (The difference between the desired output and predicted output)
			error = training_set_outputs - output
			
			# Multiply the error by the input and again by the gradient of the Sigmoid curve.
			# This means less confident weights are adjusted more
			# This means inputs, which are zero, do not cause changes to the weights
			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

			# Adjust the weights
			self.synaptic_weights += adjustment

	# The neural network thinks
	def think(self, inputs):
		# Pass inputs through the neural network (a single neuron)
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":
	# Initialize a single neuron neural network
	neural_network = NeuralNetwork()

	print("Random starting synaptic weights:", neural_network.synaptic_weights)

	# The training set which has 4 examples, each with 3 input values
	# and 1 output value
	training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	training_set_outputs = array([[0, 1, 1, 0]]).T

	# Train the neural network using a training set
	# Do it 10,000 times and make small adjustments each time
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print("New synaptic weights after training:", neural_network.synaptic_weights)

	# Test neural network with new situation
	print("Considering new situation [1, 0, 0] ->", neural_network.think(array([1, 0, 0])))
