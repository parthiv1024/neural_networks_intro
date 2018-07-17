# Introduction to Neural Networks
## Overview
This repository contains 2 programs. One is a single neuron neural network made from scratch using numpy.  
The second one is an image processing neural network created using Tensorflow and it works on the popular dataset MNIST.
## Dependencies
The neural_network.py program uses numpy. To install numpy run -  
sudo pip3 install numpy or sudo pip install numpy  
The tensorflow_neural_network.py program uses tensorflow. To install tensorflow run -  
sudo pip3 install tensorflow or sudo pip install tensorflow
## Usage
To execute the program run -  
python3 <path_to_repository>/neural_network.py  
python3 <path_to_repository>/tensorflow_neural_network.py
## NOTE
In the neural_network.py program, the neural network implemented is a single neuron with 3 inputs and 1 output. For a more realistic neural network that has multiple neurons and layer, gradient descent is used. In that case a vector of gradients (for the error function) is calculated for each of the weights and another vector is calculated for each of the biases, for the current weight and bias values and the current input values (an entire batch can be used and the sum of the values can give an overall direction). This vector is used to then update each of the individual weights and biases.
## Credits
Credits for this code go to Siraj Raval (||Source||)
