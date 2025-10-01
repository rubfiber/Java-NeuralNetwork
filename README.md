# A simple Neural Network
## Written in Java

This is an incredibly simple neural network. More coming soon, maybe.

# Installation

### Clone repo
```bash
git clone https://github.com/rubfiber/Java-NeuralNetwork
```

# Usage

### Neurons
There are two neuron classes: Neuron and VariableNeuron. Neuron has 3 inputs and is not very customizable, while VariableNeuron can have a variable amount of inputs and outputs the same number.

### Networks
There are also two neural network classes: Network and VariableNetwork. Network is not as customizable as VariableNetwork, but more efficient. VariableNetwork allows you to customize the input layer, width and depth of the hidden layers, and output layer.

### Training
Train the netwwork using the train() method. This method requires the data in the form of a List<List<Double>> and ansswers in the same form. See [Example.java](https://github.com/rubfiber/Java-NeuralNetwork/blob/master/src/Example.java) for examples on how to train the network.
