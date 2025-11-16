# A simple Neural Network
## Written in Java

This is a pretty simple neural network. More coming soon, maybe.

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


Use the train() method each one has to train the network. You must input a list containing a list of input values that correspond to the input neuron (should be a ```List<List<Double>>```), and then a list of output values that the neuron should output, where each element of the list is a smaller list.

If the network is outputting wrong answers consistently after training, then you may need to update the learningRate variable within the train() method.
# Known bugs and issues

- There is currently an issue with bias updating when training that makes the network give an incorrect output. Giving the same network different input data also results in a prediction extremely similar to the first. This is likely an error in updating deltas, but I'm not sure.
- Pretty inefficient

