# A simple Neural Network
## Written in Java

This is an pretty simple neural network. More coming soon, maybe.

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

# Known bugs and issues

- There is currently an issue with bias updating when training that makes the network give an incorrect output. Giving the saem network different input data also results in a prediction extremely similar to the first. This is likely an error in updating deltas, but I'm not sure.
- Pretty inefficient

