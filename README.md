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
So far, only the regular network has a train() method. Here is the format:
```java
class Main {
    static void main(String[] args) {
        List<List<Double>> TrainData = new ArrayList<>();
        TrainData.add(Arrays.asList(0.384, 0.7, 0.579));//1
        TrainData.add(Arrays.asList(0.713, 0.112, 0.430));//2
        TrainData.add(Arrays.asList(0.234, 0.358, 0.932));//3
        TrainData.add(Arrays.asList(0.582, 0.363, 0.746));//4
        TrainData.add(Arrays.asList(0.582, 0.392, 0.681));//5
        TrainData.add(Arrays.asList(0.582, 0.991, 0.573));//6
        TrainData.add(Arrays.asList(0.864, 0.491, 0.373));//7
        //...
        List<List<Double>> Answers = new ArrayList<>();
        Answers.add(Arrays.asList(0d, 1d, 0d));//1
        Answers.add(Arrays.asList(1d, 0d, 0d));//2
        Answers.add(Arrays.asList(0d, 0d, 1d));//3
        Answers.add(Arrays.asList(0d, 0d, 1d));//4
        Answers.add(Arrays.asList(0d, 0d, 1d));//5
        Answers.add(Arrays.asList(0d, 1d, 0d));//6
        Answers.add(Arrays.asList(1d, 0d, 0d));//7
        //...

        Answers.train(RobotTrainData, RobotAnswers);
    }    
}


```
## To-do
### - Proper train method for variableNeuron
### - Proper example - such as number reading


