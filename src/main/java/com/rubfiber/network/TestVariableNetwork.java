package com.rubfiber.network;

import java.util.*;

public class TestVariableNetwork{
    public int hiddenLayersWidth;
    public int hiddenLayersDepth;
    public int inputLayer;
    public int outputLayer;
    List<InputVariableNeuron> InputLayerList = new ArrayList<>(); //list for input neurons
    List<VariableNeuron> HiddenLayerList = new ArrayList<>();
    List<VariableNeuron> OutputLayerList = new ArrayList<>();
    List<VariableNeuron> FullNetwork = new ArrayList<>();
    public List<Double> input;
    Double[] inputLayerOutput;
    Double[] outputLayerOutput;

    List<List<Double>> hiddenOutput = new ArrayList<>();

    /**
     *
     * @param inputLayer The input layer size
     * @param outputLayer The output layer size
     * @param hiddenLayersWidth The width of the hidden layers
     * @param hiddenLayersDepth The depth of the hidden layers
     * @param input The input of the network - must be the same size as the number of input neurons
     */
    TestVariableNetwork(int inputLayer, int outputLayer, int hiddenLayersWidth, int hiddenLayersDepth, Double[] input) {
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.hiddenLayersWidth = hiddenLayersWidth;
        this.hiddenLayersDepth = hiddenLayersDepth;

        this.input = List.of(input);
        inputLayerOutput = new Double[inputLayer];
        outputLayerOutput = new Double[outputLayer];

        // Input layer
        for (int i = 0; i < inputLayer; i++) {
            InputVariableNeuron ivn = new InputVariableNeuron(input[i]);
            ivn.setInput(new ArrayList<>(Collections.singletonList(input[i])));
            ivn.Initialize();
            InputLayerList.add(ivn);
        }

        // First hidden layer
        for (int neuron = 0; neuron < hiddenLayersWidth; neuron++) {
            VariableNeuron vn = new VariableNeuron();
            vn.setInput(new ArrayList<>(Collections.nCopies(inputLayer, 0.0)));
            vn.Initialize();
            HiddenLayerList.add(vn);
        }

        // Remaining hidden layers
        for (int layer = 1; layer < hiddenLayersDepth; layer++) {
            for (int neuron = 0; neuron < hiddenLayersWidth; neuron++) {
                VariableNeuron vn = new VariableNeuron();
                vn.setInput(new ArrayList<>(Collections.nCopies(hiddenLayersWidth, 0.0)));
                vn.Initialize();
                HiddenLayerList.add(vn);
            }
        }

        // Output layer
        for (int neuron = 0; neuron < outputLayer; neuron++) {
            VariableNeuron vn = new VariableNeuron();
            vn.setInput(new ArrayList<>(Collections.nCopies(hiddenLayersWidth, 0.0)));
            vn.Initialize();
            OutputLayerList.add(vn);
        }

        FullNetwork.addAll(InputLayerList);
        FullNetwork.addAll(HiddenLayerList);
        FullNetwork.addAll(OutputLayerList);
    }

    public Double[] Predict(Double[] input) {
        // clear previous hidden outputs
        hiddenOutput.clear();

        // --- Input Layer ---
        for (int i = 0; i < inputLayer; i++) {
            // always use fresh list for input
            InputLayerList.get(i).setInput(new ArrayList<>(Collections.singletonList(input[i])));
            inputLayerOutput[i] = InputLayerList.get(i).compute();
        }

        // --- Prepare hidden layers ---
        for (VariableNeuron neuron : HiddenLayerList) {
            neuron.clearInput();
        }

        // --- First hidden layer ---
        List<Double> firstLayerOutput = new ArrayList<>();
        for (int neuron = 0; neuron < hiddenLayersWidth; neuron++) {
            HiddenLayerList.get(neuron).setInput(new ArrayList<>(Arrays.asList(inputLayerOutput)));
            firstLayerOutput.add(HiddenLayerList.get(neuron).compute(false));
        }
        hiddenOutput.add(firstLayerOutput);

        // --- Remaining hidden layers ---
        for (int layer = 1; layer < hiddenLayersDepth; layer++) {
            List<Double> currentLayerOutput = new ArrayList<>();
            for (int neuron = 0; neuron < hiddenLayersWidth; neuron++) {
                int neuronID = layer * hiddenLayersWidth + neuron;
                VariableNeuron currentNeuron = HiddenLayerList.get(neuronID);

                // feed input from previous layer
                currentNeuron.setInput(new ArrayList<>(hiddenOutput.get(layer - 1)));
                currentNeuron.Initialize();
                currentLayerOutput.add(currentNeuron.compute(false));
            }
            hiddenOutput.add(currentLayerOutput);
        }

        // --- Output Layer ---
        for (int neuron = 0; neuron < outputLayer; neuron++) {
            OutputLayerList.get(neuron).setInput(new ArrayList<>(hiddenOutput.getLast()));
            OutputLayerList.get(neuron).Initialize();
            outputLayerOutput[neuron] = OutputLayerList.get(neuron).compute(true);
        }
        Double[] trueOutput = new Double[outputLayer];

        return outputLayerOutput;
    }



    Random random = new Random();
    int randomAnswer;
    List<Double[]> values = new ArrayList<>();
    List<Double[]> answers = new ArrayList<>();
    public void train(List<Double[]> values, List<Double[]> answers) {
        this.values = values;
        this.answers = answers;
        double learningRate = 1.0;
        for (int epoch = 0; epoch < 1000; epoch++) {
            System.out.println("epoch: " + epoch);
            for (int sample = 0; sample < values.size(); sample++) {
                System.out.println("sample: " + sample);
                randomAnswer = random.nextInt(values.size());

                //forward pass
                List<Double> networkOutput = this.forwardPass();
                //output layer
                List<Double> outputDelta = new ArrayList<>();
                for (int i = 0; i < networkOutput.size(); i++) {
                    double e = networkOutput.get(i) - answers.get(randomAnswer)[i];
                    outputDelta.add(e * networkOutput.get(i) * (1-networkOutput.get(i)));
                }

                List<List<Double>> hiddenDeltas = new ArrayList<>();
                //last hidden layer
                List<Double> lastHiddenDelta = new ArrayList<>();
                for (int i = 0; i < hiddenLayersWidth; i++) {
                    double lastHiddenSum = 0.0;
                    for (int j = 0; j < outputDelta.size(); j++) {
                        double weight = OutputLayerList.get(j).getWeights().get(i); //ith value in the weight list
                        lastHiddenSum += outputDelta.get(j) * weight;
                    }
                    double delta = lastHiddenSum * hiddenOutput.get(hiddenLayersDepth - 1).get(i) *
                            (1 - hiddenOutput.get(hiddenLayersDepth - 1).get(i));
                    lastHiddenDelta.add(delta);
                    hiddenDeltas.add(lastHiddenDelta);
                    HiddenLayerList.get(hiddenLayersWidth * (hiddenLayersDepth - 1) + i).setBias(HiddenLayerList.get(hiddenLayersWidth * (hiddenLayersDepth - 1) + i).getBias() - learningRate * delta);  //i-th value of last hidden layer
                }
                //for the rest of the hidden layers
                for (int layer = 0; layer < hiddenLayersDepth - 1; layer++) {
                    List<Double> layerDelta = new ArrayList<>();
                    for (int neuron  = 0; neuron < hiddenLayersWidth; neuron++) {
                        double delta = getDelta(layer, neuron, hiddenDeltas);
                        layerDelta.add(delta);
                        HiddenLayerList.get(layer * hiddenLayersWidth + neuron).setBias(HiddenLayerList.get(layer * hiddenLayersWidth + neuron).getBias() - learningRate * delta);
                    }
                    hiddenDeltas.addFirst(layerDelta);
                }
                //weight updates for output layer
                for (int i = 0; i < outputLayer; i++) {
                    VariableNeuron neuron = OutputLayerList.get(i);
                    if (neuron.getWeights().size() != neuron.getInput().size()) {
                        neuron.Initialize();
                    }

                    List<Double> newOutputWeights = new ArrayList<>();
                    for (int j = 0; j < neuron.getInput().size(); j++) {
                        double oldWeight = neuron.getWeights().get(j);
                        double deltaWeight = learningRate * outputDelta.get(i) *
                                hiddenOutput.getLast().get(j);
                        newOutputWeights.add(oldWeight - deltaWeight);
                    }
                    neuron.setWeights(newOutputWeights);
                    neuron.setBias(neuron.getBias() - learningRate * outputDelta.get(i));
                }

                //weight updates for hidden layers (excluding input layer)
                for (int layer = 1; layer < hiddenLayersDepth; layer++) {
                    for (int neuronIndex = 0; neuronIndex < hiddenLayersWidth; neuronIndex++) { //index of current neuron in current layer

                        int neuronID = layer * hiddenLayersWidth + neuronIndex;
                        VariableNeuron neuron = HiddenLayerList.get(neuronID);

                        // Ensure neuron has correct weights
                        int expectedInputSize = hiddenOutput.get(layer - 1).size();

                        List<Double> newWeights = new ArrayList<>();
                        if (neuron.getWeights().isEmpty() || neuron.getWeights().size() != expectedInputSize) {
                            neuron.setInput(new ArrayList<>(hiddenOutput.get(layer - 1)));
                            neuron.Initialize();
                        }
                        for (int prevIndex = 0; prevIndex < expectedInputSize; prevIndex++) { //neuron before
                            double oldWeight = neuron.getWeights().get(prevIndex);
                            double delta = learningRate * hiddenDeltas.get(layer).get(neuronIndex)
                                    * hiddenOutput.get(layer - 1).get(prevIndex);
                            newWeights.add(oldWeight - delta);
                        }
                        neuron.setWeights(newWeights);
                    }
                }

                //weight updates for first hidden layer
                for (int i = 0; i < hiddenLayersWidth; i++) {
                    VariableNeuron neuron = HiddenLayerList.get(i);
                    if (neuron.getWeights().size() != inputLayer) {
                        neuron.setInput(new ArrayList<>(Arrays.asList(inputLayerOutput)));
                        neuron.Initialize();
                    }

                    List<Double> newWeights = new ArrayList<>();
                    for (int j = 0; j < inputLayer; j++) {
                        double oldWeight = neuron.getWeights().get(j);
                        double delta = learningRate * hiddenDeltas.getFirst().get(i) * inputLayerOutput[j];
                        newWeights.add(oldWeight - delta);
                    }
                    neuron.setWeights(newWeights);
                }
            }
        }

    }

    private double getDelta(int layer, int neuron, List<List<Double>> hiddenDeltas) {
        double layerSum = 0.0;
        for (int nextLayerNeuron = 0; nextLayerNeuron < hiddenLayersWidth; nextLayerNeuron++) {
            int nextLayerStart = (layer + 1) * hiddenLayersWidth;
            double weight = HiddenLayerList
                    .get(nextLayerStart + nextLayerNeuron)
                    .getWeights()
                    .get(neuron);            layerSum += hiddenDeltas.get(layer + 1).get(nextLayerNeuron) * weight;
        }
        return layerSum * hiddenOutput.get(layer).get(neuron) * (1 - hiddenOutput.get(layer).get(neuron));
    }

    public List<Double> forwardPass() {
        List<Double> forwardPassOutput;
        for (int i = 0; i < inputLayer; i++) {
            InputLayerList.get(i).getInput().clear();
            InputLayerList.get(i).getInput().add(values.get(randomAnswer)[i]);
        }
        forwardPassOutput = Arrays.asList(this.Predict(values.get(randomAnswer)));
        return forwardPassOutput;
    }
}
