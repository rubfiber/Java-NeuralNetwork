package com.rubfiber.network;

import java.util.*;

public class VariableNetwork {
    public int hiddenLayersWidth;
    public int hiddenLayersDepth;
    public int inputLayer;
    public int outputLayer;
    List<InputVariableNeuron> InputLayerList = new ArrayList<>(); //list for input neurons
    List<VariableNeuron> HiddenLayerList = new ArrayList<>();
    List<VariableNeuron> OutputLayerList = new ArrayList<>();

    /**
     *
     * @param inputLayer The input layer size
     * @param outputLayer The output layer size
     * @param hiddenLayersWidth The width of the hidden layers
     * @param hiddenLayersDepth The depth of the hidden layers
     * @param input The input of the network - must be the same ize as the number of input neurons
     */


    VariableNetwork(int inputLayer, int outputLayer, int hiddenLayersWidth, int hiddenLayersDepth, Double[] input) {
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.hiddenLayersWidth = hiddenLayersWidth;
        this.hiddenLayersDepth = hiddenLayersDepth;

        //init input layer
        for (int neuron = 0; neuron < inputLayer; neuron++) {
            InputLayerList.add(new InputVariableNeuron(input[neuron]));
            InputLayerList.get(neuron).safeInitializeWeights(input.length);
            //InputLayerList.get(neuron).Initialize();
        }
        //init first hidden layer
        for (int neuron = 0; neuron < hiddenLayersWidth; neuron++) {
            HiddenLayerList.add(new VariableNeuron());
            HiddenLayerList.get(neuron).setInput(new ArrayList<>(Collections.nCopies(inputLayer, 0.0))); //fills with 0 just for initialization
           // HiddenLayerList.get(neuron).safeInitializeWeights(HiddenLayerList.get(neuron).getInput().size());
            HiddenLayerList.get(neuron).Initialize();
            HiddenLayerList.get(neuron).clearInput(); //initialize then clear
        }
        //init rest of hidden layers
        for (int neuron = hiddenLayersWidth; neuron < hiddenLayersWidth * hiddenLayersDepth; neuron++) {
            HiddenLayerList.add(new VariableNeuron());
            HiddenLayerList.get(neuron).setInput(new ArrayList<>(Collections.nCopies(hiddenLayersWidth, 0.0)));
            HiddenLayerList.get(neuron).Initialize();
            //HiddenLayerList.get(neuron).safeInitializeWeights(hiddenLayersWidth);
            HiddenLayerList.get(neuron).clearInput();
        }
        //init for output layer
        for (int neuron = 0; neuron < outputLayer; neuron++) {
            OutputLayerList.add(new VariableNeuron());
            OutputLayerList.get(neuron).setInput(new ArrayList<>(Collections.nCopies(hiddenLayersWidth, 0.0)));
            OutputLayerList.get(neuron).Initialize();
            OutputLayerList.get(neuron).clearInput();

        }
    }
    public List<Double> getInput() {
        List<Double> networkInput = new ArrayList<>();
        for (int i = 0; i < inputLayer; i++) {
            networkInput.add(InputLayerList.get(i).getInput().getFirst());
        }
        return networkInput;
    }
    public List<List<List<Double>>> getWeights() {
        List<List<List<Double>>> weights = new ArrayList<>();
        for (int i = 0; i < hiddenLayersDepth; i++) {
            List<List<Double>> currentHiddenLayerWeights = new ArrayList<>();
            for (int j = 0; j < hiddenLayersWidth; j++) {
                currentHiddenLayerWeights.add(HiddenLayerList.get((i * hiddenLayersWidth) + j).getWeights());
            }
            weights.add(currentHiddenLayerWeights);
        }
        List<List<Double>> outputWeightList = new ArrayList<>();
        for (int i = 0; i < outputLayer; i++) {
            outputWeightList.add(OutputLayerList.get(i).getWeights());
        }
        weights.add(outputWeightList);
        return weights;

    }
    Double[] inputLayerOutput;

    Double[] outputLayerOutput;
    List<List<Double>> hiddenOutput = new ArrayList<>();
    List<List<Double>> hiddenInput = new ArrayList<>();
    

    public Double[] Predict(Double[] input) {

        // clear previous hidden outputs
        hiddenOutput.clear();
        hiddenInput.clear();
        for (InputVariableNeuron n : InputLayerList) {n.clearInput();}
        for (VariableNeuron n: HiddenLayerList) {n.clearInput();}
        for (VariableNeuron n : OutputLayerList) {n.clearInput();}
        inputLayerOutput = new Double[inputLayer];
        outputLayerOutput = new Double[outputLayer];

        // --- Input Layer ---
        for (int i = 0; i < inputLayer; i++) {
            InputLayerList.get(i).setInput(new ArrayList<>(Collections.singletonList(input[(i)])));
            inputLayerOutput[i] = InputLayerList.get(i).compute();
        }

        // --- Prepare hidden layers ---
        for (VariableNeuron neuron : HiddenLayerList) {
            neuron.clearInput();
        }

        // --- First hidden layer ---
        List<Double> firstLayerOutput = new ArrayList<>();
        for (int neuron = 0; neuron < hiddenLayersWidth; neuron++) {
            List<Double> freshInput = new ArrayList<>(Arrays.asList(inputLayerOutput));
            HiddenLayerList.get(neuron).setInput(freshInput);
            firstLayerOutput.add(HiddenLayerList.get(neuron).compute(false));

        }
        hiddenOutput.add(new ArrayList<>(firstLayerOutput));
        hiddenInput.add(new ArrayList<>(Arrays.asList(inputLayerOutput)));
        // --- Remaining hidden layers ---
        for (int layer = 1; layer < hiddenLayersDepth; layer++) {
            List<Double> currentLayerOutput = new ArrayList<>();
            for (int neuron = 0; neuron < hiddenLayersWidth; neuron++) {
                int neuronID = layer * hiddenLayersWidth + neuron;
                VariableNeuron currentNeuron = HiddenLayerList.get(neuronID);

                // feed input from previous layer (do NOT re-initialize weights here)
                currentNeuron.setInput(new ArrayList<>(hiddenOutput.get(layer - 1)));
                currentLayerOutput.add(currentNeuron.compute(false));
            }
            hiddenInput.add(new ArrayList<>(hiddenOutput.get(layer - 1)));
            hiddenOutput.add(currentLayerOutput);
        }

        // --- Output Layer ---
        outputLayerOutput = new Double[outputLayer];
            List<Double> lastHidden = hiddenOutput.getLast();
            for (int neuron = 0; neuron < outputLayer; neuron++) {
                OutputLayerList.get(neuron).setInput(new ArrayList<>(lastHidden));
                outputLayerOutput[neuron] = OutputLayerList.get(neuron).compute(true);
            }
            return outputLayerOutput;
    }
    public static List<Double> normalize(List<Double> values) {
        double mean = values.stream().mapToDouble(v -> v).average().orElse(0.0);
        double variance = values.stream().mapToDouble(v -> Math.pow(v - mean, 2)).average().orElse(0.0);
        double std = Math.sqrt(variance + 1e-8);

        List<Double> normalized = new ArrayList<>();
        for (double v : values) {
            double z = (v - mean) / std;
            // Optional clamp
            if (z > 10.0) z = 10.0;
            if (z < -10.0) z = -10.0;
            normalized.add(z);
        }
        return normalized;
    }

    Random random = new Random();
    int randomAnswer;
    List<Double[]> values = new ArrayList<>();
    List<Double[]> answers = new ArrayList<>();
    public void train(List<Double[]> values, List<Double[]> answers) {
        this.values = values;
        this.answers = answers;
        double learningRate = 0.01;
        for (int epoch = 0; epoch < 1000; epoch++) {
            System.out.println("epoch: " + epoch);
            for (int sample = 0; sample < values.size(); sample++) {
              //  System.out.println("sample: " + sample);
                randomAnswer = random.nextInt(values.size());

                //forward pass
                List<Double> networkOutput = this.forwardPass(values.get(sample));
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
                            System.out.println("neuron weights empty");
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

    public List<Double> forwardPass(Double[] inputArray) {
        Double[] outputs = this.Predict(inputArray);
        return Arrays.asList(outputs);
    }

}