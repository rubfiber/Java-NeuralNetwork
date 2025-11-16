package rubfiber.network;

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
            InputLayerList.get(neuron).Initialize(inputLayer, outputLayer);
            //InputLayerList.get(neuron).Initialize(inpuLayer, outputLayer);
        }
        //init first hidden layer
        for (int neuron = 0; neuron < hiddenLayersWidth; neuron++) {
            HiddenLayerList.add(new VariableNeuron());
            HiddenLayerList.get(neuron).setInput(new ArrayList<>(Collections.nCopies(inputLayer, 0.0))); //fills with 0 just for initialization
            // HiddenLayerList.get(neuron).Initialize(inputLayer, outputLayer);
            HiddenLayerList.get(neuron).Initialize(inputLayer, outputLayer);
            HiddenLayerList.get(neuron).clearInput(); //initialize then clear
        }
        //init rest of hidden layers
        for (int neuron = hiddenLayersWidth; neuron < hiddenLayersWidth * hiddenLayersDepth; neuron++) {
            HiddenLayerList.add(new VariableNeuron());
            HiddenLayerList.get(neuron).setInput(new ArrayList<>(Collections.nCopies(hiddenLayersWidth, 0.0)));
            HiddenLayerList.get(neuron).Initialize(inputLayer, outputLayer);
            //HiddenLayerList.get(neuron).Initialize(inputLayer, outputLayer);
            HiddenLayerList.get(neuron).clearInput();
        }
        //init for output layer
        for (int neuron = 0; neuron < outputLayer; neuron++) {
            OutputLayerList.add(new VariableNeuron());
            OutputLayerList.get(neuron).setInput(new ArrayList<>(Collections.nCopies(hiddenLayersWidth, 0.0)));
            OutputLayerList.get(neuron).Initialize(inputLayer, outputLayer);
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
    public List<List<Double>> getBias() {
        List<List<Double>> biases = new ArrayList<>();
        for (int i = 0; i < hiddenLayersDepth; i++) {
            List<Double> currentLayerBiases = new ArrayList<>();
            for (int j = 0; j < hiddenLayersWidth; j++) {
                currentLayerBiases.add(HiddenLayerList.get((i * hiddenLayersWidth) + j).getBias());
            }
            biases.add(currentLayerBiases);
        }
        List<Double> outputBiasList = new ArrayList<>();
        for (int i = 0; i < outputLayer; i++) {
            outputBiasList.add(OutputLayerList.get(i).getBias());
        }
        biases.add(outputBiasList);
        return biases;
    }
    public void setBias(List<List<Double>> newBiases) {
        for (int i = 0; i < hiddenLayersDepth; i++) {
            List<Double> currentLayerBiases = new ArrayList<>(newBiases.get(i));
            for (int j = 0; j < hiddenLayersWidth; j++) {
                HiddenLayerList.get((i * hiddenLayersWidth) + j).setBias(currentLayerBiases.get(j));
            }
        }
        List<Double> outputBiasList = new ArrayList<>(newBiases.getLast());
        for (int i = 0; i < outputLayer; i++) {
            OutputLayerList.get(i).setBias(outputBiasList.get(i));
        }
    }
    public void setWeights(List<List<List<Double>>> newWeights) {
        for (int i = 0; i < hiddenLayersDepth; i++) {
            List<List<Double>> currentHiddenLayerWeights = new ArrayList<>(newWeights.get(i));
            for (int j = 0; j < hiddenLayersWidth; j++) {
                HiddenLayerList.get((i * hiddenLayersWidth) + j).setWeights(currentHiddenLayerWeights.get(j));
            }

        }
        List<List<Double>> outputWeightList = new ArrayList<>(newWeights.getLast());
        for (int i = 0; i < outputLayer; i++) {
            OutputLayerList.get(i).setWeights(outputWeightList.get(i));
        }
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
            double maxVal = Collections.max(freshInput.stream().map(Math::abs).toList());
            if (maxVal > 10.0) {
                freshInput.replaceAll(v -> v / maxVal);
            }

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

    Random random = new Random();
    int randomAnswer;
    List<Double[]> values = new ArrayList<>();
    List<Double[]> answers = new ArrayList<>();
    public void train(List<Double[]> values, List<Double[]> answers) {
        this.values = values;
        this.answers = answers;
        double learningRate = 0.001;
        for (int epoch = 0; epoch < 1000; epoch++) {
            for (int sample = 0; sample < values.size(); sample++) {
                //  System.out.println("sample: " + sample);
                randomAnswer = random.nextInt(values.size());

                //forward pass
                List<Double> networkOutput = this.forwardPass(values.get(sample));
                //output layer
                List<Double> outputDelta = new ArrayList<>();
                for (int i = 0; i < networkOutput.size(); i++) {
                    double out = networkOutput.get(i);
                    double target = answers.get(sample)[i];
                    double delta = (out - target) * (1 - out * out);

                    outputDelta.add(delta);
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
                   // double delta = Math.max(-1.0, Math.min(1.0, lastHiddenSum * hiddenOutput.get(hiddenLayersDepth - 1).get(i) *(1 - hiddenOutput.get(hiddenLayersDepth - 1).get(i))));
                        double delta = lastHiddenSum * hiddenOutput.get(hiddenLayersDepth - 1).get(i) * (1 - hiddenOutput.get(hiddenLayersDepth - 1).get(i));
                    lastHiddenDelta.add(delta);
                    HiddenLayerList.get(hiddenLayersWidth * (hiddenLayersDepth - 1) + i).setBias(HiddenLayerList.get(hiddenLayersWidth * (hiddenLayersDepth - 1) + i).getBias() - learningRate * delta);  //i-th value of last hidden layer
                    double newBias = HiddenLayerList.get(hiddenLayersWidth * (hiddenLayersDepth - 1) + i).getBias() - learningRate * delta;
                    if (Math.abs(newBias) > 3.0) newBias = 3.0 * Math.signum(newBias);
                    HiddenLayerList.get(hiddenLayersWidth * (hiddenLayersDepth - 1)+i).setBias(newBias);
                }
                hiddenDeltas.add(lastHiddenDelta);

                //for the rest of the hidden layers
                for (int layer = hiddenLayersDepth - 2; layer >= 0; layer--) {
                    List<Double> layerDelta = new ArrayList<>();
                    for (int neuron = 0; neuron < hiddenLayersWidth; neuron++) {
                        //double delta = Math.max(-1.0, Math.min(1.0, getDelta(layer, neuron, hiddenDeltas)));
                        double delta = getDelta(layer, neuron, hiddenDeltas);
                        layerDelta.add(delta);

                        VariableNeuron hiddenNeuron = HiddenLayerList.get(layer * hiddenLayersWidth + neuron);
                        hiddenNeuron.setBias(hiddenNeuron.getBias() - learningRate * delta);
                        double newBias = hiddenNeuron.getBias() - learningRate * delta;
                        if (Math.abs(newBias) > 3.0) newBias = 3.0 * Math.signum(newBias);
                        hiddenNeuron.setBias(newBias);

                    }
                    hiddenDeltas.addFirst(layerDelta); // add at bottom cuz we're working backwards
                }

                //weight updates for output layer
                for (int i = 0; i < outputLayer; i++) {
                    VariableNeuron neuron = OutputLayerList.get(i);
                    if (neuron.getWeights().size() != neuron.getInput().size()) {
                        neuron.Initialize(inputLayer, outputLayer);
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

                    if (Math.abs(neuron.getBias()) > 1) neuron.setBias(Math.signum(neuron.getBias()));
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
                            neuron.Initialize(inputLayer, outputLayer);
                        }
                        for (int prevIndex = 0; prevIndex < expectedInputSize; prevIndex++) { //neuron before
                            double oldWeight = neuron.getWeights().get(prevIndex);
                            //double delta = Math.max(-1.0, Math.min(1.0, learningRate * hiddenDeltas.get(layer).get(neuronIndex)* hiddenOutput.get(layer - 1).get(prevIndex)));
                            double delta = learningRate * hiddenDeltas.get(layer).get(neuronIndex)* hiddenOutput.get(layer - 1).get(prevIndex);
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
                        neuron.Initialize(inputLayer, outputLayer);
                    }

                    List<Double> newWeights = new ArrayList<>();
                    for (int j = 0; j < inputLayer; j++) {
                        double oldWeight = neuron.getWeights().get(j);
                        //double delta = Math.max(-1.0, Math.min(1.0, learningRate * hiddenDeltas.getFirst().get(i) * inputLayerOutput[j]));
                        double delta = learningRate * hiddenDeltas.getFirst().get(i)*inputLayerOutput[j];
                        newWeights.add(oldWeight - delta);
                    }
                    neuron.setWeights(newWeights);
                }
            }
        }

    }

    private double getDelta(int layer, int neuron, List<List<Double>> deltas) {
        double sum = 0.0;

        // Always use the most recent deltas (the "next" layer in backprop)
        List<Double> nextLayerDeltas = deltas.getLast();
        int nextLayerStart = (layer + 1) * hiddenLayersWidth;

        for (int nextNeuron = 0; nextNeuron < hiddenLayersWidth; nextNeuron++) {
            VariableNeuron next = HiddenLayerList.get(nextLayerStart + nextNeuron);
            double weightToThis = next.getWeights().get(neuron);
            sum += nextLayerDeltas.get(nextNeuron) * weightToThis;
        }

        double hOut = hiddenOutput.get(layer).get(neuron);
        return sum * hOut * (1 - hOut);
    }




    public List<Double> forwardPass(Double[] inputArray) {
        Double[] outputs = this.Predict(inputArray);
        return Arrays.asList(outputs);
    }

}