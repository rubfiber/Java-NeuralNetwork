import java.util.*;

public class VariableNetwork {
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
    Double[] hiddenLayerOutput;
    Double[] outputLayerOutput;

    List<List<Double>> hiddenOutput = new ArrayList<>();

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
        //inputLayerDoubles and hiddenLayerDoubles are used later
        this.input = List.of(input);
        inputLayerOutput = new Double[inputLayer];
        hiddenLayerOutput = new Double[hiddenLayersWidth * hiddenLayersDepth];
        outputLayerOutput = new Double[outputLayer];

        for (int i = 0; i < inputLayer; i++) {
            InputLayerList.add(new InputVariableNeuron(input[i]));
        }
        for (int i = 0; i < hiddenLayersDepth * hiddenLayersWidth; i++) { //depth (number of layers) * width (neurons per layer)
            HiddenLayerList.add(new VariableNeuron());
        }
        for (int i = 0; i < outputLayer; i++) {
            OutputLayerList.add(new VariableNeuron());
        }
        FullNetwork.addAll(InputLayerList);
        FullNetwork.addAll(HiddenLayerList);
        FullNetwork.addAll(OutputLayerList); //Full network
    }
    public Double[] Predict() {


        //input layer
        for (int neuron = 0; neuron < inputLayer; neuron++) {
            InputLayerList.get(neuron).setInput(new ArrayList<>(Collections.singletonList(input.get(neuron))));
            InputLayerList.get(neuron).Initialize();
            inputLayerOutput[neuron] = InputLayerList.get(neuron).compute();
        }
        //prep for output layer
        for (int i = 0; i < hiddenLayersWidth * hiddenLayersDepth; i++) {
            HiddenLayerList.get(i).clearInput();
        }
        //first hidden layer
        List<Double> firstLayerOutput = new ArrayList<>();
        for (int neuron = 0; neuron < hiddenLayersWidth; neuron++) {
            HiddenLayerList.get(neuron).setInput(new ArrayList<>(Arrays.asList(inputLayerOutput))); //it has to be changeable
            HiddenLayerList.get(neuron).Initialize();
            firstLayerOutput.add(HiddenLayerList.get(neuron).compute());
        }
        hiddenOutput.add(firstLayerOutput);
        //rest of hidden layers
// rest of hidden layers
        for (int layer = 1; layer < hiddenLayersDepth; layer++) { // exclude first layer
            List<Double> currentLayer = new ArrayList<>();
            for (int neuron = 0; neuron < hiddenLayersWidth; neuron++) {
                int neuronID = layer * hiddenLayersWidth + neuron;
                VariableNeuron currentNeuron = HiddenLayerList.get(neuronID);

                currentNeuron.setInput(new ArrayList<>(hiddenOutput.get(layer - 1)));
                try {
                    currentNeuron.Initialize();
                } catch (IllegalStateException e) {
                    System.out.println("ERROR: IllegalStateException at: \n current neuron: " + neuron + " at layer: " + layer + "\n");
                }

                currentLayer.add(currentNeuron.compute());
            }
            hiddenOutput.add(currentLayer);
        }

        //output layer
        for (int neuron = 0; neuron < outputLayer; neuron++) {
            OutputLayerList.get(neuron).setInput(hiddenOutput.getLast()); //input is the last hidden layer
            OutputLayerList.get(neuron).Initialize();
            outputLayerOutput[neuron] = OutputLayerList.get(neuron).compute();
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
            VariableNeuron nextNeuron = HiddenLayerList.get(nextLayerStart + nextLayerNeuron);
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
            InputLayerList.get(i).input.clear();
            InputLayerList.get(i).input.add(values.get(randomAnswer)[i]);
        }
        forwardPassOutput = List.of((this.Predict()));

        return forwardPassOutput;
    }
}