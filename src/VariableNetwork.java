import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class VariableNetwork {
    public int hiddenLayersWidth;
    public int hiddenLayersDepth;
    public int inputLayer;
    public int outputLayer;
    List<VariableNeuron> InputLayerList = new ArrayList<>(); //list for input neurons
    List<VariableNeuron> HiddenLayerList = new ArrayList<>();
    List<VariableNeuron> OutputLayerList = new ArrayList<>();
    List<VariableNeuron> FullNetwork = new ArrayList<>();
    public List<Double> publicInput;
    Double[] inputLayerDoubles;
    Double[] hiddenLayerDoubles;
    Double[] outputLayerDoubles;

    VariableNetwork(int inputLayer, int outputLayer, int hiddenLayersWidth, int hiddenLayersDepth, Double[] input) {
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.hiddenLayersWidth = hiddenLayersWidth;
        this.hiddenLayersDepth = hiddenLayersDepth;
        //inputLayerDoubles and hiddenLayerDoubles are used later
        publicInput = List.of(input);
        inputLayerDoubles = new Double[inputLayer];
        hiddenLayerDoubles = new Double[hiddenLayersWidth * hiddenLayersDepth];
        outputLayerDoubles = new Double[outputLayer];

        for (int i = 0; i < inputLayer; i++) {
            InputLayerList.add(new VariableNeuron());
        }
        for (int i = 0; i < hiddenLayersDepth * hiddenLayersWidth; i++) { //depth (number of layers) * width (neurons per layer)
            HiddenLayerList.add(new VariableNeuron());
        }
        for (int i = 0; i < outputLayer; i++) {
            OutputLayerList.add(new VariableNeuron());
            OutputLayerList.get(i).input.add(publicInput.get(i));
            OutputLayerList.get(i).Initialize();
        }
        FullNetwork.addAll(InputLayerList);
        FullNetwork.addAll(HiddenLayerList);
        FullNetwork.addAll(OutputLayerList); //Full network
    }

    public Double[] Predict() {
        for (int i = 0; i < inputLayer; i++) {

            InputLayerList.get(i).Initialize();
            inputLayerDoubles[i] = (InputLayerList.get(i).compute());
        }
        for (int i = 0; i < hiddenLayersWidth; i++) { //first hidden layer needs input from the input layer
            HiddenLayerList.get(i).input.addAll(List.of(inputLayerDoubles));
            HiddenLayerList.get(i).Initialize();
            hiddenLayerDoubles[i] = HiddenLayerList.get(i).compute();
        }

        for (int layer = 1; layer < hiddenLayersDepth; layer++) {
            int startPrev = (layer - 1) * hiddenLayersWidth;
            int endPrev = layer * hiddenLayersWidth; // exclusive

            // For each neuron in this hidden layer
            for (int n = 0; n < hiddenLayersWidth; n++) {
                int neuronIndex = layer * hiddenLayersWidth + n;

                // Assign inputs from previous layer outputs
                HiddenLayerList.get(neuronIndex).input =
                        Arrays.asList(hiddenLayerDoubles).subList(startPrev, endPrev);

                HiddenLayerList.get(neuronIndex).Initialize();
                hiddenLayerDoubles[neuronIndex] = HiddenLayerList.get(neuronIndex).compute();
            }
        }
        for (int i = 0; i < outputLayer; i++) {
            OutputLayerList.get(i).input = List.of(hiddenLayerDoubles).subList(hiddenLayersWidth, hiddenLayerDoubles.length); //also pretty sketchy
            OutputLayerList.get(i).Initialize();
            outputLayerDoubles[i] = OutputLayerList.get(i).compute();
        }
        return outputLayerDoubles;
    }
}