import java.util.Arrays;
import java.util.List;

class Network {
    List<Neuron> NeuronNetwork = Arrays.asList(
            new Neuron(), new Neuron(), new Neuron(), //Input Layer
            new Neuron(), new Neuron(), new Neuron(), //Hidden Layer 1
            new Neuron(), new Neuron(), new Neuron(), //Hidden Layer 2
            new Neuron(), new Neuron(), new Neuron(),              //Hidden Layer 3
            new Neuron(), new Neuron(), new Neuron());                            //Output Layer

    public List<Double> predict(Double inputA, Double inputB, Double inputC) {
// Layer 1
        double n0 = NeuronNetwork.get(0).compute(inputA, inputB, inputC);
        double n1 = NeuronNetwork.get(1).compute(inputA, inputB, inputC);
        double n2 = NeuronNetwork.get(2).compute(inputA, inputB, inputC);

// Layer 2
        double n3 = NeuronNetwork.get(3).compute(n0, n1, n2);
        double n4 = NeuronNetwork.get(4).compute(n0, n1, n2);
        double n5 = NeuronNetwork.get(5).compute(n0, n1, n2);

// Layer 3
        double n6 = NeuronNetwork.get(6).compute(n3, n4, n5);
        double n7 = NeuronNetwork.get(7).compute(n3, n4, n5);
        double n8 = NeuronNetwork.get(8).compute(n3, n4, n5);

// Layer 4 (assuming 2 neurons here: 9 and 10)
        double n9 = NeuronNetwork.get(9).compute(n6, n7, n8);
        double n10 = NeuronNetwork.get(10).compute(n6, n7, n8);
        double n11 = NeuronNetwork.get(11).compute(n6, n7, n8);

// Output
        return Arrays.asList(NeuronNetwork.get(12).compute(n9, n10, n11), NeuronNetwork.get(13).compute(n9, n10, n11), NeuronNetwork.get(14).compute(n9, n10, n11));
    }
}
/*
Inputs:  0  1   2
        | \ | / |
Hidden 1:3   4   5
        | \ | / |
Hidden 2:6   7   8
        | \ | /
Hidden 3:9 10
        \  |
Output:    11

 */