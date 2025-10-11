import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Network {
    List<Neuron> inputComputeLayer = Arrays.asList(
            new Neuron(), new Neuron(), new Neuron()); //input
    List<Neuron> hiddenLayer = Arrays.asList(
            new Neuron(), new Neuron(), new Neuron(), //hidden 1
            new Neuron(), new Neuron(), new Neuron() //hidden 2
    );
    List<Neuron> outputLayer = Arrays.asList(
            new Neuron(), new Neuron(), new Neuron()); //output
    List<Neuron> NeuronNetwork = new ArrayList<>();
    Network() { //add them together
        NeuronNetwork.addAll(inputComputeLayer);
        NeuronNetwork.addAll(hiddenLayer);
        NeuronNetwork.addAll(outputLayer);
    }

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

// Layer 4
        double n9 = NeuronNetwork.get(9).compute(n6, n7, n8);
        double n10 = NeuronNetwork.get(10).compute(n6, n7, n8);
        double n11 = NeuronNetwork.get(11).compute(n6, n7, n8);

// Output
        return Arrays.asList(n9, n10, n11);
    }
    public void train(List<List<Double>> values, List<List<Double>> answers) {
     /*  Double bestLossEpoch = null;
        for (int i = 0; i < 1000; i++) {

            List<List<Double>> predictions = new ArrayList<>();
            for (List<Double> value : values) { //for each value in values
                predictions.add(this.predict( //add the prediction from each of the lists in value's smaller lists - this.predict returns a List
                        value.get(0),         //you're basically adding this neural network's output into a list of predictions iterating through the training data
                        value.get(1),
                        value.get(2)
                ));
            }
            //output layer
            List<List<Double>> outputErrors = new ArrayList<>();
            for (int j = 0; j < predictions.size(); j++) {
                List<Double> predicted = predictions.get(j);
                List<Double> actual = answers.get(j);

                List<Double> sampleErrors = new ArrayList<>();
                for (Double aDouble : predicted) {
                    double error = aDouble - actual.get(j);
                    sampleErrors.add(error);
                }
                outputErrors.add(sampleErrors);
            }

            List<List<Double>> outputDeltas = new ArrayList<>();

            for (int j = 0; j < predictions.size(); j++) {
                List<Double> sampleDeltas = getDoubles(answers, predictions, j);
                outputDeltas.add(sampleDeltas);
            }
            List<Double> hiddenOutputs = new ArrayList<>();
            for (int j = 1; j < 3; j++) { //Number of hidden layers

                for (int k = 0; k < 3; k++) { //number of neurons per hidden layer
                    hiddenOutputs.add(NeuronNetwork.get(3 + j*k).compute(3*j-1, 3*j-2, 3*j-3));
                }

            }

            double[] hiddenDeltas = new double[6]; //size of input layer
            for (int h = 0; h <= 2; h++) {
                double currentHiddenOutput = hiddenOutputs.get(h);
                double downstreamError = 0.0;

                for (int o = 0; o < outputLayer.size(); o++) {
                    downstreamError += outputDeltas.get(o) * outputLayer.get(o).weights.get(h); //TODO: make a list for output hidden and input neurons
                }

                hiddenDeltas[h] = downstreamError * currentHiddenOutput * (1 - currentHiddenOutput);
            }


        }
    }
    double learningRate = 0.1;

    private static @NotNull List<Double> getDoubles(List<List<Double>> answers, List<List<Double>> predictions, int j) {
        List<Double> predicted = predictions.get(j);
        List<Double> actual = answers.get(j);

        List<Double> sampleDeltas = new ArrayList<>();
        for (int k = 0; k < predicted.size(); k++) {
            double output = predicted.get(k); // already sigmoid-activated
            double target = actual.get(k);

            double error = output - target;
            double delta = error * output * (1 - output); // (error × sigmoid derivative)
            sampleDeltas.add(delta);
        }
        return sampleDeltas;
    }

    public static Double meanSquareLoss(List<List<Double>> correctAnswers, List<List<Double>> predictedAnswers) {
        double sumSquare = 0;
        int totalCount = 0;
        for (int i = 0; i < correctAnswers.size(); i++) {
            for (int j = 0; j < correctAnswers.get(i).size(); j++) {
                double error = correctAnswers.get(i).get(j) - predictedAnswers.get(i).get(j);
                sumSquare += (error * error);
                totalCount++;
            }
        }
        return sumSquare / totalCount;

      */
    }

}