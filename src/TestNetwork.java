import java.util.*;

public class TestNetwork extends Network {
    Random random = new Random();
    List<List<Double>> values = new ArrayList<>();
    List<List<Double>> answers = new ArrayList<>();
    int randomAnswer;

    TestNetwork() {
        NeuronNetwork.addAll(inputComputeLayer);
        NeuronNetwork.addAll(hiddenLayer);
        NeuronNetwork.addAll(outputLayer);
    }

    @Override
    public void train(List<List<Double>> values, List<List<Double>> answers) {
        this.values = values;
        this.answers = answers;
        double learningRate = 1.0;

        for (int epoch = 0; epoch < 1000; epoch++) {
            for (int sample = 0; sample < values.size(); sample++) {
                randomAnswer = random.nextInt(values.size());

                // Forward pass
                List<Double> networkOutput = this.forwardPass();

                // Output layer error and delta
                List<Double> error = new ArrayList<>();
                List<Double> outputDelta = new ArrayList<>();
                for (int i = 0; i < networkOutput.size(); i++) {
                    double e = networkOutput.get(i) - answers.get(randomAnswer).get(i);
                    error.add(e);
                    outputDelta.add(e * networkOutput.get(i) * (1 - networkOutput.get(i)));
                }

                // Hidden layer 2
                List<Double> hiddenL2Output = calcHiddenL2Output(values.get(randomAnswer).get(0),
                        values.get(randomAnswer).get(1), values.get(randomAnswer).get(2));
                List<Double> hiddenDeltaHL2 = new ArrayList<>();

                for (int i = 0; i < 3; i++) {
                    double sumHL2 = 0.0;
                    for (int j = 0; j < outputLayer.size(); j++) {
                        double weight = (i == 0) ? outputLayer.get(j).getWeight1() : (i == 1) ? outputLayer.get(j).getWeight2() : outputLayer.get(j).getWeight3();
                        sumHL2 += outputDelta.get(j) * weight;
                    }
                    double delta = sumHL2 * hiddenL2Output.get(i) * (1 - hiddenL2Output.get(i));
                    hiddenDeltaHL2.add(delta);
                    hiddenLayer.get(i + 3).setBias(hiddenLayer.get(i + 3).getBias() - learningRate * delta);
                }

                // Hidden layer 1
                List<Double> hiddenL1Output = calcHiddenL1Output(values.get(randomAnswer).get(0),
                        values.get(randomAnswer).get(1), values.get(randomAnswer).get(2));
                List<Double> hiddenDeltaHL1 = new ArrayList<>();

                for (int i = 0; i < 3; i++) {
                    double sumHL1 = 0.0;
                    for (int j = 0; j < 3; j++) {
                        Neuron neuron = hiddenLayer.get(j + 3);
                        double weight = (i == 0) ? neuron.getWeight1() : (i == 1) ? neuron.getWeight2() : neuron.getWeight3();
                        sumHL1 += hiddenDeltaHL2.get(j) * weight;
                    }
                    double delta = sumHL1 * hiddenL1Output.get(i) * (1 - hiddenL1Output.get(i));
                    hiddenDeltaHL1.add(delta);
                    hiddenLayer.get(i).setBias(hiddenLayer.get(i).getBias() - learningRate * delta);
                }

                // Weight updates (Output layer)
                for (int j = 0; j < outputLayer.size(); j++) {
                    Neuron neuron = outputLayer.get(j);
                    neuron.setWeight1(neuron.getWeight1() - learningRate * outputDelta.get(j) * hiddenL2Output.get(0));
                    neuron.setWeight2(neuron.getWeight2() - learningRate * outputDelta.get(j) * hiddenL2Output.get(1));
                    neuron.setWeight3(neuron.getWeight3() - learningRate * outputDelta.get(j) * hiddenL2Output.get(2));
                    neuron.setBias(neuron.getBias() - learningRate * outputDelta.get(j));
                }

                // Weight updates (Hidden layer 2)
                for (int j = 0; j < 3; j++) {
                    Neuron neuron = hiddenLayer.get(j + 3);
                    neuron.setWeight1(neuron.getWeight1() - learningRate * hiddenDeltaHL2.get(j) * hiddenL1Output.get(0));
                    neuron.setWeight2(neuron.getWeight2() - learningRate * hiddenDeltaHL2.get(j) * hiddenL1Output.get(1));
                    neuron.setWeight3(neuron.getWeight3() - learningRate * hiddenDeltaHL2.get(j) * hiddenL1Output.get(2));
                }

                // Weight updates (Hidden layer 1)
                for (int j = 0; j < 3; j++) {
                    Neuron neuron = hiddenLayer.get(j);
                    neuron.setWeight1(neuron.getWeight1() - learningRate * hiddenDeltaHL1.get(j) * values.get(randomAnswer).get(0));
                    neuron.setWeight2(neuron.getWeight2() - learningRate * hiddenDeltaHL1.get(j) * values.get(randomAnswer).get(1));
                    neuron.setWeight3(neuron.getWeight3() - learningRate * hiddenDeltaHL1.get(j) * values.get(randomAnswer).get(2));
                }
            }
        }
    }

    public List<Double> forwardPass() {
        return this.predict(values.get(randomAnswer).getFirst(), values.get(randomAnswer).get(1), values.get(randomAnswer).getLast());
    }

    List<Double> calcHiddenL2Output(Double inputA, Double inputB, Double inputC) {
        double n0 = NeuronNetwork.get(0).compute(inputA, inputB, inputC);
        double n1 = NeuronNetwork.get(1).compute(inputA, inputB, inputC);
        double n2 = NeuronNetwork.get(2).compute(inputA, inputB, inputC);

        double n3 = NeuronNetwork.get(3).compute(n0, n1, n2);
        double n4 = NeuronNetwork.get(4).compute(n0, n1, n2);
        double n5 = NeuronNetwork.get(5).compute(n0, n1, n2);

        double n6 = NeuronNetwork.get(6).compute(n3, n4, n5);
        double n7 = NeuronNetwork.get(7).compute(n3, n4, n5);
        double n8 = NeuronNetwork.get(8).compute(n3, n4, n5);

        return Arrays.asList(n6, n7, n8);
    }

    List<Double> calcHiddenL1Output(Double inputA, Double inputB, Double inputC) {
        double n0 = NeuronNetwork.get(0).compute(inputA, inputB, inputC);
        double n1 = NeuronNetwork.get(1).compute(inputA, inputB, inputC);
        double n2 = NeuronNetwork.get(2).compute(inputA, inputB, inputC);

        double n3 = NeuronNetwork.get(3).compute(n0, n1, n2);
        double n4 = NeuronNetwork.get(4).compute(n0, n1, n2);
        double n5 = NeuronNetwork.get(5).compute(n0, n1, n2);

        return Arrays.asList(n3, n4, n5);
    }
}