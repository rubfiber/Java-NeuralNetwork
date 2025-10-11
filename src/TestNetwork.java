import java.util.*;

public class TestNetwork extends Network{
    Random Random = new Random();
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
        double learningRate = 0.5;
        for (int epoch = 0; epoch < 1000; epoch++) {
            randomAnswer = Random.nextInt(0, values.size() - 1);
            //calculate error
            List<Double> networkOutput = this.forwardPass();
            List<Double> error = new ArrayList<>();
            List<Double> outputDelta = new ArrayList<>();
            for (int i = 0; i < networkOutput.size(); i++) {
                error.add(networkOutput.get(i) - answers.get(randomAnswer).get(i)); //gets the current error from the current example
                outputDelta.add(error.get(i) * networkOutput.get(i) * (1- networkOutput.get(i))); //multiplied by derivative of sigmoid for networkOutput layer
            }
            //HL deltas
            List<Double> hiddenDeltaHL2 = new ArrayList<>();
            double sumHL2 = 0.0; //sun for hidden layer 2
            List<Double> hiddenL2Output;
            hiddenL2Output = calcHiddenL2Output(values.get(randomAnswer).get(0), values.get(randomAnswer).get(1), values.get(randomAnswer).get(2));
            for (int i = 0; i < 3; i++) { //from the start to end of the 2ND layer
                for (int j = 0; j < outputLayer.size(); j++) { // for weights
                    if (j == 0) { //sorry this was made with if statements and not some other method - for VariableNeuron I have a better approach
                        sumHL2 += outputDelta.get(j) * outputLayer.get(j).getWeight1();
                        outputLayer.get(j).setWeight1(-learningRate * networkOutput.get(j) * outputDelta.get(j));

                    } else if (j ==1) {
                        sumHL2 += outputDelta.get(j) * outputLayer.get(j).getWeight2();
                        outputLayer.get(j).setWeight2(-learningRate * networkOutput.get(j) * outputDelta.get(j));

                    } else if (j == 2) {
                        sumHL2 += outputDelta.get(j) * outputLayer.get(j).getWeight3();
                        outputLayer.get(j).setWeight3(-learningRate * networkOutput.get(j) * outputDelta.get(j));
                    }

                }
                System.out.println(hiddenL2Output.get(i));
                hiddenDeltaHL2.add(sumHL2 * (hiddenL2Output.get(i) * (1-hiddenL2Output.get(i)))); //pretty sketchy
            }
            List<Double> hiddenL1Output, hiddenDeltaHL1 = new ArrayList<>();
            double sumHL1 = 0.0;
            hiddenL1Output = calcHiddenL1Output(values.get(randomAnswer).get(0), values.get(randomAnswer).get(1), values.get(randomAnswer).get(2));
            for (int i = 0; i < 3; i++) { //for start to end of 1ST hidden layer
                for (int j = 0; j < hiddenLayer.size() - 3; j++) { //iterating through the weights from HL2
                    if (j == 3) {
                        sumHL1 += hiddenDeltaHL2.get(j) * hiddenLayer.get(j).getWeight1();
                        hiddenLayer.get(j).setWeight1(-learningRate * hiddenL2Output.get(j) * hiddenDeltaHL1.get(j));

                    } else if (j == 4) {
                        sumHL1 += hiddenDeltaHL2.get(j) * hiddenLayer.get(j).getWeight2();
                        hiddenLayer.get(j).setWeight2(-learningRate * hiddenL2Output.get(j) * hiddenDeltaHL1.get(j));

                    } else if (j == 5) {
                        sumHL1 += hiddenDeltaHL2.get(j) * hiddenLayer.get(j).getWeight3();
                        hiddenLayer.get(j).setWeight3(-learningRate * hiddenL2Output.get(j) * hiddenDeltaHL1.get(j));

                    }
                }
                hiddenDeltaHL1.add(sumHL1 * (hiddenL1Output.get(i) * (1-hiddenL1Output.get(i))));
            }
        }
    }
    public List<Double> forwardPass() {
        return this.predict(values.get(randomAnswer).getFirst(), values.get(randomAnswer).get(1), values.get(randomAnswer).getLast());
    }
     List<Double> calcHiddenL2Output(Double inputA, Double inputB, Double inputC) { //mb for this trust VariableNetwork will be better
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
    List<Double> calcHiddenL1Output(Double inputA, Double inputB, Double inputC) { //pretty bad but it works (hopefully)
        double n0 = NeuronNetwork.get(0).compute(inputA, inputB, inputC);
        double n1 = NeuronNetwork.get(1).compute(inputA, inputB, inputC);
        double n2 = NeuronNetwork.get(2).compute(inputA, inputB, inputC);

        double n3 = NeuronNetwork.get(3).compute(n0, n1, n2);
        double n4 = NeuronNetwork.get(4).compute(n0, n1, n2);
        double n5 = NeuronNetwork.get(5).compute(n0, n1, n2);
        return Arrays.asList(n3, n4, n5);
    }

}
