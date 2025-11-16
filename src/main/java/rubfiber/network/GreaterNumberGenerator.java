package rubfiber.network;

import java.util.*;

public class GreaterNumberGenerator {
    public static void main(String[] args) {
        Random rand = new Random();
        List<List<Double>> listRobotTrainData = new ArrayList<>();
        List<List<Double>> listRobotAnswers = new ArrayList<>();

        List<Double[]> arrayRobotTrainData = new ArrayList<>();
        List<Double[]> arrayRobotAnswers = new ArrayList<>();

        for (int i = 0; i < 1000; i++) {
            double left = rand.nextDouble();
            double center = rand.nextDouble();
            double right = rand.nextDouble();

            List<Double> input = Arrays.asList(left, center, right);
            listRobotTrainData.add(input);
            arrayRobotTrainData.add(new Double[]{left, center, right});

            List<Double> listOutput;
            Double[] arrayOutput;

            // Pick the direction with the *largest* open space
            if (center >= left && center >= right) {
                listOutput = Arrays.asList(0d, 1d, 0d); // go straight
                arrayOutput = new Double[]{0d, 1d, 0d};
            } else if (left >= center && left >= right) {
                listOutput = Arrays.asList(1d, 0d, 0d); // turn left
                arrayOutput = new Double[]{1d, 0d, 0d};
            } else {
                listOutput = Arrays.asList(0d, 0d, 1d); // turn right
                arrayOutput = new Double[]{0d, 0d, 1d};
            }

            listRobotAnswers.add(listOutput);
            arrayRobotAnswers.add(arrayOutput);
        }
        TestNetwork neuralNetwork = new TestNetwork();
        neuralNetwork.train(listRobotTrainData, listRobotAnswers);

        VariableNetwork variableNetwork = new VariableNetwork(3, 3, 6,6, new Double[]{0.0, 0.0, 0.0});
        variableNetwork.train(arrayRobotTrainData, arrayRobotAnswers);
        System.out.println("Trained fixed network robot prediction: " + neuralNetwork.predict(0.12, 0.007656, 0.99) + "\n");

        Double[] input = new Double[]{rand.nextDouble(), rand.nextDouble(), rand.nextDouble()};

        System.out.println("INPUT for trained network 1: " + Arrays.toString(input));
        System.out.println("Trained variable network robot prediction: " + Arrays.toString(variableNetwork.Predict(input)));
        System.out.println("1st prediction outputs per layer: \n input layer: " + Arrays.toString(variableNetwork.inputLayerOutput) +" \n hidden layer: " + variableNetwork.hiddenOutput + "\n output layer: " + Arrays.toString(variableNetwork.outputLayerOutput) + "\n\n");
        System.out.println("1st prediction inputs per layer: \n input layer: " + variableNetwork.getInput() + " \n hidden layers: " + variableNetwork.hiddenInput +  " \n output layer: " + variableNetwork.hiddenOutput.getLast() + "\n");

        System.out.println("last layer of prediction 1 inputs: " + variableNetwork.hiddenOutput.getLast() + "\n\n\n\n");
        System.out.println("1st prediction weights (output layer is the last list): " + variableNetwork.getWeights() + "\n\n");

        input = new Double[]{rand.nextDouble(), rand.nextDouble(), rand.nextDouble()};

        System.out.println("INPUT for trained network 2: " + Arrays.toString(input));
        System.out.println("Other Trained variable network prediction: " + Arrays.toString(variableNetwork.Predict(input)) + "\n");
        System.out.println("2nd prediction outputs per layer: \n input layer: " + Arrays.toString(variableNetwork.inputLayerOutput) +" \n hidden layers: " + variableNetwork.hiddenOutput + "\n output layer: " + Arrays.toString(variableNetwork.outputLayerOutput) + "\n");
        System.out.println("2nd prediction inputs per layer: \n input layer: " + variableNetwork.getInput() + " \n hidden layers: " + variableNetwork.hiddenInput +  " \n output layer: " + variableNetwork.hiddenOutput.getLast() + "\n");
        System.out.println("last layer of prediction 2 inputs: " + variableNetwork.hiddenOutput.getLast() + "\n\n\n");
        System.out.println("Bias (output layer is the last list): " + variableNetwork.getBias() + "\n\n");

        input = new Double[]{rand.nextDouble(), rand.nextDouble(), rand.nextDouble()};

        System.out.println("Other Trained variable network prediction: " + Arrays.toString(variableNetwork.Predict(input)) + "\n");

        System.out.println("Other Trained variable network prediction: " + Arrays.toString(variableNetwork.Predict(input)) + "\n");

        System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");

        VariableNetwork newNetwork = new VariableNetwork(3, 3, 6, 6, new Double[]{0.0, 0.0, 0.0});
        newNetwork.setWeights(variableNetwork.getWeights());
       newNetwork.setBias(variableNetwork.getBias());
        input = new Double[]{rand.nextDouble(), rand.nextDouble(), rand.nextDouble()};
        input = new Double[]{0.01, 0.99, 0.5};
        System.out.println(Arrays.toString(newNetwork.Predict(input)) + " input: " + Arrays.toString(input));

        newNetwork.saveNetworkState("src/main/resources/networkState.json");
        newNetwork.loadNetworkState("src/main/resources/networkState.json");
    }
}
