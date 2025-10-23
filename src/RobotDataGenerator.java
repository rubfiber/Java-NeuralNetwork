import java.util.*;

public class RobotDataGenerator {
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

        // Print the generated data
        /*
        for (int i = 0; i < listRobotTrainData.size(); i++) {
           System.out.println("Input: " + listRobotTrainData.get(i) + " -> Output: " + robotAnswers.get(i));
         }*/
        TestNetwork neuralNetwork = new TestNetwork();
        neuralNetwork.train(listRobotTrainData, listRobotAnswers);

        VariableNetwork variableNetwork = new VariableNetwork(3, 3, 6,6, new Double[]{0.8, 0.3, 0.2});
        variableNetwork.train(arrayRobotTrainData, arrayRobotAnswers);
        System.out.println("Trained fixed network robot prediction: " + neuralNetwork.predict(0.8, 0.93, 0.21));
        System.out.println("Trained variable network robot prediction: " + Arrays.toString(variableNetwork.Predict()));
    }
}
