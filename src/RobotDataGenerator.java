import java.util.*;

public class RobotDataGenerator {
    public static void main(String[] args) {
        Random rand = new Random();
        List<List<Double>> robotTrainData = new ArrayList<>();
        List<List<Double>> robotAnswers = new ArrayList<>();

        for (int i = 0; i < 1000; i++) {
            double left = rand.nextDouble();
            double center = rand.nextDouble();
            double right = rand.nextDouble();

            List<Double> input = Arrays.asList(left, center, right);
            robotTrainData.add(input);

            List<Double> output;

            // Pick the direction with the *largest* open space
            if (center >= left && center >= right) {
                output = Arrays.asList(0d, 1d, 0d); // go straight
            } else if (left >= center && left >= right) {
                output = Arrays.asList(1d, 0d, 0d); // turn left
            } else {
                output = Arrays.asList(0d, 0d, 1d); // turn right
            }

            robotAnswers.add(output);
        }

        // Print the generated data
        //for (int i = 0; i < robotTrainData.size(); i++) {
       //     System.out.println("Input: " + robotTrainData.get(i) + " -> Output: " + robotAnswers.get(i));
       // }
        TestNetwork neuralNetwork = new TestNetwork();
        neuralNetwork.train(robotTrainData, robotAnswers);
        System.out.println("Trained robot prediction: " + neuralNetwork.predict(0.8, 0.93, 0.21));
    }
}
