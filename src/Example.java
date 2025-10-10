import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Example {
    public static void main(String[] args) {
        Network neuralNetwork = new Network();

        List<Double> RobotPrediction = neuralNetwork.predict(0.3, 0.234, 0.938); //Based on distances from objects near a robot's left, right, and center, determine if it is a good idea to turn left, right or not turn at all
        System.out.println("Random prediction for Robot: " + RobotPrediction);
        Double[] input = new Double[]{0.41, 0.85, 0.23};
        List<List<Double>> RobotTrainData = new ArrayList<>();
        RobotTrainData.add(Arrays.asList(0.384, 0.7, 0.579));//1
        RobotTrainData.add(Arrays.asList(0.713, 0.112, 0.430));//2
        RobotTrainData.add(Arrays.asList(0.234, 0.358, 0.932));//3
        RobotTrainData.add(Arrays.asList(0.582, 0.363, 0.746));//4
        RobotTrainData.add(Arrays.asList(0.582, 0.392, 0.681));//5
        RobotTrainData.add(Arrays.asList(0.582, 0.991, 0.573));//6
        RobotTrainData.add(Arrays.asList(0.864, 0.491, 0.373));//7

        List<List<Double>> RobotAnswers = new ArrayList<>();
        RobotAnswers.add(Arrays.asList(0d, 1d, 0d));//1
        RobotAnswers.add(Arrays.asList(1d, 0d, 0d));//2
        RobotAnswers.add(Arrays.asList(0d, 0d, 1d));//3
        RobotAnswers.add(Arrays.asList(0d, 0d, 1d));//4
        RobotAnswers.add(Arrays.asList(0d, 0d, 1d));//5
        RobotAnswers.add(Arrays.asList(0d, 1d, 0d));//6
        RobotAnswers.add(Arrays.asList(1d, 0d, 0d));//7
        neuralNetwork.train(RobotTrainData, RobotAnswers);
        System.out.println("Trained robot prediction: " + neuralNetwork.predict(0.8, 0.63, 0.21));

        VariableNetwork nn = new VariableNetwork(3, 9, 83, 45, input);
        System.out.println(Arrays.toString(nn.Predict()));

        VariableNeuron Neuron = new VariableNeuron();
        List<Double> inputLarge = new ArrayList<>();
        inputLarge.add(5747.0);
        inputLarge.add(949.0);
        inputLarge.add(5932.0);
        Neuron.input = inputLarge;
        Neuron.Initialize();
        System.out.println(Neuron.compute());
    }

}
