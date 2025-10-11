import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Example {
    public static void main(String[] args) {
        TestNetwork neuralNetwork = new TestNetwork();

        List<Double> RobotPrediction = neuralNetwork.predict(0.3, 0.234, 0.938); //Based on distances from objects near a robot's left, right, and center, determine if it is a good idea to turn left, right or not turn at all
        System.out.println("Random prediction for Robot: " + RobotPrediction);
        Double[] input = new Double[]{0.41, 0.85, 0.23};

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
