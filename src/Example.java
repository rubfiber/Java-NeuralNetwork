import java.util.List;

public class Example {
    public static void main(String[] args) {
        Network neuralNetwork = new Network();
        List<Double> characterPrediction = neuralNetwork.predict(0.5, 0.5, 0.2); //Given a fictional character's health in percentage, ammunition (max 12.0), and distance to an enemy, see if it is a good idea to attack, run or hide
        System.out.println("Random prediction for fictional character:" + characterPrediction);
        List<Double> RobotPrediction = neuralNetwork.predict(0.3, 0.234, 0.938); //Based on distances from objects near a robot's left, right, and center, determine if it is a good idea to turn left, right or not turn at all
        System.out.println("Random prediction for Robot: " + RobotPrediction);
    }

}
