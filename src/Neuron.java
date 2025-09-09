import java.util.Random;

public class Neuron { //For neurons
    Random randomDouble = new Random(); //for doubles
    public Double weightA = randomDouble.nextDouble(-1, 1); //new double from -1 to 1
    public Double weightB = randomDouble.nextDouble(-1, 1);
    public Double weightC = randomDouble.nextDouble(-1, 1);
    public Double bias = randomDouble.nextDouble(-1, 1); //bias value
    public double compute(double inputA, double inputB, double inputC) { //compute the output of the neuron
        double weightedSum = weightA * inputA + weightB * inputB + weightC * inputC; //weighted sum
        return 1/(1+Math.exp(-weightedSum)); //sigmoid activation function

    }
}
