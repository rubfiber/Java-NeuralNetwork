import java.util.Random;

public class Neuron { //For neurons
    Random randomDouble = new Random(); //for doubles
    public Double weightA = randomDouble.nextDouble(-1, 1); //new double from -1 to 1
    public Double weightB = randomDouble.nextDouble(-1, 1);
    public Double weightC = randomDouble.nextDouble(-1, 1);
    public Double bias = randomDouble.nextDouble(-1, 1); //bias value
    public double compute(double inputA, double inputB, double inputC) { //compute the output of the neuron
        double weightedSum = weightA * inputA + weightB * inputB + weightC * inputC + bias;//weighted sum
        return 1/(1+Math.exp(-weightedSum)); //sigmoid activation function

    }
    public double newWeightA = randomDouble.nextDouble(-1, 1);
    public double newWeightB = randomDouble.nextDouble(-1, 1);
    public double newWeightC = randomDouble.nextDouble(-1, 1);
    public double newBias = randomDouble.nextDouble(-1, 1);
    public void mutate() {
        int mutateProperty = randomDouble.nextInt(0, 5);
        double factorOfChange = randomDouble.nextDouble(-1, 1);
        if (mutateProperty == 0) {
            newBias += factorOfChange;
        } else if (mutateProperty == 1) {
            newWeightA += factorOfChange;
        } else if (mutateProperty == 2) {
            newWeightB += factorOfChange;
        } else {
            newWeightC += factorOfChange;
        }
    }
    public void forget() {
        newBias = bias;
        newWeightA = weightA;
        newWeightB = weightB;
        newWeightC = weightC;
    }
    public void Remember() {
        bias = newBias;
        weightA = newWeightA;
        weightB = newWeightB;
        weightC = newWeightC;
    }
}
