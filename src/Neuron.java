import java.util.Random;

public class Neuron { //For neurons
    Random randomDouble = new Random(); //for doubles
    private Double weight1 = randomDouble.nextDouble(-1, 1); //new double from -1 to 1
    private Double weight2 = randomDouble.nextDouble(-1, 1);
    private Double weight3 = randomDouble.nextDouble(-1, 1);
    private Double bias = randomDouble.nextDouble(-1, 1); //bias value
    public double compute(double input1, double input2, double input3) { //compute the output of the neuron
        double weightedSum = weight1 * input1 + weight2 * input2 + weight3 * input3 + bias;//weighted sum
        return 1/(1+Math.exp(-weightedSum)); //sigmoid activation function

    }
    public double getWeight1() {return weight1;}
    public double getWeight2() {return weight2;}
    public double getWeight3() {return weight3;}
    public double getBias() {return bias;}
    public void setWeight1(double newVal) {weight1 = newVal;}
    public void setWeight2(double newVal) {weight2 = newVal;}
    public void setWeight3(double newVal) {weight3 = newVal;}
    public void setBias(double newVal) {bias = newVal;}

    private double newWeight1 = randomDouble.nextDouble(-1, 1);
    private double newWeight2 = randomDouble.nextDouble(-1, 1);
    private double newWeight3 = randomDouble.nextDouble(-1, 1);
    private double newBias = randomDouble.nextDouble(-1, 1);
    @Deprecated //don't use anything below this comment cuz it's pretty bad
    public void mutate() {
        int mutateProperty = randomDouble.nextInt(0, 5);
        double factorOfChange = randomDouble.nextDouble(-1, 1);
        if (mutateProperty == 0) {
            newBias += factorOfChange;
        } else if (mutateProperty == 1) {
            newWeight1 += factorOfChange;
        } else if (mutateProperty == 2) {
            newWeight2 += factorOfChange;
        } else {
            newWeight3 += factorOfChange;
        }
    }
    @Deprecated
    public void forget() {
        newBias = bias;
        newWeight1 = weight1;
        newWeight2 = weight2;
        newWeight3 = weight3;
    }
    @Deprecated
    public void Remember() {
        bias = newBias;
        weight1 = newWeight1;
        weight2 = newWeight2;
        weight3 = newWeight3;
    }
}
