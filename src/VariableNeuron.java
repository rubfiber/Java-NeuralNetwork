import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class VariableNeuron { //For neurons
    Random random = new Random(); //for doubles
    double bias = random.nextDouble(-1, 1); //Bias

    List<Double> input = new ArrayList<>(); //input as list
    List<Double> Weights = new ArrayList<>(); //weights
    List<Double> newWeights = new ArrayList<>(); //new weights for forget, mutate and remember
    List<Double> output = new ArrayList<>(); //Output before activation function
    void Initialize() { //Add waits and bias
        Weights.clear();
        double limit = 1.0 / Math.sqrt(input.size());
        for (Double ignored : input) { //my ide told me to rename this to ignored
            Weights.add((random.nextDouble(-1, 1) * 2 * limit) - limit); //equal number of weights as inputs
        }

    }
    Double compute() {
        output.clear();
       for (int i = 0; i < input.size(); i++) {
           output.add(input.get(i) * Weights.get(i)); //multiplies nth input by nth weight
       }
       double sum = 0.0;
       for (Double currentDouble : output) {
           sum += currentDouble; //adds the multiplication together
       }
       sum += bias; //adds sum to bias jut so bias is taken into account
        return (sum < 0) ? 0.01 * sum : sum;
    }
    double newBias;
    void mutate() {
        newBias = random.nextDouble(-1, 1);
        double factorOfChange = random.nextDouble(-1, 1);
        newWeights.clear();
        for (int j = 0; j < Weights.size(); j++) {
            double newVal = random.nextDouble(-1, 1);
            newWeights.add(newVal);
        }


        boolean biasOrNot = random.nextBoolean(); //chooses to change bias or weights. sorry for the variable names, I couldn't think of anything better.
        if (biasOrNot) {
            newBias += factorOfChange;
        } else {
            int Index = random.nextInt(newWeights.size());
            Double weightToChange = newWeights.get(Index); //chooses a random weight to change
            newWeights.set(Index, weightToChange + factorOfChange); //increases (or decreases) that weight by the double factorOfChange

        }

    }
    void forget() { //if the new value was worse than the old, restore the old value
        newBias = bias;
        newWeights = Weights;
    }
    void remember() { //if the new value is better than the old, keep it
        bias = newBias;
        Weights = newWeights;
    }
}