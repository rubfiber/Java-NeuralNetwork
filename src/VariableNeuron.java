import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class VariableNeuron { //For neurons
    Random random = new Random(); //for doubles
    double bias = random.nextDouble(-1, 1); //Bias


    List<Double> getWeights() {
        return weights;
    }
    void setWeights(List<Double> newWeights) {
        weights = newWeights;
    }
    List<Double> getInput() {
        return input;
    }
    void setInput(List<Double> newInput) {
        input = newInput;
    }
    double getBias() {
        return bias;
    }
    void clearInput() {
        input.clear();
    }
    void clearWeights() {
        weights.clear();
    }

    List<Double> input = new ArrayList<>(); //input as list
    List<Double> weights = new ArrayList<>();
    List<Double> newWeights = new ArrayList<>(); //new weights for forget, mutate and remember

    public void Initialize() {
        if (input == null || input.isEmpty()) {
            System.out.println("bad input");
            throw new IllegalStateException("Neuron inputs not set before initialization");
        }
        if (weights.isEmpty()) {
            for (Double ignored : input) {
                weights.add((Math.random() * 2 - 1)); // Between -1 and 1
            }
        }

    }

    void setBias(double newVal) {
        bias = newVal;
    }
    Double compute() {
        if (input == null || weights == null) {
            throw new IllegalStateException("Inputs or weights not set before compute()");
        }
        if (input.size() != weights.size()) {
            throw new IllegalStateException("Input and weight size mismatch: " + input.size() + " vs " + weights.size());
        }

        double sum = 0.0;
        for (int i = 0; i < input.size(); i++) {
            sum += input.get(i) * weights.get(i);
        }
        sum += bias; // Add bias

        // Apply sigmoid activation
        double activated = sigmoid(sum);

        // Sanity check
        if (Double.isNaN(activated) || Double.isInfinite(activated)) {
            throw new IllegalStateException("NaN or Infinity detected in compute(). Sum=" + sum);
        }

        return activated;
    }

    private double sigmoid(double x) {
        // Numerically stable sigmoid to avoid overflow/underflow
        if (x >= 0) {
            double z = Math.exp(-x);
            return 1 / (1 + z);
        } else {
            double z = Math.exp(x);
            return z / (1 + z);
        }
    }

    double newBias;
    @Deprecated
    void mutate() {
        newBias = random.nextDouble(-1, 1);
        double factorOfChange = random.nextDouble(-1, 1);
        newWeights.clear();
        for (int j = 0; j < weights.size(); j++) {
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
    @Deprecated
    void forget() { //if the new value was worse than the old, restore the old value
        newBias = bias;
        newWeights = weights;
    }
    @Deprecated
    void remember() { //if the new value is better than the old, keep it
        bias = newBias;
        weights = newWeights;
    }
}