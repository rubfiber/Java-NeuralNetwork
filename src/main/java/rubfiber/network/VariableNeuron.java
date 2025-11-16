package rubfiber.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class VariableNeuron { //For neurons
    private final Random random = new Random(); //for doubles
    double bias = random.nextDouble(-1, 1); //Bias


    List<Double> getWeights() {
        return weights;
    }
    void setWeights(List<Double> newWeights) {
        weights.clear();
        weights = newWeights;
    }
    List<Double> getInput() {
        return input;
    }
    void setInput(List<Double> newInput) {
        input.clear();
        input = new ArrayList<>(newInput);
    }
    double getBias() {
        return bias;
    }
    public void clearInput() {
        input.clear();
        input = new ArrayList<>();
    }
    List<Double> input = new ArrayList<>(); //input as list
    List<Double> weights = new ArrayList<>();

    public void Initialize(int numInputs, int numOutputs) {
        bias = 0.0;
        weights.clear();

        if (input == null || input.isEmpty()) {
            throw new IllegalStateException("Neuron inputs not set before initialization");
        }
            for (int i = 0; i < input.size(); i++) {
                double limit = Math.sqrt(6.0 / (numInputs + numOutputs));
                weights.add(random.nextDouble(-limit, limit)); // xavier init
        }
    }



    void setBias(double newVal) {
        bias = newVal;
    }
    Double compute(boolean isOutputLayer) {
        if (input == null || weights == null)
            throw new IllegalStateException("Inputs or weights not set before compute()");
        if (input.size() != weights.size())
            throw new IllegalStateException("Input and weight size mismatch");

        double sum = 0.0;
        for (int i = 0; i < input.size(); i++) {
            if (input.get(i) == null || weights.get(i) == null)
                throw new IllegalStateException("Null input or weight detected");
            sum += input.get(i) * weights.get(i);
        }
        sum += bias;

        double activated;
        if (isOutputLayer) {
            activated = Math.tanh(sum);
        } else {
            activated = sigmoid(sum);
        }

        if (Double.isNaN(activated) || Double.isInfinite(activated))
            throw new IllegalStateException("NaN or Infinity in compute(). Sum=" + sum);

        return activated;
    }

    private double relu(double x) {return (x > 0) ? x : 0.01 * x; //not used for testing
    }

    private double sigmoid(double x) {
        if (x >= 0) {
            double z = Math.exp(-x);
            return 1 / (1 + z);
        } else {
            double z = Math.exp(x);
            return z / (1 + z);
        }
    }
}