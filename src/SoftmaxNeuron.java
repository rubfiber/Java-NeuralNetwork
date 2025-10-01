import java.sql.Array;
@Deprecated
public class SoftmaxNeuron extends Neuron{
    @Override
    public double compute(double inputA, double inputB, double inputC) {
        double[] inputArray = new double[]{inputA, inputB, inputC};
        double sum = 0d;
        for (double i : calculateSoftmax(inputArray)) {
             sum += i;
        }
        return sum/inputArray.length;
    }

    public static double[] calculateSoftmax(double[] inputs) {
        if (inputs == null || inputs.length == 0) {
            return new double[0];
        }

        double sumExp = 0.0;
        for (double input : inputs) {
            sumExp += Math.exp(input);
        }

        double[] softmaxOutputs = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            softmaxOutputs[i] = Math.exp(inputs[i]) / sumExp;
        }

        return softmaxOutputs;
    }

}
