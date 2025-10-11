@Deprecated
public class SoftmaxNeuron extends Neuron{
    @Override
    public double compute(double input1, double input2, double input3) {
        double[] inputArray = new double[]{input1, input2, input3};
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
