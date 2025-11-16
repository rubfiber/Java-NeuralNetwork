package rubfiber.network;

import java.util.ArrayList;
import java.util.List;

public class InputVariableNeuron extends VariableNeuron {

    InputVariableNeuron(Double inputValue) {
        this.setInput(new ArrayList<>(List.of(inputValue)));
        this.setWeights(new ArrayList<>()); // no weights needed for input neurons
        this.setBias(0.0);
    }

    @Override
    public void setInput(List<Double> newInput) {
        // Keep reference consistent — overwrite content instead of replacing the list
        if (this.getInput() == null) {
            this.setInput(new ArrayList<>(newInput));
        } else {
            input.clear();
            input.addAll(newInput);
        }
    }

    Double compute() {
        if (this.getInput() == null || this.getInput().isEmpty()) {
            throw new IllegalStateException("Input neuron has no input!");
        }
        // Simply pass value forward (identity activation)
        return this.getInput().getFirst();
    }
}
