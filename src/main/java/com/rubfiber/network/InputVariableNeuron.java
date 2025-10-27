package com.rubfiber.network;

import java.util.ArrayList;
import java.util.List;

public class InputVariableNeuron extends VariableNeuron {

    InputVariableNeuron(Double inputValue) {
        this.input = new ArrayList<>(List.of(inputValue));
        this.weights = new ArrayList<>(); // no weights needed for input neurons
        this.bias = 0.0;
    }

    @Override
    public void setInput(List<Double> newInput) {
        // Keep reference consistent — overwrite content instead of replacing the list
        if (this.input == null) {
            this.input = new ArrayList<>(newInput);
        } else {
            this.input.clear();
            this.input.addAll(newInput);
        }
    }

    Double compute() {
        if (input == null || input.isEmpty()) {
            throw new IllegalStateException("Input neuron has no input!");
        }
        // Simply pass value forward (identity activation)
        return input.getFirst();
    }
}
