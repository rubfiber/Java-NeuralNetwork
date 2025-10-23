package com.rubfiber.network;

import java.util.ArrayList;
import java.util.List;

public class InputVariableNeuron extends VariableNeuron {
    InputVariableNeuron(Double inputValue) {
        input = new ArrayList<>(List.of(inputValue));
    }

    @Override
    Double compute() {
        return input.getFirst(); // safer than getFirst()
    }
}