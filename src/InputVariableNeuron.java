import java.util.ArrayList;
import java.util.List;

public class InputVariableNeuron extends VariableNeuron {
    InputVariableNeuron(Double inputValue) {
        input = new ArrayList<>(List.of(inputValue));
    }

    @Override
    Double compute() {
        return input.get(0); // safer than getFirst()
    }
}