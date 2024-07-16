package optimizer;

import engine.Value;

import java.util.List;

public class Adam extends Optimizer {
    public Adam(List<Value> parameters, float learningRate) {
        super(parameters, learningRate);
    }
}
