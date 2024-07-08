package optimizer;

import tensor.Tensor;

import java.util.List;

public class SGD extends Optimizer {
    public SGD(List<Tensor> parameters, float learningRate) {
        super(parameters, learningRate);
    }
}
