package loss;

import tensor.Tensor;

public interface ILoss {
    public Tensor forward(Tensor yTrue, Tensor yPred) throws Exception;
}
