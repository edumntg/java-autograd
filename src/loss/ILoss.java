package loss;

import tensor.Tensor;

public interface ILoss {
    public float forward(Tensor yTrue, Tensor yPred) throws Exception;
    public void backward();
}
