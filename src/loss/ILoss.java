package loss;

import tensor.Tensor;

public interface ILoss {
    public float forward(Tensor yTrue, Tensor yPred);
    public void backward();
}
