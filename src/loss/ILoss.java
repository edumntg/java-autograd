package loss;

import engine.Value;
import tensor.Tensor;

public interface ILoss {
    public Tensor forward(Tensor yTrue, Tensor yPred) throws Exception;

    public Value forwardV(Value[] yTrue, Value[] yPred);
}
