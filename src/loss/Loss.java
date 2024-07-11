package loss;

import engine.Value;
import tensor.Tensor;

public class Loss implements ILoss {
    @Override
    public Tensor forward(Tensor yTrue, Tensor yPred) throws Exception {
        return null;
    }

    @Override
    public Value forwardV(Value[] yTrue, Value[] yPred) {
        return null;
    }
}
