package loss;

import engine.Value;

public interface ILoss {
    public Value forwardV(Value[] yTrue, Value[] yPred);
}
