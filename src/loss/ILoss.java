package loss;

import engine.Value;

public interface ILoss {
    public Value forward(Value[] yTrue, Value[] yPred);
}
