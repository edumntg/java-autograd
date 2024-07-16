package loss;

import engine.Value;

public class MAELoss extends Loss {
    @Override
    public Value forward(Value[] yTrue, Value[] yPred) {
        int N = yTrue.length;
        Value out = new Value(0.0f);

        for (int i = 0; i < yTrue.length; i++) {
            out = out.add(yTrue[i].sub(yPred[i]).abs());
        }

        return out.div(N);
    }
}
