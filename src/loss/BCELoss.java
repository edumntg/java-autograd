package loss;

import engine.Value;

public class BCELoss extends Loss {
    @Override
    public Value forward(Value[] yTrue, Value[] yPred) {
        int N = yTrue.length;
        Value out = new Value(0.0f);

        for(int i = 0; i < N; i++) {
            out = out.add(yTrue[i].mul(yPred[i].log()).add(yTrue[i].neg().add(1).mul(yPred[i].neg().add(1).log())));
        }

        return out.div(N).neg();
    }
}
