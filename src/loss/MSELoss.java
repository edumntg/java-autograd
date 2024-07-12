package loss;

import engine.Value;

public class MSELoss extends Loss {

    @Override
    public Value forwardV(Value[] yTrue, Value[] yPred) {
        int N = yTrue.length;
        Value[] out = new Value[N];
        for(int i = 0; i < N; i++) {
            out[i] = (yTrue[i].sub(yPred[i])).pow(2.0f); // (y - yhat)^2
        }

        Value out2 = out[0];
        for(int i = 1; i < N; i++) {
            out2 = out2.add(out[i]); // sum((y-yhat)^2)
        }

        return out2.div((float)N);
    }
}
