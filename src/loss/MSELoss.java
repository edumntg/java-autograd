package loss;

import engine.Value;

public class MSELoss extends Loss {

    @Override
    public Value forward(Value[] yTrue, Value[] yPred) {
        int N = yTrue.length;
        Value out = new Value(0.0f);
        for(int i = 0; i < N; i++) {
            out = out.add(yTrue[i].sub(yPred[i]).pow(2));
        }

        // Finally, we divide the sum by N
        out = out.div(N);

        return out;
    }
}
