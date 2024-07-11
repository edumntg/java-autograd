package loss;

import engine.Value;
import tensor.Tensor;

public class MSELoss extends Loss {

    @Override
    public Tensor forward(Tensor yTrue, Tensor yPred) throws Exception {
        int N = yTrue.size();

        Tensor output = (yTrue.sub(yPred)).pow(2.0f);

        return output.sum().scalarDivision((float)N);
    }

    @Override
    public Value forwardV(Value[] yTrue, Value[] yPred) {
        int N = yTrue.length;
        Value[] out = new Value[N];
        for(int i = 0; i < N; i++) {
            out[i] = yTrue[i].sub(yPred[i]).pow(2.0f);
        }

        Value out2 = new Value(0.0f);
        for(int i = 0; i < N; i++) {
            out2 = out2.add(out[i]);
        }

        return out2.div((float)N);
    }
}
