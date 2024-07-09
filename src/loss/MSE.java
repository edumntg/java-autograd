package loss;

import tensor.Tensor;

public class MSE extends Loss {
    @Override
    public float forward(Tensor yTrue, Tensor yPred) throws Exception {
        int N = yTrue.size();

        Tensor output = (yTrue.sub(yPred)).pow(2.0f);

        return (float)(1/N)*output.sum();
    }
}
