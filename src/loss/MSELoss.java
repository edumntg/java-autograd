package loss;

import tensor.Tensor;

public class MSELoss extends Loss {

    @Override
    public Tensor forward(Tensor yTrue, Tensor yPred) throws Exception {
        int N = yTrue.size();

        Tensor output = (yTrue.sub(yPred)).pow(2.0f);

        return output.sum().scalarDivision((float)N);
    }
}
