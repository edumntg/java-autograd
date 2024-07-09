package loss;

import tensor.Tensor;

public class Loss implements ILoss {
    @Override
    public float forward(Tensor yTrue, Tensor yPred) {
        return 0;
    }

    @Override
    public void backward() {

    }
}
