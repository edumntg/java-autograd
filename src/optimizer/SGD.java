package optimizer;

import engine.Value;
import tensor.Tensor;

import java.util.List;

public class SGD extends Optimizer {
    private float momentum;
    private float[] velocity;

    public SGD(List<Value> parameters, float learningRate) {
        super(parameters, learningRate);
        velocity = new float[this.parameters().size()];
    }

    public SGD(List<Value> parameters, float learningRate, float momentum) {
        super(parameters, learningRate);
        this.momentum = momentum;
        velocity = new float[this.parameters().size()];
    }

    @Override
    public void step() {
        int i = 0;
        for(Value param : this.parameters()) {
            velocity[i] = this.momentum * velocity[i] - this.lr * param.grad;
            param.data += velocity[i];
        }
    }
}
