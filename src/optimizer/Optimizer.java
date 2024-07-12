package optimizer;

import engine.Value;
import tensor.Tensor;

import java.util.List;

public class Optimizer implements IOptimizer {
    private List<Value> parameters;
    public float lr;

    public Optimizer(List<Value> parameters, float learningRate) {
        if(learningRate <= 0.0f) {
            learningRate = 1E-3f;
        }
        this.parameters = parameters;
        this.lr = learningRate;
    }

    @Override
    public void step() {
        for(Value param : this.parameters) {
            param.optimize(this.lr);
        }
    }

    @Override
    public List<Value> parameters() {
        return this.parameters;
    }

    @Override
    public void print() {
        for(Value param: this.parameters) {
            System.out.println(param.toString());
        }
    }

    @Override
    public void zeroGrad() {
        // Reset all gradients
        for(Value param: this.parameters) {
            param.grad = 0.0f;
        }
    }
}
