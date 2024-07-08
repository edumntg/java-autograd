package tensor;

import enums.Operator;

public class TensorOperation {
    public Operator operation;
    public Tensor[] actors;

    public TensorOperation(Operator op, Tensor A, Tensor B) {
        this.operation = op;
        this.actors = new Tensor[] {A, B};
    }
}
