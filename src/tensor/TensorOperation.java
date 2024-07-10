package tensor;

import enums.Operator;

import java.util.*;

public class TensorOperation {
    public Operator operation;
    public List<Tensor> childrens;

    public TensorOperation(Operator op, Tensor A, Tensor B) {
        this.operation = op;
        this.childrens = new ArrayList<Tensor>(Arrays.asList(A, B));
    }
}
