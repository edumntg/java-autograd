package nn;

import engine.Value;
import tensor.Tensor;

import java.util.List;

public abstract class Module {
    public void zeroGrad() {
        return;
    }

    public List<Value> parameters() {
        return null;
    }

}
