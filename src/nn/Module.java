package nn;

import engine.Value;

import java.util.List;

public abstract class Module {
    public void zeroGrad() {
        return;
    }

    public List<Value> parameters() {
        return null;
    }

}
