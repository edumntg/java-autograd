package nn;

import tensor.Tensor;

import java.util.List;

public class Module {
    private List<Tensor> parameters;

    public void zeroGrad() {
        for(Tensor p: parameters) {
            p.gradient.zero_();
        }
    }

    public List<Tensor> parameters() {
        return this.parameters;
    }

}
