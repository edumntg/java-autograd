package optimizer;

import tensor.Tensor;

import java.util.List;

public interface IOptimizer {
    public void step();
    public List<Tensor> getParameters();
    public List<Tensor> parameters();
    public void print();


}
