package optimizer;

import engine.Value;

import java.util.List;

public interface IOptimizer {
    public void step();
    public List<Value> parameters();
    public void print();
    public void zeroGrad();


}
