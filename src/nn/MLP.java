package nn;

import engine.Value;

import java.util.ArrayList;
import java.util.List;

public class MLP extends Module {
    /**
     * Multi-Layer perceptron
     */
    private final List<Layer> layers;

    public MLP() {
        layers = new ArrayList<Layer>();
    }

    public void add(Layer layer) {
        this.layers.add(layer);
    }

    public Value[] forward(Value[][] x) {
        // x has size (nSamples, nFeatures)
        for(Layer layer : layers) {
//            System.out.printf("Before fforward: (%d, %d)\n", x.length, x[0].length);
            x = layer.forward(x);
//            System.out.printf("After fforward: (%d, %d)\n", x.length, x[0].length);
        }

        // In the end, x should be: (nSamples, 1), so we reduce it to just (nSamples,)
        return Value.flatten(x); // from 2d to 1d
    }

    @Override
    public List<Value> parameters() {
        List<Value> out = new ArrayList<Value>();
        for(Layer layer : this.layers) {
            out.addAll(layer.parameters());
        }

        return out;
    }

    public List<Layer> getLayers() {
        return this.layers;
    }

    @Override
    public void zeroGrad() {
        for(Value value : parameters()) {
            value.grad = 0.0f;
        }
    }
}
