package nn;

import engine.Value;

import java.util.ArrayList;
import java.util.List;

public class MLP extends Module {
    /**
     * Multi-Layer perceptron
     */
    private List<Layer> layers;

    public MLP() {
        layers = new ArrayList<Layer>();
    }

    public void add(Layer layer) {
        this.layers.add(layer);
    }

    public Value[] forward(Value[] x) {
        // x has size (nSamples, nFeatures)
        int i = 0;
        for(Layer layer : layers) {
            x = layer.forward(x);
        }

        return x;
    }

    @Override
    public List<Value> parameters() {
        List<Value> out = new ArrayList<Value>();
        for(Layer layer : layers) {
            for(Value value : layer.parameters()) {
                out.add(value);
            }
        }

        return out;
    }

    @Override
    public void zeroGrad() {
        for(Value value : parameters()) {
            value.grad = 0.0f;
        }
    }
}
