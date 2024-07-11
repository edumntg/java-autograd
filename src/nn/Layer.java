package nn;

import engine.Value;

import java.util.ArrayList;
import java.util.List;

public class Layer extends Module {
    private Neuron[] neurons;
    public Layer(int inSize, int outSize) {
        neurons = new Neuron[outSize];
        for(int i = 0; i < outSize; i++) {
            neurons[i] = new Neuron(inSize);
        }
    }

    public Value[] forward(Value[] x) {
        // Input is (nSamples, ...)
        // Output should be (nSamples, this.neurons.length)
        Value[] out = new Value[this.neurons.length];
        for(int i = 0; i < this.neurons.length; i++) {
            out[i] = this.neurons[i].forward(x);
        }

        return out;
    }

    @Override
    public List<Value> parameters() {
        List<Value> out = new ArrayList<Value>();
        for(Neuron n: neurons) {
            for(Value p: n.parameters()) {
                out.add(p);
            }
        }

        return out;
    }

    @Override
    public String toString() {
        return String.format("Layer(%d neurons of size %d", neurons.length, neurons[0].getWeights().length);
    }
}
