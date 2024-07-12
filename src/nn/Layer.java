package nn;

import engine.Value;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class Layer extends Module {
    private Neuron[] neurons;

    public Value[][] W;
    public Value[] b;
    private boolean nonLin;

    public Layer(int inSize, int outSize) {
//        neurons = new Neuron[outSize];
//        for(int i = 0; i < outSize; i++) {
//            neurons[i] = new Neuron(inSize);
//        }
        W = new Value[inSize][outSize];
        b = new Value[outSize];
        for (int i = 0; i < inSize; i++) {
            for (int j = 0; j < outSize; j++) {
                W[i][j] = Value.random();
            }
        }

        for(int i = 0; i < outSize; i++) {
            b[i] = new Value(0.0f);
        }

        nonLin = true;
    }

    public Value[][] forward(Value[][] x) {
        /**
         * x: (nSamples, nFeatures)
         * w: (nFeatures, nNeurons)
         * b: (nNeurons,)
         *
         * output: x*w + b = (nSamples, nNeurons)
         */
        // Apply matrix multiplication x*w + b
        Value[][] out = new Value[x.length][b.length]; // (nSamples, nNeurons)
        // Initialize
        for(int i = 0; i < x.length; i++) {
            for(int j = 0; j < b.length; j++) {
                out[i][j] = new Value(0.0f);
            }
        }

        // Apply matrix multiplication x*w + b
        for(int i = 0; i < x.length; i++) {
            for(int j = 0; j < W[0].length; j++) {
                for(int k = 0; k < W.length; k++) {
                    out[i][j] = out[i][j].add(x[i][k].mul(W[k][j]));
                }
                // Add bias
                out[i][j] = out[i][j].add(b[j]);

                if(nonLin) {
                    out[i][j] = out[i][j].relu();
                }
            }
        }

        return out;
    }

    @Override
    public List<Value> parameters() {
        List<Value> out = new ArrayList<Value>();
//        for(Neuron n: neurons) {
//            for(Value p: n.parameters()) {
//                out.add(p);
//            }
//        }

        for(int i = 0; i < this.W.length; i++) {
            for(int j = 0; j < this.W[0].length; j++) {
                out.add(this.W[i][j]);
            }
        }

        // Add bias
        for(int i = 0; i < this.b.length; i++) {
            out.add(this.b[i]);
        }

        return out;
    }

    @Override
    public String toString() {
        String name = "";
        if(nonLin) {
            name = "ReLU";
        }
        return String.format("%sLayer(shape=(%d,%d))", name, this.W.length, this.W[0].length);
    }
}
