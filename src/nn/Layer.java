package nn;

import engine.Value;

import java.util.ArrayList;
import java.util.List;

public class Layer extends Module {

    public Value[][] W;
    public Value[] b;
    private boolean nonLin;

    public Layer(int inSize, int outSize) {
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

    public Layer(int inSize, int outSize, boolean nonLin) {
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

        this.nonLin = nonLin;
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

        // Apply matrix multiplication x*w + b
        for(int i = 0; i < x.length; i++) {
            for(int j = 0; j < W[0].length; j++) {
                for(int k = 0; k < x[0].length; k++) {
                    if(out[i][j] != null) {
                        out[i][j] = out[i][j].add(x[i][k].mul(W[k][j]));
                    } else {
                        out[i][j] = x[i][k].mul(W[k][j]);
                    }

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
