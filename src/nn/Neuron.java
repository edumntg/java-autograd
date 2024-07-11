package nn;

import engine.Value;

import java.util.ArrayList;
import java.util.List;

public class Neuron extends Module {
    private Value[] w;
    private Value b;
    private boolean nonLin;

    public Neuron(int inSize, boolean nonLin) {
        this.w = new Value[inSize];
        for(int i = 0; i < inSize; i++) {
            this.w[i] = Value.random();
        }
        this.b = new Value(0.0f);
        this.nonLin = nonLin;
    }

    public Neuron(int in_size) {
        this.w = new Value[in_size];
        for(int i = 0; i < in_size; i++) {
            this.w[i] = Value.random();
        }
        this.b = new Value(0.0f);
        this.nonLin = true;
    }

    public Value forward(Value[] x) {
        Value output = (this.w[0].mul(x[0])).add(this.b);

        // x and this.w must have same length
        for(int i = 1; i < this.w.length; i++) {
            output = output.add((this.w[i].mul(x[i])).add(this.b)); // w*x + b
        }

        if(nonLin) {
            return output.relu();
        } else {
            return output;
        }
    }

    public Value[] getWeights() {
        return this.w;
    }

    @Override
    public List<Value> parameters() {
        List<Value> out = new ArrayList<Value>();
        for(int i = 0; i < w.length; i++) {
            out.add(w[i]);
        }
        out.add(b);

        return out;
    }
}
