package nn;

import engine.Value;

public class Functional {

    public static Value relu(Value x) {
        return x.relu();
    }

    public static Value[] relu(Value[] x) {
        Value[] out = new Value[x.length];
        for(int i = 0; i < x.length; i++) {
            out[i] = Functional.relu(x[i]);
        }

        return out;
    }

    public static Value sigmoid(Value x) {
        return x.sigmoid();
    }

    public static Value[] sigmoid(Value[] x) {
        Value[] out = new Value[x.length];
        for(int i = 0; i < x.length; i++) {
            out[i] = Functional.sigmoid(x[i]);
        }

        return out;
    }

    public static Value[] softmax(Value[] x) {
        Value[] out = new Value[x.length];
        Value expSum = new Value(0.0f);
        for (Value value : x) {
            expSum = expSum.add(value.exp());
        }

        for(int i = 0; i < x.length; i++) {
            out[i] = x[i].exp().div(expSum);
        }

        return out;
    }
}
