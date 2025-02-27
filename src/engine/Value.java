package engine;

import java.util.*;

public class Value {
    public float data;
    public float grad;
    public BackwardMethod _backward;
    public Set<Value> _prev;
    public boolean requiresGrad;

    public static final Random rng = new Random();

    public Value(float data, HashSet<Value> children) {
        this.data = data;
        this.grad = 0;
        this._backward = () -> { return; };
        this._prev = children;
        this.requiresGrad = true;
    }

    public Value(float data) {
        this.data = data;
        this.grad = 0;
        this._backward = () -> { return; };
        this._prev = new HashSet<Value>();
        this.requiresGrad = true;
    }

    public Value add(Value other) {
        Value out = new Value(this.data + other.data, new HashSet<Value>(Arrays.asList(this, other)));

        out._backward = () -> {
            if(!this.requiresGrad && !other.requiresGrad) {
                return;
            }
            this.grad += out.grad;
            other.grad += out.grad;
        };

        return out;
    }

    public Value add(float other) {
        Value otherValue = new Value(other, new HashSet<Value>());
        return this.add(otherValue);
    }

    public Value mul(Value other) {
        Value out = new Value(this.data * other.data, new HashSet<Value>(Arrays.asList(this, other)));
        out._backward = () -> {
            if(!this.requiresGrad && !other.requiresGrad) {
                return;
            }
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
        };

        return out;
    }

    public Value mul(float other) {
        Value otherValue = new Value(other, new HashSet<Value>());
        return this.mul(otherValue);
    }

    public Value pow(int other) {
        return this.pow((float)other);
    }

    public Value pow(float other) {
        Value out = new Value((float) Math.pow(this.data, other), new HashSet<Value>(List.of(this)));
        out._backward = () -> {
            if(!this.requiresGrad) {
                return;
            }
            this.grad += (float) ((other * Math.pow(this.data, other-1))*out.grad);
        };

        return out;
    }

    public Value neg() {
        return this.mul(-1.0f);
    }

    public Value sub(Value other) {
        return this.add(other.neg());
    }

    public Value sub(float other) {
        return this.add(-other);
    }

    public Value div(Value other) {
        return this.mul(other.pow(-1.0f));
    }

    public Value div(float other) {
        Value otherV = new Value(other, new HashSet<Value>());
        return this.div(otherV);
    }

    public Value inv() {
        return this.pow(-1.0f);
    }

    public Value exp() {
        Value out = new Value((float)Math.exp(this.data), new HashSet<Value>(List.of(this)));
        out._backward = () -> {
            if(!this.requiresGrad) {
                return;
            }

            this.grad += (float)Math.exp(this.data)*out.grad;

        };

        return out;
    }

    public Value abs() {
        Value out = new Value((float)Math.abs(this.data), new HashSet<Value>(List.of(this)));
        out._backward = () -> {
            if(!this.requiresGrad) {
                return;
            }
            float value = -1.0f;
            if(this.data > 0) {
                value = 1.0f;
            }

            this.grad += value*out.grad;
        };

        return out;
    }

    public Value log() {
        Value out = new Value((float)Math.log(this.data), new HashSet<Value>(List.of(this)));
        out._backward = () -> {
            if(!this.requiresGrad) {
                return;
            }
            this.grad += this.inv().data*out.grad;
        };

        return out;
    }

    public Value relu() {
        float current = this.data;
        if(current < 0) {
            current = 0.0f;
        }

        Value out = new Value(current, new HashSet<Value>(List.of(this)));
        out._backward = () -> {
            if(!this.requiresGrad) {
                return;
            }
            float value = 0.0f;
            if(out.data > 0.0f) {
                value = 1.0f;
            }
            this.grad += (value) * out.grad;
        };

        return out;
    }

    public Value sigmoid() {
        return this.neg().exp().add(1.0f).inv();
    }

    public void backward() throws Exception {
        List<Value> topo = new ArrayList<Value>();
        Set<Value> visited = new HashSet<Value>();

        buildTopo(this, topo, visited);
        this.grad = 1.0f;

        for(int i = topo.size() - 1; i >=0; i--) {
            topo.get(i)._backward.run();
        }
    }

    private void buildTopo(Value v, List<Value> topo, Set<Value> visited) {
        if(!visited.contains(v)) {
            visited.add(v);
            for(Value child: v._prev) {
                buildTopo(child, topo, visited);
            }
            topo.add(v);
        }
    }

    public void optimize(float lr) {
        this.data -= lr*this.grad;
    }

    public static Value random() {
        return new Value(Value.rng.nextFloat()*(1 + 1) - 1);
    }

    public static Value[] flatten(Value[][] arr) {
        Value[] out = new Value[arr.length];
        for(int i = 0; i < arr.length; i++) {
            out[i] = arr[i][0];
        }

        return out;
    }

    public float item() {
        return this.data;
    }

    public Value detach() {
        // Returns a new Value with same data but detached from graph
        Value out = new Value(this.data);
        out.requiresGrad = false;
        return out;
    }

    @Override
    public String toString() {
        return "Value(data=" + this.data + ", grad=" + this.grad + ", children=" + this._prev.toString() + ")";
    }
}
