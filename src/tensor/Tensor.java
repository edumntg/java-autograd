package tensor;

import java.util.Random;

public class Tensor {
    private int[] shape;
    private float[][] data;
    public float[][] grad;
    public Runnable gradFunction;
    private boolean empty;

    final static Random rng = new Random();

    // Create empty tensor
    public Tensor(int[] shape) {
        this.shape = shape;
        this.data = new float[shape[0]][shape[1]];
        this.empty = true;
    }

    public static Tensor Zeros(int[] shape) {
        Tensor tensor = new Tensor(shape);
        for(int i = 0; i < shape[0]; i++) {
            for(int j = 0; j < shape[1]; j++) {
                tensor.Set(i, j, 0.0f);
            }
        }

        return tensor;
    }

    public static Tensor Ones(int[] shape) {
        Tensor tensor = new Tensor(shape);
        for(int i = 0; i < shape[0]; i++) {
            for(int j = 0; j < shape[1]; j++) {
                tensor.Set(i, j, 1.0f);
            }
        }

        return tensor;
    }

    public static Tensor Diag(int[] shape) {
        Tensor tensor = new Tensor(shape);
        for(int i = 0; i < shape[0]; i++) {
            for(int j = 0; j < shape[1]; j++) {
                tensor.Set(i, j, 0.0f);
            }
            tensor.Set(i,i, 1.0f);
        }

        return tensor;
    }

    public void Set(int row, int column, float value) {
        this.data[row][column] = value;
        this.empty = false;
    }

    public int[] Shape() {
        return this.shape;
    }

    public float Get(int row, int column) {
        return this.data[row][column];
    }

    public Tensor T() {
        // Reshape the array
        Tensor out = new Tensor(new int[] {this.Shape()[1], this.Shape()[0]});
        for(int i = 0; i < this.Shape()[0]; i++) {
            for(int j = 0; j < this.Shape()[1]; j++) {
                out.Set(j, i, this.Get(i, j));
            }
        }

        return out;
    }

    public static Tensor Random(int[] shape) {
        Tensor tensor = new Tensor(shape);
        for(int i = 0; i < shape[0]; i++) {
            for(int j = 0; j < shape[1]; j++) {
                tensor.Set(i, j, rng.nextFloat());
            }
        }

        return tensor;
    }

    public void Print() {
        if(this.isEmpty()) {
            System.out.println("Tensor is empty");
            return;
        }

        System.out.println("Tensor with shape (" + this.Shape()[0] + ", " + this.Shape()[1] + ")");
        for(int i = 0; i < this.Shape()[0]; i++) {
            System.out.print("[");
            for(int j = 0; j < this.Shape()[1]; j++) {
                System.out.print(this.Get(i,j));
                if(j < this.Shape()[1]-1) {
                    System.out.print(", ");
                }
            }
            System.out.print("]\n");
        }
    }

    public boolean isEmpty() {
        return this.empty;
    }

    // Operations
    public Tensor Add(Tensor other) {
        // TODO: check shapes
        Tensor tensor = new Tensor(this.Shape());
        for(int i = 0; i < this.Shape()[0]; i++) {
            for(int j = 0; j < this.Shape()[1]; j++) {
                tensor.Set(i, j, this.Get(i, j) + other.Get(i, j));
            }
        }

        tensor.gradFunction = () -> {
            for(int i = 0; i < this.Shape()[0]; i++) {
                for(int j = 0; j < this.Shape()[1]; j++) {
                    this.grad[i][j] += tensor.Get(i,j);
                    other.grad[i][j] += tensor.Get(i,j);
                }
            }
        };

        return tensor;
    }

    public Tensor Sub(Tensor other) {
        // TODO: check shapes
        Tensor tensor = new Tensor(this.Shape());
        for(int i = 0; i < this.Shape()[0]; i++) {
            for(int j = 0; j < this.Shape()[1]; j++) {
                tensor.Set(i, j, this.Get(i, j) - other.Get(i, j));
            }
        }

        tensor.gradFunction = () -> {
            for(int i = 0; i < this.Shape()[0]; i++) {
                for(int j = 0; j < this.Shape()[1]; j++) {
                    this.grad[i][j] -= tensor.Get(i,j);
                    other.grad[i][j] -= tensor.Get(i,j);
                }
            }
        };

        return tensor;
    }

    public Tensor Mul(Tensor other) {
        // TODO: check shapes
        Tensor tensor = new Tensor(new int[] {this.Shape()[0], other.Shape()[1]});

        float total = 0.0f;
        for(int i = 0; i < this.Shape()[0]; i++) {
            for(int j = 0; j < this.Shape()[1]; j++) {
                for(int k = 0; k < this.Shape()[1]; k++) {
                    total += this.Get(i, k) + other.Get(k, j);
                }
                tensor.Set(i, j, total);
                total = 0.0f;
            }
        }

        return tensor;
    }
}