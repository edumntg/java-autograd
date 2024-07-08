package tensor;

import exceptions.InvalidShapeException;

import java.util.Random;

public class Tensor {
    private int[] shape;
    private float[][] data;
    public float[][] grad;
    public Runnable gradFunction;
    private boolean empty;

    final static Random rng = new Random();

    // Create empty tensor
    public Tensor(int rows, int columns) {
        this.shape = new int[] {rows, columns};
        this.data = new float[rows][columns];
        this.empty = true;
    }

    public static Tensor Zeros(int rows, int columns) {
        Tensor tensor = new Tensor(rows, columns);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                tensor.Set(i, j, 0.0f);
            }
        }

        return tensor;
    }

    public static Tensor Ones(int rows, int columns) {
        Tensor tensor = new Tensor(rows, columns);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                tensor.Set(i, j, 1.0f);
            }
        }

        return tensor;
    }

    public static Tensor Identity(int size) {
        Tensor tensor = new Tensor(size, size);
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
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
        Tensor out = new Tensor(this.Shape()[1], this.Shape()[0]);
        for(int i = 0; i < this.Shape()[0]; i++) {
            for(int j = 0; j < this.Shape()[1]; j++) {
                out.Set(j, i, this.Get(i, j));
            }
        }

        return out;
    }

    public static Tensor Random(int rows, int columns) {
        Tensor tensor = new Tensor(rows, columns);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
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

    public boolean isSquare() {
        return this.Shape()[0] == this.Shape()[1];
    }

    // Operations
    public Tensor Add(Tensor other) {
        // TODO: check shapes
        Tensor tensor = new Tensor(other.Shape()[0], other.Shape()[1]);
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
        Tensor tensor = new Tensor(other.Shape()[0], other.Shape()[1]);
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
        Tensor tensor = new Tensor(this.Shape()[0], other.Shape()[1]);

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

    public Tensor LinMul(Tensor other) {
        // TODO: Check shapes
        // Perform linear multiplication: e.g: a[i,j]*b[i,j]
        Tensor tensor = new Tensor(this.Shape()[0], other.Shape()[1]);

        for(int i = 0; i < this.Shape()[0]; i++) {
            for(int j = 0; j < this.Shape()[1]; j++) {
                tensor.Set(i, j, this.Get(i, j) * other.Get(i, j));
            }
        }

        return tensor;
    }

    public Tensor ScalarMul(float k) {
        // Multiply each element in the tensor by scalar k
        Tensor tensor = new Tensor(this.Shape()[0], this.Shape()[1]);

        for(int i = 0; i < this.Shape()[0]; i++) {
            for(int j = 0; j < this.Shape()[1]; j++) {
                tensor.Set(i, j, this.Get(i, j) * k);
            }
        }

        return tensor;
    }

    public float Det() throws Exception {
        // Returns determinant of tensor (only if square)
        if(!this.isSquare()) {
            throw new InvalidShapeException("Tensor must be square");
        }

        if(this.Shape()[0] == 1) {
            // Single element tensor
            return this.Get(0,0);
        } else if(this.Shape()[1] == 2) {
            // 2x2 matrix, use the 2x2 matrix determinant formula
            return this.Get(0,0)*this.Get(1,1) - this.Get(1,0) * this.Get(0,1);
        } else {
            float sums = 0.0f;
            float retInDet = 1.0f;
            int size = this.Shape()[0];

            for(int colDet = 0; colDet < size; colDet++) {
                Tensor inner = Tensor.Zeros(size-1, size-1);
                for(int row = 1, rowInner = 0; row < size; row++) {
                    for(int col = 0, colInner = 0; col < size; col++) {
                        if(col == colDet) {
                            continue;
                        }

                        inner.Set(rowInner, colInner, this.Get(row, col));
                        colInner++;
                    }
                    rowInner++;
                }
                sums += retInDet * this.Get(0, colDet) * inner.Det();
                retInDet *= -1;
            }
            return sums;
        }
    }
}