package tensor;

import enums.Operator;
import exceptions.InvalidShapeException;
import utils.Utils;

import java.util.Random;

public class Tensor {
    private int[] shape;
    private float[][] data;
    public Tensor gradient;
    private boolean empty;
    public boolean requiresGrad;

    public TensorOperation dependencies;

    final static Random rng = new Random();

//    private static float[][] initializeGradients(int rows, int columns) {
//        float[][] grad = new float[rows][columns];
//        for(int i = 0; i < rows; i++) {
//            for(int j = 0; j < columns; j++) {
//                //grad[i][j] = 0.0f;
//                grad[i][j] = rng.nextFloat();
//            }
//        }
//
//        return grad;
//    }

    public Tensor(int rows, int columns, boolean requiresGrad) {
        this.shape = new int[] {rows, columns};
        this.data = new float[rows][columns];
        this.requiresGrad = requiresGrad;
        if(requiresGrad) {
            this.gradient = new Tensor(rows, columns, false);
        }

        this.dependencies = new TensorOperation(Operator.NONE, null, null);
    }

    // Create empty tensor
    public Tensor(int rows, int columns) {
        this.shape = new int[] {rows, columns};
        this.data = new float[rows][columns];
        this.empty = true;

        // Initialize gradients
        this.gradient = new Tensor(rows, columns, false);
        this.gradient.zero_();

        //this.gradient = initializeGradients(rows, columns);
        this.requiresGrad = true;
        this.dependencies = new TensorOperation(Operator.NONE, null,null);
    }

    public Tensor(float[][] data) {
        // Create from a 2D array
        this.data = data;
        this.shape = new int[] {data.length, data[0].length};
        this.requiresGrad = true;
        this.gradient = new Tensor(this.shape[0], this.shape[1], false);
        this.gradient.zero_();
        this.dependencies = new TensorOperation(Operator.NONE, null, null);
    }

    public static Tensor zeros(int rows, int columns) {
        Tensor tensor = new Tensor(rows, columns);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                tensor.set(i, j, 0.0f);
            }
        }

        return tensor;
    }

    public static Tensor ones(int rows, int columns) {
        Tensor tensor = new Tensor(rows, columns);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                tensor.set(i, j, 1.0f);
            }
        }

        return tensor;
    }

    public static Tensor identity(int size) {
        Tensor tensor = new Tensor(size, size);
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                tensor.set(i, j, 0.0f);
            }
            tensor.set(i,i, 1.0f);
        }

        return tensor;
    }

    public void set(int row, int column, float value) {
        this.data[row][column] = value;
        this.empty = false;
    }

    public int[] shape() {
        return this.shape;
    }

    public float get(int row, int column) {
        return this.data[row][column];
    }

    public Tensor T() {
        // Reshape the array
        Tensor out = new Tensor(this.shape()[1], this.shape()[0]);
        for(int i = 0; i < this.shape()[0]; i++) {
            for(int j = 0; j < this.shape()[1]; j++) {
                out.set(j, i, this.get(i, j));
            }
        }

        return out;
    }

    public static Tensor random(int rows, int columns) {
        Tensor tensor = new Tensor(rows, columns);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                tensor.set(i, j, rng.nextFloat());
            }
        }

        return tensor;
    }

    public void print() {
        if(this.isEmpty()) {
            System.out.println("Tensor is empty");
            return;
        }

        System.out.println("Tensor with shape (" + this.shape()[0] + ", " + this.shape()[1] + ", requires_grad=" + this.requiresGrad + ")");
        for(int i = 0; i < this.shape()[0]; i++) {
            System.out.print("[");
            for(int j = 0; j < this.shape()[1]; j++) {
                System.out.print(this.get(i,j));
                if(j < this.shape()[1]-1) {
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
        return this.shape()[0] == this.shape()[1];
    }

    public void zero_() {
        // Fill tensor with zeros
        for(int i = 0; i < this.shape()[0]; i++) {
            for(int j = 0; j < this.shape()[1]; j++) {
                this.set(i, j, 0.0f);
            }
        }
    }

    // Operations
    public Tensor add(Tensor other) throws Exception {
        // Check shapes
        if(!Utils.ShapesEquals(this.shape(), other.shape())) {
            throw new InvalidShapeException("Tensors A and B must have equal shapes");
        }
        Tensor tensor = new Tensor(other.shape()[0], other.shape()[1]);
        for(int i = 0; i < this.shape()[0]; i++) {
            for(int j = 0; j < this.shape()[1]; j++) {
                tensor.set(i, j, this.get(i, j) + other.get(i, j));
            }
        }

//        tensor.gradFunction = () -> {
//            if(!this.requiresGrad) {
//                return;
//            }
//            System.out.println("add grad func");
//            for(int i = 0; i < this.shape()[0]; i++) {
//                for(int j = 0; j < this.shape()[1]; j++) {
//                    this.gradient[i][j] += tensor.get(i,j);
//                    other.gradient[i][j] += tensor.get(i,j);
//                }
//            }
//        };

        tensor.dependencies = new TensorOperation(Operator.ADD, this, other);
        //this.dependencies = new TensorOperation(Operator.ADD, this, other);
        //other.dependencies = new TensorOperation(Operator.ADD, this, other);


        return tensor;
    }

    public void backward() {
        if(!this.requiresGrad) {
            return;
        }

        if(this.dependencies.operation == Operator.NONE) {
            return;
        }

        Tensor a = this.dependencies.actors[0];
        Tensor b = this.dependencies.actors[1];
        switch(this.dependencies.operation) {
            case ADD:
            case MUL:
                for(int i = 0; i < a.shape()[0]; i++) {
                    for(int j = 0; j < a.shape()[1]; j++) {
                        a.gradient.set(i,j, a.gradient.get(i,j) + b.get(i,j));
                        b.gradient.set(i,j, b.gradient.get(i,j) + a.get(i,j));
                    }
                }
                break;

            case SUB:
                for(int i = 0; i < a.shape()[0]; i++) {
                    for(int j = 0; j < a.shape()[1]; j++) {
                        a.gradient.set(i,j, a.gradient.get(i,j) + b.get(i,j));
                        b.gradient.set(i,j, b.gradient.get(i,j) - a.get(i,j));
                    }
                }
                break;

            case DIV:
                for(int i = 0; i < a.shape()[0]; i++) {
                    for(int j = 0; j < a.shape()[1]; j++) {
                        a.gradient.set(i,j, (float) (a.gradient.get(i,j) + 1.0/b.get(i,j)));
                        b.gradient.set(i,j, (float) (b.gradient.get(i,j) - a.get(i,j)/Math.pow(b.get(i,j), 2.0)));
                    }
                }
                break;

        }
    }

    public Tensor sub(Tensor other) throws Exception {
        // Check shapes
        if(!Utils.ShapesEquals(this.shape(), other.shape())) {
            throw new InvalidShapeException("Tensors A and B must have equal shapes");
        }
        Tensor tensor = new Tensor(other.shape()[0], other.shape()[1]);
        for(int i = 0; i < this.shape()[0]; i++) {
            for(int j = 0; j < this.shape()[1]; j++) {
                tensor.set(i, j, this.get(i, j) - other.get(i, j));
            }
        }

//        tensor.gradFunction = () -> {
//            if(!this.requiresGrad) {
//                return;
//            }
//            for(int i = 0; i < this.shape()[0]; i++) {
//                for(int j = 0; j < this.shape()[1]; j++) {
//                    this.gradient[i][j] -= tensor.get(i,j);
//                    other.gradient[i][j] -= tensor.get(i,j);
//                }
//            }
//        };

        tensor.dependencies = new TensorOperation(Operator.SUB, this, other);

        return tensor;
    }

    public Tensor mul(Tensor other) throws Exception {
        // Check shapes
        if(!Utils.ShapesMultiplicative(this.shape(), other.shape())) {
            throw new InvalidShapeException("Number of columns in Tensor A must be equal to number of rows in Tensor B");
        }
        Tensor tensor = new Tensor(this.shape()[0], other.shape()[1]);

        float total = 0.0f;
        for(int i = 0; i < this.shape()[0]; i++) {
            for(int j = 0; j < this.shape()[1]; j++) {
                for(int k = 0; k < this.shape()[1]; k++) {
                    total += this.get(i, k) + other.get(k, j);
                }
                tensor.set(i, j, total);
                total = 0.0f;
            }
        }

        tensor.dependencies = new TensorOperation(Operator.MUL, this, other);

        return tensor;
    }

    public Tensor linMul(Tensor other) throws Exception {
        // Check shapes
        if(!Utils.ShapesEquals(this.shape(), other.shape())) {
            throw new InvalidShapeException("Tensors A and B must have equal shapes");
        }
        // Perform linear multiplication: e.g: a[i,j]*b[i,j]
        Tensor tensor = new Tensor(this.shape()[0], other.shape()[1]);

        for(int i = 0; i < this.shape()[0]; i++) {
            for(int j = 0; j < this.shape()[1]; j++) {
                tensor.set(i, j, this.get(i, j) * other.get(i, j));
            }
        }

        return tensor;
    }

    public Tensor scalarMul(float k) {
        // Multiply each element in the tensor by scalar k
        Tensor tensor = new Tensor(this.shape()[0], this.shape()[1]);

        for(int i = 0; i < this.shape()[0]; i++) {
            for(int j = 0; j < this.shape()[1]; j++) {
                tensor.set(i, j, this.get(i, j) * k);
            }
        }

        return tensor;
    }

    public float det() throws Exception {
        // Returns determinant of tensor (only if square)
        if(!this.isSquare()) {
            throw new InvalidShapeException("Tensor must be square");
        }

        if(this.shape()[0] == 1) {
            // Single element tensor
            return this.get(0,0);
        } else if(this.shape()[1] == 2) {
            // 2x2 matrix, use the 2x2 matrix determinant formula
            return this.get(0,0)*this.get(1,1) - this.get(1,0) * this.get(0,1);
        } else {
            float sums = 0.0f;
            float retInDet = 1.0f;
            int size = this.shape()[0];

            for(int colDet = 0; colDet < size; colDet++) {
                Tensor inner = Tensor.zeros(size-1, size-1);
                for(int row = 1, rowInner = 0; row < size; row++) {
                    for(int col = 0, colInner = 0; col < size; col++) {
                        if(col == colDet) {
                            continue;
                        }

                        inner.set(rowInner, colInner, this.get(row, col));
                        colInner++;
                    }
                    rowInner++;
                }
                sums += retInDet * this.get(0, colDet) * inner.det();
                retInDet *= -1;
            }
            return sums;
        }
    }

    public void update(float learningRate) {
        // Update tensor gradients
        this.backward();
        // Updates its value given a learningRate
        for(int i = 0; i < this.shape()[0]; i++) {
            for(int j = 0; j < this.shape()[1]; j++) {
                this.set(i, j, this.get(i, j) + learningRate*this.gradient.get(i, j));
            }
        }
    }

}