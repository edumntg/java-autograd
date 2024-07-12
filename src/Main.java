import engine.Value;
import loss.MSELoss;
import nn.Layer;
import nn.MLP;
import optimizer.Optimizer;
import optimizer.SGD;
import tensor.Tensor;
//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws Exception {
        // Create a random dataset to fit the function y = 0.5*x1 -7.5*x2 + 12.5
        int N = 100; // number of samples
        int EPOCHS = 100; // number of epochs
        float lr = 1E-2f; // learning rate

        int nFeatures = 1;
        Value[][] x = new Value[N][nFeatures];
        Value[] y = new Value[N];
        // Fill x and y
        for(int i = 0; i < N; i++) {
            for(int j = 0 ; j < nFeatures ; j++) {
                x[i][j] = Value.random();
            }
            y[i] = new Value(x[i][0].mul(0.5f).add(20.0f).data);
        }

        // Now, create a MLP with 2 layers
        MLP model = new MLP();
        model.add(new Layer(nFeatures, 64));
        model.add(new Layer(64, 1));

        // Create loss
        MSELoss criterion = new MSELoss();
        Value loss = null;

        // Create optimizer
        Optimizer optimizer = new SGD(model.parameters(), lr, 0.05f);

        Value[] scores = new Value[N];
        for(int epoch = 1; epoch <= EPOCHS; epoch++) {
            // Set all gradients to zero
            optimizer.zeroGrad();

            // Compute scores
            scores = model.forward(x);

            // Compute loss
            loss = criterion.forwardV(y, scores);

            // Compute gradients
            loss.backward();

            // Update parameters
            optimizer.step();

            // Print loss
            System.out.println(String.format("Epoch %d, loss: %s", epoch, loss.data));
        }

//        // Predict
//        Value[] yPred = model.forward(x);
//        for(int i = 0; i < N; i++) {
//            System.out.println(String.format("(xi = %.4f, yhat=%.4f, ytrue=%.4f)", x[i][0].data, yPred[i].data, y[i].data));
//        }

        //System.out.println(loss.toString());


//        Value x = new Value(-4.0f);
//        Value z = x.mul(2.0f).add(x.add(2));
//        Value q = z.relu().add(z.mul(x));
//        Value h = z.mul(z).relu();
//        Value y = h.add(q).add(q.mul(x));
//        y.backward();
//        System.out.println(x.grad);
//        System.out.println(h.data);
    }
}