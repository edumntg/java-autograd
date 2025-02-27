import engine.Value;
import loss.MSELoss;
import nn.Layer;
import nn.MLP;
import optimizer.Optimizer;
import optimizer.SGD;
import scaler.StandardScaler;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws Exception {
        // Create a random dataset to fit the function y = 0.5*x1 -7.5*x2 + 12.5
        int N = 100; // number of samples
        int EPOCHS = 100; // number of epochs
        float lr = 1E-3f; // learning rate

        int nFeatures = 8;
        Value[][] x = new Value[N][nFeatures];
        Value[] y = new Value[N];
        // Fill x and y
        for(int i = 0; i < N; i++) {
            for(int j = 0 ; j < nFeatures ; j++) {
                x[i][j] = Value.random();
                x[i][j].requiresGrad = false;
            }
            y[i] = new Value((x[i][0].mul(0.5f).add(x[i][1].mul(-7.5f)).add(20.0f)).data);
            y[i].requiresGrad = false;
        }

        StandardScaler scaler = new StandardScaler();
        scaler.fitTransform(x);
        scaler.fitTransform(y);

        // Now, create MLP with 2 layers
        MLP model = new MLP();
        model.add(new Layer(nFeatures, 128)); // input layer
        model.add(new Layer(128, 64));
        model.add(new Layer(4, 16));
        model.add(new Layer(16, 1)); // output

        // Create loss
        MSELoss criterion = new MSELoss();
        Value loss = null;

        // Create optimizer
        Optimizer optimizer = new SGD(model.parameters(), lr, 0.01f);
        optimizer.scheduler = () -> {
            //optimizer.lr *= 0.99f;
        };

        Value[] scores = new Value[N];
        long start, end, elapsed;
        long[] timesPerEpoch = new long[EPOCHS];
        for(int epoch = 1; epoch <= EPOCHS; epoch++) {
            start = System.currentTimeMillis();
            // Set all gradients to zero
            optimizer.zeroGrad();

            // Compute scores
            scores = model.forward(x);

            // Compute loss
            loss = criterion.forward(y, scores);

            // Compute gradients
            loss.backward();

            // Update parameters
            optimizer.step();

            end = System.currentTimeMillis();
            elapsed = end - start;
            timesPerEpoch[epoch-1] = elapsed;

            // Print loss
            System.out.printf("Epoch %d, loss: %s, lr = %.10f, time = %o (ms)%n", epoch, loss.data, optimizer.lr, elapsed);
           //System.out.println(loss.toString());
        }

        float avg = 0.0f;
        for(int i = 0; i < EPOCHS; i++) {
            avg += (float)timesPerEpoch[i];
        }

        avg /= (float)EPOCHS;
        System.out.printf("\nAvg. time per epoch: %.4f (ms)%n", avg);
    }
}