import engine.Value;
import loss.MSELoss;
import nn.Layer;
import nn.MLP;
import tensor.Tensor;
//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws Exception {
        // Create a random dataset to fit the function y = 10*x + 20
        int N = 100; // number of samples

        int nFeatures = 1;
        Value[] x = new Value[N];
        Value[] y = new Value[N];
        // Fill x
        //float yval;
        for(int i = 0; i < N; i++) {
            //yval = 0;
            x[i] = Value.random();
//            for(int j = 0 ; j < nFeatures ; j++) {
//                x[i][j] = Value.random();
//                yval = 0.5f*x[i][j].data + 3.0f;
//            }
            //x[i] = new Value(i);
            y[i] = new Value(0.5f*x[i].data + 3.0f);
        }

        // Now, create a MLP with 2 layers
        MLP model = new MLP();
        model.add(new Layer(nFeatures, 32));
        model.add(new Layer(32, 100));
        //model.add(new Layer(64, 1));

        // Create loss
        MSELoss criterion = new MSELoss();
        Value loss = null;

        // Declare number of epochs
        int EPOCHS = 30;
        // Declare learning rate
        float lr = 1E-2f;

        Value[] scores = new Value[N];
        for(int epoch = 1; epoch <= EPOCHS; epoch++) {
            // Set all gradients to zero
            model.zeroGrad();

            // Compute scores
            scores = model.forward(x);

            // Compute loss
            loss = criterion.forwardV(y, scores);

            // Compute gradients
            loss.backward();

            // Update parameters
            for(Value v: model.parameters()) {
                v.optimize(lr);
            }

            // Print loss
            System.out.println(String.format("Epoch %d, loss: %s", epoch, loss.data));
        }

        // Predict
        Value[] yPred = model.forward(x);
        for(int i = 0; i < N; i++) {
            System.out.println(String.format("(yhat=%.4f, ytrue=%.4f)", yPred[i].data, y[i].data));
        }

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