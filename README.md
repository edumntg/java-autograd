A Java Autograd engine that implements backpropagation over a dynamically built DAG. Based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd/tree/master)

This project was created for educational purposes

### Example usage
```java
MLP model = new MLP();
model.add(new Layer(2, 64)); // 2 input features
model.add(new Layer(64, 32));
model.add(new Layer(32, 1)); // 1 output

// Create loss function
Loss criterion = new MSELoss();

// Create optimizer
Optimizer optimizer = new SGD(model.parameters(), 1.0E-3f); // (parameters, learningRate)

Value[] scores = model.forward(x); // x is a Value[][] array

// Compute loss
Value loss = criterion.forward(scores, y);

// Perform backward operation to compute all gradients
loss.backward();

// Update parameters
optimizer.step();
```

If you want to add an scheduler to the optimizer, you can do it in the following way:
```java
optimizer.scheduler = () -> {
    // multiply learning rate by 0.99 on each step
    // scheduler is executed BEFORE parameter optimization
    optimizer.lr *= 0.99f;
};
```

--------------
You can also perform operations and compute gradients
```java
Value a = new Value(-4.0f);
Value b = new Value(2.0f);
Value c = a.add(b); // a + b
Value d = a.mul(b).add(b.pow(3)); // a * b + b^3
c = c.add(c.add(1)); // c += c + 1
c = c.add(c.add(1).add(a.neg())); // 1 + c + (-a)
d = d.add(d.mul(2).add(b.add(a).relu())); // d+= 2*d + (b+a).relu()
d = d.add(d.mul(3).add(b.sub(a).relu())); // 3*d + (b-a).relu()
Value e = c.sub(d); // c - d
Value f = e.pow(2); // e^2
Value g = f.div(2); // f / 2
g = g.add(f.pow(-1).mul(10)); // g + 10 / f
System.out.println(g.data); // prints 24.7041

// Compute gradients
g.backward();

System.out.println(a.grad); // 138.8338 or dg/da
System.out.println(b.grad); // 645.5773 or dg/db
```

--------------
### Housing Dataset Demo
```java
// Set seed
Value.rng.setSeed(1218);

// Read dataset
List<List<String>> records = new ArrayList<>();
Value[][] x = new Value[5000][5]; // 5000 samples, 5 features
Value[] y = new Value[5000];

// Read dataset
try(BufferedReader br = new BufferedReader(new FileReader("housing_dataset.csv"))) {
    String line;
    int i = 0;
    while((line = br.readLine()) != null) {
        String[] values = line.split(",");
        for(int j = 0; j < 5; j++) {
            x[i][j] = new Value(Float.parseFloat(values[j]));
        }
        y[i] = new Value(Float.parseFloat(values[5]));
        i++;
    }
} catch(Exception ex) {
    System.out.println(ex.getMessage());
    return;
}

// Scale dataset
StandardScaler scaler = new StandardScaler();
scaler.fitTransform(x);
scaler.fitTransform(y);

// Create model with one hidden layer
MLP model = new MLP();
model.add(new Layer(5, 16)); // input layer
//model.add(new Layer(64, 32)); // hidden
model.add(new Layer(16, 1)); // output

// Criterion and optimizer
Optimizer optimizer = new SGD(model.parameters(), 1E-2f, 0.05f);
Loss criterion = new MSELoss();

// Train
Value[] scores;
Value loss = null;
for(int epoch = 1; epoch <= 100; epoch++) {
    optimizer.zeroGrad(); // reset all gradients;

    // Compute scores
    scores = model.forward(x);

    // Compute loss
    loss = criterion.forward(y, scores);

    // Compute gradients
    loss.backward();

    // Update parameters
    optimizer.step();

    // Print loss
    System.out.println(String.format("Epoch %d, loss = %.10f", epoch, loss.item()));
    //System.out.println(loss.toString());
}
```