A Java Autograd engine that implements backpropagation over a dynamically built DAG. Based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd/tree/master)

This project was created for educational purposes

### Example usage
```java
MLP model = new MLP();
model.add(new Layer(2, 64));
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