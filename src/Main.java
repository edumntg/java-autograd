import engine.Value;
import loss.MSELoss;
import tensor.Tensor;
//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws Exception {
        Value x = new Value(-4.0f);
        Value z = x.mul(2.0f).add(x.add(2));
        Value q = z.relu().add(z.mul(x));
        Value h = z.mul(z).relu();
        Value y = h.add(q).add(q.mul(x));
        y.backward();
        System.out.println(x.grad);
        System.out.println(h.data);
    }
}