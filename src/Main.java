import optimizer.Optimizer;
import optimizer.SGD;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws Exception {
        Tensor a = new Tensor(new float[][] {{2}});
        Tensor b = new Tensor(new float[][] {{3}});
        Tensor result = a.add(b);
        result.backward();
        a.gradient.print();
        b.gradient.print();
;
    }
}