import optimizer.Optimizer;
import optimizer.SGD;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws Exception {
        //TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
        // to see how IntelliJ IDEA suggests fixing it.


        // Test optimizer
        List<Tensor> parameters = new ArrayList<Tensor>();
        for(int i = 0; i < 5; i++) {
            parameters.add(Tensor.random(3,3));
        }

        // Perform some basic operations so the gradients are not zero
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < 5; j++) {
                parameters.get(i).add(parameters.get(j));
            }
        }

        Optimizer sgd = new SGD(parameters, 0.1f);
        sgd.print();
        System.out.println("\n");
        // Update parameters
        sgd.step();
        sgd.print();

    }
}