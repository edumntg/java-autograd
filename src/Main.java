import tensor.Tensor;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws Exception {
        //TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
        // to see how IntelliJ IDEA suggests fixing it.
        Tensor tensor = Tensor.Zeros(5,5);
        tensor.Print();

        Tensor tensor2 = Tensor.Random(3,7);
        tensor2.Print();

        // Test transpose
        Tensor tensor3 = Tensor.Random(3,7);
        tensor3.Print();
        Tensor tensor3T = tensor3.T();
        tensor3T.Print();

        // Test identity
        Tensor I = Tensor.Identity(4);
        I.Print();

        System.out.println("");
        System.out.print(I.Det());
    }
}