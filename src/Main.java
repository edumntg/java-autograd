import tensor.Tensor;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {
        //TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
        // to see how IntelliJ IDEA suggests fixing it.
        Tensor tensor = Tensor.Zeros(new int[] {5,5});
        tensor.Print();

        Tensor tensor2 = Tensor.Random(new int[] {3, 7});
        tensor2.Print();
    }
}