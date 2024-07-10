package tensor;

import org.junit.Test;

import static org.junit.Assert.*;

public class TensorTest {

    @Test
    public void add() throws Exception {
        Tensor a = new Tensor(new float[][] {{2}});
        Tensor b = new Tensor(new float[][] {{3}});
        Tensor result = a.add(b);
        assertEquals(5.0f, result.item(), 0.0);
    }

    @Test
    public void item() {
        Tensor a = new Tensor(new float[][] {{2}});

        assertEquals(a.item(), 2.0f, 0.0);
    }

    @Test
    public void t() {
    }

    @Test
    public void sub() throws Exception {
        Tensor a = new Tensor(new float[][] {{2}});
        Tensor b = new Tensor(new float[][] {{3}});
        Tensor result = a.sub(b);
        assertEquals(-1.0f, result.item(), 0.0);
    }

    @Test
    public void mul() throws Exception {
        Tensor a = new Tensor(new float[][] {{1,2,3}, {4,5,6}}); // 2x3
        Tensor b = new Tensor(new float[][] {{7,8}, {9,10}, {11,12}}); // 3x2

        Tensor expected = new Tensor(new float[][] {{58, 64}, {139, 154}});

        Tensor result = a.mul(b);
        float error = result.sub(expected).sum().item();

        assertEquals(error, 0.0f, 0.0);
    }

    @Test
    public void pow() {
    }

    @Test
    public void sum() {
    }
}