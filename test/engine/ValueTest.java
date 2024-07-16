package engine;

import org.junit.Test;

import static org.junit.Assert.*;

public class ValueTest {

    @Test
    public void add() {
    }

    @Test
    public void testOperations() throws Exception {
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
        g.backward();
        System.out.println(a.grad); // 138.8338 or dg/da
        System.out.println(b.grad); // 645.5773 or dg/db

        assertEquals(g.data, 24.7041, 1E-3);
        assertEquals(a.grad, 138.8338, 1E-3);
        assertEquals(b.grad, 645.5773, 1E-3);
    }
}