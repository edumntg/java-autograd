package utils;

import java.util.Arrays;

public class Utils {
    public static boolean ShapesEquals(int[] shape1, int[] shape2) {
        return Arrays.equals(shape1, shape2);
    }

    public static boolean ShapesMultiplicative(int[] shape1, int[] shape2) {
        return shape1[1] == shape2[0];
    }
}
