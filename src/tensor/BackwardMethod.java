package tensor;

@FunctionalInterface
public interface BackwardMethod {
    public void run() throws Exception;
}
