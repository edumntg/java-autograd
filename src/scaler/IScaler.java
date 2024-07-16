package scaler;

import engine.Value;

public interface IScaler {
    public void fit(Value[] x);
    public void fit(Value[][] x);
    public void fitTransform(Value[] x);
    public void fitTransform(Value[][] x);
}
