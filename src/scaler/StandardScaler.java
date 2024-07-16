package scaler;

import engine.Value;

public class StandardScaler implements IScaler {
    private float singleMean;
    private float singleStd;
    private float[] mean;
    private float[] std;

    private boolean single;
    
    @Override
    public void fit(Value[] x) {
        // Compute std and mean
        float mean = 0.0f;
        float std = 0.0f;
        int N = x.length;

        // Mean
        for (Value item : x) {
            mean += item.item();
            std += (float) Math.pow(item.item() - mean, 2);
        }

        // Std
        for (Value value : x) {
            std += (float) Math.pow(value.item() - mean, 2);
        }
        mean /= N;
        std = (float) Math.sqrt(std/N);

        this.singleMean = mean;
        this.singleStd = std;
        this.single = true;
    }

    @Override
    public void fit(Value[][] x) {
        this.single = false;

        // Compute mean and std for each column in x
        this.mean = new float[x.length];
        this.std = new float[x.length];
        float columnMean = 0.0f;
        float columnStd = 0.0f;
        for(int j = 0; j < x[0].length; j++) {
            // Loop through columns
            for (Value[] value : x) {
                // Loop through each value in column j
                columnMean += value[j].item();
            }
            columnMean /= x.length;
            for (Value[] values : x) {
                // Loop through each value in column j
                columnStd += (float) Math.pow(values[j].item() - columnMean, 2);
            }
            columnStd = (float)Math.sqrt(columnStd / x.length);

            this.mean[j] = columnMean;
            this.std[j] = columnStd;
        }
    }

    @Override
    public void fitTransform(Value[] x) {
        // Fit
        this.fit(x);

        // Now, transform
        for(int i = 0; i < x.length; i++) {
            x[i] = x[i].sub(this.singleMean).div(this.singleStd);
        }
    }

    @Override
    public void fitTransform(Value[][] x) {
        // Fit
        this.fit(x);

        // Transform
        for(int i = 0; i < x.length; i++) {
            for(int j = 0; j < x[i].length; j++) {
                x[i][j] = x[i][j].sub(this.mean[j]).div(this.std[j]);
            }
        }
    }
}
