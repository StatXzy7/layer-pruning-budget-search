package pruner;

/**
 * Point-wise activation function used by a feed-forward layer.
 */
public enum Activation {
    IDENTITY {
        @Override public double apply(double z) {
            return z;
        }
    },
    RELU {
        @Override public double apply(double z) {
            return z > 0.0 ? z : 0.0;
        }
    },
    TANH {
        @Override public double apply(double z) {
            return Math.tanh(z);
        }
    },
    SIGMOID {
        @Override public double apply(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }
    };

    public abstract double apply(double z);
}
