package com.dof.nn;

public class LossFunction {
    private LossFunction() {
    }

    public static Matrix crossEntropy(Matrix expected, Matrix actual) {
        return actual.apply((index, value) -> -expected.get(index) * Math.log(value)).sumColumns();
    }
}
