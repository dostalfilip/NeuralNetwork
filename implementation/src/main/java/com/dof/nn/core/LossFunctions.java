package com.dof.nn.core;

import com.dof.nn.matrix.Matrix;

public class LossFunctions {
    private LossFunctions() {
    }
    public static Matrix crossEntropy(Matrix expected, Matrix actual) {
        return actual.apply((index, value) -> -expected.get(index) * Math.log(value)).sumColumns();
    }
}
