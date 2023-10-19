package com.dof.nn.core;

import com.dof.nn.matrix.Matrix;

import java.util.function.Function;

public class Approximator {

    public static Matrix gradient(Matrix input, Function<Matrix, Matrix> transform) {

        final double INC = 0.000001;

        Matrix loss1 = transform.apply(input);

        assert loss1.getCols() == input.getCols() : "Input/loss columns not equal";
        assert loss1.getRows() == 1 : "Transform does not return one single row.";

        Matrix result = new Matrix(input.getRows(), input.getCols(), i -> 0);

        input.forEach((row, col, index, value) -> {
            Matrix incremented = input.addIncrement(row, col, INC);

            Matrix loss2 = transform.apply(incremented);

            double rate = (loss2.get(col) - loss1.get(col)) / INC;

            result.set(row, col, rate);
        });

        return result;
    }
}
