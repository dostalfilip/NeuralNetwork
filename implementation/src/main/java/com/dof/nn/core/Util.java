package com.dof.nn.core;

import com.dof.nn.matrix.Matrix;

import java.util.Random;

public class Util {

    private Util() {
    }

    private static Random random = new Random();

    public static Matrix generateInputMatrix(int rows, int cols) {
        return new Matrix(rows, cols, index -> (random.nextGaussian()));
    }

    public static Matrix generateExpectedMatrix(int rows, int cols) {
        Matrix expected = new Matrix(rows, cols, index -> 0);

        for (int col = 0; col < expected.getCols(); col++) {
            int randomRow = random.nextInt(rows);

            expected.set(randomRow, col, 1);
        }
        return expected;
    }

    public static Matrix generateTrainableExpectedMatrix(int outputRows, Matrix input) {
        Matrix expected = new Matrix(outputRows, input.getCols());

        Matrix columnSums = input.sumColumns();

        columnSums.forEach(((row, col, index, value) -> {
            int rowIndex = (int) ( outputRows * (Math.sin(value) + 1.0) / 2.0);

            expected.set(rowIndex, col, 1);
        }));

            System.out.println(expected);
        return expected;
    }
}
