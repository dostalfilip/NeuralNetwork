package com.dof.nn.core;

import com.dof.nn.matrix.Matrix;

import java.util.Random;

public class Util {

    private Util() {
    }

    public static TrainingMatrices generateTrainingMatrices(int inputRows, int outputRows, int cols) {
        Matrix input = new Matrix(inputRows, cols);
        Matrix output = new Matrix(outputRows, cols);

        for (int col = 0; col < cols; col++) {
            int radius = random.nextInt(outputRows);

            double[] values = new double[inputRows];

            double initialRadius = 0;
            for (int row = 0; row < inputRows; row++) {
                double value = random.nextGaussian();
                values[row] = value;
                initialRadius += value * value;
            }

            initialRadius = Math.sqrt(initialRadius);

            for (int row = 0; row < inputRows; row++) {
                input.set(row, col, values[row] / initialRadius);
            }

            output.set(radius, col, 1);
        }

        return new TrainingMatrices(input, output);
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
            int rowIndex = (int) (outputRows * (Math.sin(value) + 1.0) / 2.0);

            expected.set(rowIndex, col, 1);
        }));

        return expected;
    }
}
