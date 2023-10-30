package com.dof.nn.matrix;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;


class MatrixTest {
    private Random random = new Random();

    @Test
    void testGetGreatestRowNumber() {
        double[] values = {2,-6,7,7,2,-6,11,-1,1};
        Matrix m = new Matrix(3,3, i -> values[i]);

        Matrix result = m.getGreatestRowNumber();

        double[] expectedValues = {2,1,0};
        Matrix matrixExpected = new Matrix(1, 3, i -> expectedValues[i]);

        assertEquals(matrixExpected, result);
    }

    @Test
    void testAverageColumn() {
        int rows = 7;
        int cols = 5;

        Matrix m = new Matrix(rows, cols, i -> 2 * i - 3);
        double averageIndex = (cols - 1) / 2.0;

        Matrix expected = new Matrix(rows, 1);
        expected.modify((row, col, value) -> 2 * (row * cols + averageIndex) - 3);

        Matrix result = m.averageColumn();

        assertEquals(expected, result);
    }

    @Test
    void testTranspose() {
        Matrix m = new Matrix(2, 3, i -> i);
        Matrix result = m.transpose();

        double[] expectedValues = {0, 3, 1, 4, 2, 5};
        Matrix expected = new Matrix(3, 2, i -> expectedValues[i]);

        assertTrue(expected.equals(result));
    }

    @Test
    void testAddIncrement() {
        Matrix m = new Matrix(5, 8, i -> random.nextGaussian());

        int row = 3;
        int col = 2;
        double inc = 10;

        Matrix result = m.addIncrement(row, col, inc);

        double incrementValue = result.get(row, col);
        double originalValue = m.get(row, col);

        assertTrue(Math.abs(incrementValue - originalValue - inc) < 0.00001);
    }

    @Test
    void testSoftMax() {
        Matrix m = new Matrix(5, 8, i -> random.nextGaussian());

        Matrix result = m.softMax();

        double[] colSum = new double[8];

        result.forEach((row, col, value) -> {
            assertTrue(value >= 0 && value <= 1);
            colSum[col] += value;
        });

        for (var sum : colSum) {
            assertTrue(Math.abs(sum - 1.0) < 0.00001);
        }
    }

    @Test
    void testSumColumns() {
        Matrix m = new Matrix(4, 5, i -> i);
        Matrix result = m.sumColumns();

        double[] expectedValues = {+30.00000, +34.00000, +38.00000, +42.00000, +46.00000};
        Matrix expected = new Matrix(1, 5, i -> expectedValues[i]);

        assertEquals(expected, result);
    }

    @Test
    void testMultiply() {
        Matrix m1 = new Matrix(2, 3, i -> i);
        Matrix m2 = new Matrix(3, 2, i -> i);

        double[] expectedValues = {10, 13, 28, 40};
        Matrix expected = new Matrix(2, 2, i -> expectedValues[i]);

        Matrix result = m1.multiply(m2);

        assertEquals(result, expected);
    }

    @Test
    void testMultiplySpeed() {
        int rows = 1000;
        int cols = rows;
        int mid = 50;

        Matrix m1 = new Matrix(rows, mid, i -> i);
        Matrix m2 = new Matrix(mid, cols, i -> i);

        var start = System.currentTimeMillis();
        Matrix result = m1.multiply(m2);
        var end = System.currentTimeMillis();

        System.out.printf("Matrix multilication time taken: %dms\n", end - start);
        assertTrue(end - start < 50);
    }

    @Test
    void testEquals() {
        Matrix m1 = new Matrix(3, 4, i -> 2 * (i - 6));
        Matrix m2 = new Matrix(3, 4, i -> 2 * (i - 6));
        Matrix m3 = new Matrix(3, 4, i -> 2 * (i - 6.2));

        assertEquals(m1, m2);
        assertNotEquals(m1, m3);
    }

    @Test
    void testMultiplyDouble() {
        Matrix m = new Matrix(3, 4, i -> 2.5 * (i - 6));

        double x = 0.5;
        Matrix expected = new Matrix(3, 4, i -> x * 2.5 * (i - 6));

        Matrix result = m.apply((index, value) -> x * value);

        assertTrue(Math.abs(expected.get(0) + 7.50000) < 0.0001);
        assertEquals(expected, result);
    }

    @Test
    void testAddMatrices() {
        Matrix m1 = new Matrix(2, 2, i -> i);
        Matrix m2 = new Matrix(2, 2, i -> i * 1.5);
        Matrix expected = new Matrix(2, 2, i -> i * 2.5);

        Matrix result = m1.apply((index, value) -> value + m2.get(index));

        assertEquals(result, expected);
    }

    @Test
    void testToString() {
        Matrix matrix = new Matrix(3, 4, i -> i * 2);
        String result = matrix.toString();

        System.out.println(result);

        int index = 0;
        double[] expected = new double[12];

        for (int i = 0; i < expected.length; i++) {
            expected[i] = i * 2;
        }

        var rawRows = result.split("\n");

        assertEquals(3, rawRows.length);

        for (var rawRow : rawRows) {
            var rawCols = rawRow.split("\\s+");

            for (var rawCol : rawCols) {

                if (rawCol.length() == 0) {
                    continue;
                }

                assertTrue(Math.abs(expected[index] - Double.valueOf(rawCol)) < 0.00001);

                index++;
            }

        }
    }

}
