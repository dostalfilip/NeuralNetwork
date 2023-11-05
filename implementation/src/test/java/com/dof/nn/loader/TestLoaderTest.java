package com.dof.nn.loader;

import com.dof.nn.loader.test.TestLoader;
import com.dof.nn.matrix.Matrix;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TestLoaderTest {

    @Test
    void test() {

        int batchSize = 33;

        Loader testLoader = new TestLoader(600, batchSize);

        MetaData metaData = testLoader.open();

        int numberItems = metaData.getNumberItems();

        int lastBatchSize = numberItems % batchSize;

        int numberBatches = metaData.getNumberBatches();

        for (int i = 0; i < numberBatches; i++) {
            BatchData batchData = testLoader.readBatch();

            assertNotNull(batchData);

            int itemsRead = metaData.getItemsRead();

            int inputSize = metaData.getInputSize();
            int expectedSize = metaData.getExpectedSize();

            Matrix input = new Matrix(inputSize, itemsRead, batchData.getInputBatch());
            Matrix expected = new Matrix(expectedSize, itemsRead, batchData.getExpectedBatch());

            assertNotEquals(0, input.sum());
            assertEquals(itemsRead, expected.sum());

            if (i == numberBatches - 1) {
                assertTrue(itemsRead == lastBatchSize);
            } else {
                assertTrue(itemsRead == batchSize);
            }
        }
    }

}
