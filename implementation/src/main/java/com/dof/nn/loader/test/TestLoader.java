package com.dof.nn.loader.test;

import com.dof.nn.core.Util;
import com.dof.nn.loader.BatchData;
import com.dof.nn.loader.Loader;
import com.dof.nn.loader.MetaData;

public class TestLoader implements Loader {
    private MetaData metaData;

    private int numberItems = 60_000;
    private int inputSize = 500;
    private int expectedSize = 3;
    private int numberBatches;
    private int batchSize = 32;

    private int totalItemsRead;
    private int itemsRead;

    public TestLoader(MetaData metaData) {
        this.metaData = metaData;
        metaData.setNumberItems(numberItems);
        numberBatches = numberItems / batchSize;
        if (numberBatches % batchSize != 0) {
            numberBatches += 1;
        }
        metaData.setNumberBatches(numberBatches);
        metaData.setInputSize(inputSize);
        metaData.setExpectedSize(expectedSize);

    }

    @Override
    public MetaData open() {
        return metaData;
    }

    @Override
    public void close() {
    }

    @Override
    public MetaData getMetaData() {
        return metaData;
    }

    @Override
    public BatchData readBatch() {
        if (totalItemsRead == numberItems) {
            return null;
        }

        itemsRead = batchSize;
        totalItemsRead += itemsRead;

        int excessItems = totalItemsRead - numberItems;
        if (excessItems > 0) {
            totalItemsRead -= excessItems;
        }

        var io = Util.generateTrainingArrays(inputSize, expectedSize, itemsRead);

        var batchData = new TestBatchData();
        batchData.setInputBatch(io.getInput());
        batchData.setExpectedBatch(io.getOutput());

        metaData.setTotalItemsRead(totalItemsRead);
        metaData.setItemsRead(itemsRead);

        return batchData;
    }
}
