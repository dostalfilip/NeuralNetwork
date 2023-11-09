package com.dof.nn.loader.image;

import com.dof.nn.loader.BatchData;
import com.dof.nn.loader.Loader;
import com.dof.nn.loader.MetaData;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;

public class ImageLoader implements Loader {
    private Path imageFileName;
    private Path labelFileName;
    private int batchSize;

    private DataInputStream dsImages;
    private DataInputStream dsLabels;

    public ImageLoader(Path imageFileName, Path labelFileName, int batchSize) {
        this.imageFileName = imageFileName;
        this.labelFileName = labelFileName;
        this.batchSize = batchSize;
    }

    @Override
    public MetaData open() {
        try {
            dsImages = new DataInputStream(new FileInputStream(imageFileName.toFile()));
        } catch (Exception e) {
            throw new LoaderException("Cannot open " + imageFileName, e);
        }

        try {
            dsLabels = new DataInputStream(new FileInputStream(labelFileName.toFile()));
        } catch (Exception e) {
            throw new LoaderException("Cannot open " + labelFileName, e);
        }
        readMetaData();
        return null;
    }

    private MetaData readMetaData() {

        int numberItems = 0;

        try {
            int magicLabelNumber = dsLabels.readInt();
            if (magicLabelNumber != 2049) {
                throw new LoaderException("Label file: " + labelFileName + " has wrong format.");
            }
            numberItems = dsLabels.readInt();
        } catch (IOException e) {
            throw new LoaderException("Cannot to read " + labelFileName, e);
        }

        try {
            int magicImageNumber = dsImages.readInt();
            if (magicImageNumber != 2051) {
                throw new LoaderException("Image file: " + imageFileName + " has wrong format.");
            }

            if (dsImages.readInt() != numberItems) {
                throw new LoaderException("Image file: " + imageFileName + " has different number of items to " + labelFileName);
            }

            int height = dsImages.readInt();
            int width = dsImages.readInt();

            System.out.println(height + " , " + width);
        } catch (IOException e) {
            throw new LoaderException("Cannot to read " + imageFileName, e);
        }

        return null;
    }

    @Override
    public void close() {
        try {
            dsImages.close();
        } catch (Exception e) {
            throw new LoaderException("Cannot close" + imageFileName, e);
        }
        try {
            dsLabels.close();
        } catch (Exception e) {
            throw new LoaderException("Cannot close" + labelFileName, e);
        }
    }

    @Override
    public MetaData getMetaData() {
        return null;
    }

    @Override
    public BatchData readBatch() {
        return null;
    }
}
