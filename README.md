# AtlasNet
This is a pytorch version of the implementation code for article 
***Deep Atlas Network for Efficient 3D Left Ventricle Segmentation on Echocardiography***
published above ***Medical Image Analysis***.

## install
Use the following command to install the environment.

```angular2html
conda env create -f atlasnet.yaml
```

## dataset
We use the CETUS dataset in our code, and the image size was changed to 128x128x128.

In the folder ./CETUS, we use some examples to show the format of the data.

## train/test
After setting params in the ./Model/config.py
Run train.py/test.py directly to start training/testing.
