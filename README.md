# ESRGAN-pytorch

This repository implements a deep-running model for super resolution.
 Super resolution allows you to pass low resolution images to CNN and restore them to high resolution. 
 We refer to the following article.  
 [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)  
 
 ## architecture
 [Overall Architecture]
 ![ESRGAN architecture](./image/architecture.PNG)  
 [Basic block]  
 ![BasicBlock](./image/basicBlock.PNG)
 
 ## Prepare dataset
 ### Use Flicker2K and DIV2K
```bash
cd datasets
python prepare_datasets.py
cd ..
```
### custom dataset
Make dataset like this; size of hr is 128x128 ans lr is 32x32
```
datasets/
    hr/
        0001.png
        sdf.png
        0002.png
        0003.png
        0004.png
        ...
    lr/
        0001.png
        sdf.png
        0002.png
        0003.png
        0004.png
        ...
```

## how to train
run main file
```bash
python main.py --is_perceptual_oriented True --num_epoch=10
python main.py --is_perceptual_oriented False --epoch=10
```

## Sample
From the top is a low resolution image, a restored high resolution image, and ground truth.  

The sample is still being imported during the training, and the completeness is low.  

 ![Sample](./image/sample1.png)
 ![Sample](./image/sample2.png)
 ![Sample](./image/sample3.png)
 
