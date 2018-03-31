# U-Net-Pytorch
Implementation of U-Net in Pytorch

## Directory Tree

```
+checkpoints_unet
+optimizer_checkpoints_unet
+runs
+graphs_unet
+Samples
+data
	+test
	+train
	+validate
-api.py
-train_Unet.py
-data_augment.py
-networks.py

```
1. **checkpoints_unet**: Contains checkpoints for a model pretrained on [Kaggle's Datascience-Bowl-2018 dataset](https://www.kaggle.com/c/data-science-bowl-2018/data). This will also store the checkpoints that will be used for further training.

2. **optimizer_checkpoints_unet**: Contains optimizer checkpoints corresponding to the ones in checkpoints_unet.(Quite useful when using momentum based optimizers for discontinuous training)

3. **runs**: folder created by TensorBoardX.Can be used to view the training and validation loss. 

```sh

tensorboard --logdir= .

```
4. **graph_unet**: contains log of training (same data as runs) as JSON file. Can be used for plotting with Matplotlib.

5. **Samples**: Samples from [Kaggle's Datascience-Bowl-2018 dataset](https://www.kaggle.com/c/data-science-bowl-2018/data) run through the pretrained UNet.

6. **data**: The data folder contains test,training and validation folder (training and validation folder are during the training). I have provided sample processed datapoints from [Kaggle's Datascience-Bowl-2018 dataset](https://www.kaggle.com/c/data-science-bowl-2018/data) as a reference. Please note that the change in the structure and storage of datapoints has to be reflected in ImageDataset class in train_Unet.py .

7. **api.py**: is used to get segmented output of a given input for a pretrained neural network.It will give output in the folder containing script as Output_unet.png
```
python api.py /path/to/image/image_name.extension 
```
    

8. **train_Unet.py**: Training script for UNet. Please review the comments in the script to understand its hyperparameters and working.

9. **data_augment.py**: Contains data augmentation class.It is used to augment data during training the Unet.

10. **networks.py**: Unet Network architecture is defined here.

Note: This network is compatible with Pytorch 0.3.0 . The next version of Pytorch might cause some major syntax changes.
