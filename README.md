# SimCLR

Self supervised learning in medical images(X-ray images). 
Notice that the augmentation functions are only random flip, random cropping and brightness in coherent with X-ray images.

## Train the model

### Arguments:
* --epochs: training epochs
* --image_path: the dataset path
* --temperature: the hyperparameter for contrastive loss
* --checkpoint_path: path to save the model checkpoint 

Feel free to change the hyperparameters in the dictionary 'para'

```
python3 train_SSL.py
```

## Reference:

https://keras.io/examples/vision/semisupervised_simclr/

https://arxiv.org/abs/2002.05709
