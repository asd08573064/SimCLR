para_SSL = {
    'image_size': 256, 
    'image_channels': 3,
    'width': 256,
    'temperature': 0.1
}

para_trans = {
    'num_layers' : 4,
    'd_model' : 128,
    'dff' : 128,
    'num_heads' : 4,
    'dropout_rate' : 0.3
}

contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
classification_augmentation = {"min_area": 0.75, "brightness": 0.3, "jitter": 0.1}