from dataloader import DataLoader
import numpy as np

image_loader = DataLoader(platform='Local', # ["Local", "Kaggle", "Colab"]
                          n_classes=2,      # [2, 3]
                          data_dir=None,    
                          txt_dir=None,     
                          img_size=224,     
                          combined=True,    
                          channels='RGB')   # ["RGB", "L"]

X_train, _, Y_train, _ = image_loader.load_train_val(load_full_train=True)

np.save('./data/input/full_xtrain.npy', X_train)
np.save('./data/input/full_ytrain.npy', Y_train)

X_test, Y_test = image_loader.load_test()

np.save('./data/input/xtest.npy', X_test)
np.save('./data/input/ytest.npy', Y_test)