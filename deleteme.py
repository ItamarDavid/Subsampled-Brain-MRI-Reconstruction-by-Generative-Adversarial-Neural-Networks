import numpy as np
import scipy.io
import os
import pickle


mask_path = './Masks/'
lisdir = os.listdir(mask_path)

for mask_name in lisdir:
    if mask_name.endswith('.mat'):
        mat = scipy.io.loadmat(mask_path + mask_name)
        masks = {'mask0': mat['mask_1'],
                 'mask1': mat['mask_2'],
                 'mask2': mat['mask_3']}

        mask_name_p = mask_name.replace('mat', 'pickle')
        # np.save(mask_path + mask_name_np, masks)
        pickle.dump(masks, open(mask_path + mask_name_p, "wb"))

