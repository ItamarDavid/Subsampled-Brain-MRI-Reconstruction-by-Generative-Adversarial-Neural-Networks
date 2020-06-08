
import h5py
import os
import numpy as np

def save_data(pred_Im0, pred_K0, pred_Im1, pred_K1, noisy0, noisy1, out_file_name):
    data_shape = pred_Im0.shape
    with h5py.File(out_file_name, 'w') as f:
        dset = f.create_dataset('pred_Im0', data_shape, data=pred_Im0, compression="gzip", compression_opts=9)
        dset = f.create_dataset('pred_K0', data_shape, data=pred_K0, compression="gzip", compression_opts=9)
        dset = f.create_dataset('pred_Im1', data_shape, data=pred_Im1, compression="gzip", compression_opts=9)
        dset = f.create_dataset('pred_K1', data_shape, data=pred_K1, compression="gzip", compression_opts=9)
        dset = f.create_dataset('noisy0', data_shape, data=noisy0, compression="gzip", compression_opts=9)
        dset = f.create_dataset('noisy1', data_shape, data=noisy1, compression="gzip", compression_opts=9)