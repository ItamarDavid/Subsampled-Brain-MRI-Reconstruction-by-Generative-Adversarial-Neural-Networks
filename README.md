
# Subsampled Brain MRI Reconstruction by Generative Adversarial Neural Networks : PyTorch Implementation
![Architecture](https://raw.githubusercontent.com/ItamarDavid/Subsampled-Brain-MRI-Reconstruction-by-Generative-Adversarial-Neural-Networks/master/figures/architecture.jpg)
This is a PyTorch implementation of ["Subsampled Brain MRI Reconstruction by Generative Adversarial Neural Networks"](https://authors.elsevier.com/a/1bIAU4rfPm3jGs). 
Since the original code was implemented in TensorFlow<1.0 and Python 2.7 we reimplemented the code in PyTorch.

There are several changes from the code that was used for the published results:
 - Generator network model is slightly modified (Now uses bilinear upsampling instead of transposed convolutions, Number of convs slightly changed).
 - Discriminator network is now a "PatchGAN". Images are center cropped before the discriminator so blank areas won't be classified.
 - Slightly different preprocessing scheme.
 - Slightly different loss weighting.  
The code still acheives comparable (or even better results) to those published.

If you use this code for your research, please cite:

Subsampled Brain MRI Reconstruction by Generative Adversarial Neural Networks.
Roy Shaul*, Itamar David*, Ohad Shitrit and Tammy Riklin Taviv (* equal contributions).
Medical Image Analysis 2020.
BibTeX:

>        @article{shaul2020subsampled,
>       title={Subsampled Brain MRI Reconstruction by Generative Adversarial Neural Networks},
>       author={Shaul, Roy and David, Itamar and Shitrit, Ohad and Raviv, Tammy Riklin},
>       journal={Medical Image Analysis},
>       pages={101747},
>       year={2020},
>       publisher={Elsevier}
>     }

---

## Using the code:
### Data and preprocessing:
The code currently supports the [IXI dataset](https://brain-development.org/ixi-dataset/). However, it should be fairly simple to convert the preprocessing code and data generator to process any dataset saved nifti format.

**IXI preprocessing:**
1. Download IXI T1 data from: http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
2. Extract the tar file `tar -xvf IXI-T1.tar`.
3. Run the script 'IXI_preproccing.py' which would normalize the data and convert it to hdf5 format.

### Training the model
Once the data is preprocessed the network can be trained using the script 'train.py'.
The training should be configured using the 'config.yaml' file provided in the repository.
The cofiguration file provided is configured for the IXI dataset reconstruction and can be used after setting the preprocessed data paths.
Many network and model parameters can be changed using the configuration file - e.g. Turn off adverserial loss (to achieve higher PSNR), change the sampling mask precentage, change batch size and many more.

### TensorBoard
By default the script saves losses and images to [TensorBoard](https://www.tensorflow.org/tensorboard) so you can track and review the training process.
The TensorBoard logs are saved in the training output directory.

### Process Test Data
The test data can be processed using 'predict.py'. The script loads a trained model, process the dataset test folder and save/show the model predictions.
The script is also configured using the 'config.yaml' file.


## Acknowledgments:
We were inspired and used the following (awesome!) github projects:
 - [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/)
 - [UNet: semantic segmentation with PyTorch](https://github.com/milesial/Pytorch-UNet)
 
 
 


