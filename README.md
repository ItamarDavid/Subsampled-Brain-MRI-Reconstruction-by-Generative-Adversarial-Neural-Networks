# Subsampled Brain MRI Reconstruction by Generative Adversarial Neural Networks : PyTorch Implementation

If you use this code for your research, please cite:

Subsampled Brain MRI Reconstruction by Generative Adversarial Neural Networks.
Roy Shaul*, Itamar David*, Ohad Shitrit and Tammy Riklin Taviv (* equal contributions). Accepted to Medical Image Analysis Journal.  [Bibtex - TBD]

---



Data preprocessing:
1. download IXI T1 data from: http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
2. extract it:  tar -xvf IXI-T1.tar
3. run the script 'IXI_preproccing' which would normalize the data and convert it to '*.hdf5' 