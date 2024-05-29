# Deeper, Sharper, Faster: Application of Efficient Transformer to Galaxy Image Restoration 

[![arXiv](https://img.shields.io/badge/arXiv-2404.00102-b31b1b.svg)](https://arxiv.org/abs/2404.00102)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11378660.svg)](https://doi.org/10.5281/zenodo.11378660)

<hr />

> **Abstract:** *The Transformer architecture has revolutionized the field of deep learning over the past several years in diverse areas, including natural language processing, code generation, image recognition, time series forecasting, etc. We propose to apply Zamir et al.'s efficient transformer to perform deconvolution and denoising to enhance astronomical images. We conducted experiments using pairs of high-quality images and their degraded versions, and our deep learning model demonstrates exceptional restoration of photometric, structural, and morphological information. When compared to the ground-truth JWST images, the enhanced versions of our HST-quality images reduce the scatter of isophotal photometry, Sersic index, and half-light radius by factors of 4.4, 3.6, and 4.7, respectively, with Pearson correlation coefficients approaching unity.
The performance is observed to degrade when input images exhibit severely correlated noise, point-like sources, and artifacts. We anticipate that this deep learning model will prove valuable for a number of scientific applications, including precision photometry, morphological analysis, and shear calibration.* 
<hr />


## Installation

Follow these intructions

1. Clone Repository
```
git clone https://github.com/JOYONGSIK/GalaxyRestoration.git
cd GalaxyRestoration
```

2. Make conda environment
```
conda create -n galaxy python=3.7
conda activate galaxy
```

3. Install dependencies
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips timm 
pip install wandb==0.13.10
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```

## Dataset Preperation

The test data should be placed in ```dataset/{HST or JWST}/``` directory.
```
└───dataset
    ├───JWST
    │   ├───gt
    │   └───lq
    └───HST
```

## Inference 

To obtain restored galaxy images from the model ```{HST or JWST}_inference.py``` can be used.

```
### You have to check dataset path. 
### You can revise data-path in inference.py file.

# If you want to restore JWST Dataset
python JWST_inference.py 

# If you want to restore HST Dataset
python HST_inference.py 
```

## Contact 

Should you have any questions, please contact josik@unist.ac.kr, hyosun.park@yonsei.ac.kr

**Acknowledgment:** This code is based on the [Restormer](https://github.com/swz30/Restormer) repository. 
