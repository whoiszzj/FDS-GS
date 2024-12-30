# FDS-GS: Frequency-Aware Density Control via Reparameterization for High-Quality Rendering of 3D Gaussian Splatting (AAAI'25)

## Overview

This work is the official implementation of FDS-GS. For more information you may refer to our [paper. Here's an abstract about it.

![pipeline](pics/Pipeline.png)

By adaptively controlling the density and generating more Gaussians in regions with high-frequency information, 3D Gaussian Splatting (3DGS) can better represent scene details. From the signal processing perspective, representing details usually needs more Gaussians with relatively smaller scales. However, 3DGS currently lacks an explicit constraint linking the density and scale of 3D Gaussians across the domain, leading to 3DGS using improper-scale Gaussians to express frequency information, resulting in the loss of accuracy. In this paper, we propose to establish a direct relation between density and scale through the reparameterization of the scaling parameters and ensure the consistency between them via explicit constraints (i.e., density responds well to changes in frequency). Furthermore, we develop a frequency-aware density control strategy, consisting of densification and deletion, to improve representation quality with fewer Gaussians. A dynamic threshold encourages densification in high-frequency regions, while a scale-based filter deletes Gaussians with improper scale. Experimental results on various datasets demonstrate that our method outperforms existing state-of-the-art methods quantitatively and qualitatively.

![experiment_vis](pics/ExperimentVis.jpg)

## Installing

Our code were tested under Ubuntu 22.04 with GPU A100. The Cuda version is 11.8 and the PyTorch version is 2.4.1.

 ```bash
 git clone https://github.com/whoiszzj/FDS-GS.git
 cd FDS-GS
 conda env create --file environment.yml
 # install torch kdtree
 git clone https://github.com/thomgrand/torch_kdtree
 cd torch_kdtree
 git submodule init
 git submodule update
 pip install .
 ```

## Usage

```bash
# train
python train.py -s ./data/MipNeRF360/bicycle -m ./output/MipNeRF360/bicycle/gaussians/FDS-GS --eval
# render
python  render.py -m ./output/MipNeRF360/bicycle/gaussians/FDS-GS --iteration 30000
# eval
python metrics.py -m ./output/MipNeRF360/bicycle/gaussians/FDS-GS

# other settings metain the same as the original 3DGS
# you may also use the run.py to test the full dataset of MipNeRF360
python run.py

# our gaussian model save an another parameter 'R' in the ply file
# you can use the convert2gs.py to convert our formart to the 3DGS's
python convert2gs.py -i ./output/MipNeRF360/bicycle/gaussians/FDS-GS/point_cloud/iteration_30000/point_cloud.ply
```

## Data

You may follow the 3DGS to prepare the dataset. Download the MipNeRF360 from [here](https://jonbarron.info/mipnerf360). And the Tanks \& Temples and Deep Blending dataset can be download from [here](https://drive.google.com/drive/folders/162wk_fA6DyRM1BfbnFluKPVoGtfnM2QZ?usp=sharing). Besides we provide some results, which you can directly view with 3DGS viewers, such as [supersplat](https://playcanvas.com/supersplat/editor/).

## Thanks

Thanks for all the open source projects / datasets.

> 3DGS: https://github.com/graphdeco-inria/gaussian-splatting
>
> TorchKDTree: https://github.com/thomgrand/torch_kdtree
>
> MipNeRF360: https://jonbarron.info/mipnerf360/
>
> Tanks \& Temples: https://www.tanksandtemples.org/
>
> Deep Blending: https://github.com/Phog/DeepBlending

