# 3DI2I: 3D-Aware Multi-Class Image-to-Image Translation with NeRFs</sub>

![Random Sample](./docs/random_sample.png)

**3DI2I: 3D-Aware Multi-Class Image-to-Image Translation with NeRFs**<br>

Abstract: *Recent advances in 3D-aware generative models (3D-aware GANs) combined with Neural Radiance Fields (NeRF) have achieved impressive results for novel view synthesis. However no prior works investigate 3D-aware GANs for 3D consistent multi-class image-to-image (3D-aware I2I) translation. Naively using 2D-I2I translation methods suffers from unrealistic shape/identity change. To perform 3D-aware multi-class I2I translation, we decouple this learning process into a multi-class 3D-aware GAN step and a 3D-aware I2I translation step.   In the first step, we propose two novel techniques: a new conditional architecture and a effective training strategy.  In the second step, based on the well-trained multi-class 3D-aware GAN architecture that preserves view-consistency,  we construct a 3D-aware I2I translation system. To further reduce the view-consistency problems, we propose several new techniques, including a U-net-like adaptor network design, a hierarchical representation constrain and a relative regularization loss.   In extensive experiments on two datasets, quantitative and qualitative results demonstrate  that we successfully perform  3D-aware I2I translation  with  multi-view  consistency.*

## Requirements
The codebase is tested on 
* Python 3.8
* PyTorch 1.7.0
* 2× Quadro RTX 3090 GPUs (24 GB VRAM) with CUDA version 11.7

For additional python libraries, please install by:

```
pip install -r requirements.txt
```

## Datasets
Preparing datasets following [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch#preparing-datasets) for 3D-aware I2I translation.

**1. dataset for unconditional stylenerf:**
```
python apps/dataset_tool.py --source=~/data/afhq/train/ --dest=~/datasets/afhq.zip --width 256 --height 256
```
```
python apps/dataset_tool.py --source=~/data/celeba_hq/train/ --dest=~/datasets/celeba_hq.zip --width 256 --height 256
```


**2. dataset for conditional stylenerf**

2.1. create labels for afhq and celeba-hq datasets. 
```
python apps/dataset_labels.py --out=~/data/afhq/train/dataset.json --source=~/data/afhq/train/
```
```
python apps/dataset_labels.py --out=~/data/celeba_hq/train/dataset.json --source=~/data/celeba_hq/train/
```

2.2. create dataset with lables (dataset.json) for afhq and celeba-hq.
```
python apps/dataset_tool.py --source=~/data/afhq/train/ --dest=~/datasets/afhq3c_labels.zip --width 256 --height 256
```
```
python apps/dataset_tool.py --source=~/data/celeba_hq/train/ --dest=~/datasets/celeba2c_labels.zip --width 256 --height 256
```

## Training
**1. unconditional 3D-aware generative model (using [StyleNeRF](https://github.com/facebookresearch/StyleNeRF) with stylenerf_afhq.yaml).**

cd ${CodePath}/StyleNeRF/
finetune using mixed afhq(cat, dog and wild) datasets and ffhq_256.pkl pretrained model (unconditional stylenerf).
```
python run_train.py outdir=./output data=~/datasets/afhq.zip spec=paper256 model=stylenerf_afhq  resume='ffhq256' cond=False
```
finetune using mixed celeba-hq(female and male) datasets and ffhq_256.pkl pretrained model (unconditional stylenerf).
```
python run_train.py outdir=./output data=~/datasets/celeba_hq.zip spec=paper256 model=stylenerf_afhq  resume='ffhq256' cond=False
```

**2. conditional 3D-aware generative model**

```
python run_train.py outdir=./output data=~/datasets/afhq3c_labels.zip spec=paper256 model=stylenerf_afhq  resume=./pretrained/afhq_256.pkl cond=True gpus=2
```
```
python run_train.py outdir=./output data=~/datasets/celeba2c_labels.zip spec=paper256 model=stylenerf_afhq  resume=./pretrained/celeba_256_0.2dloss.pkl cond=True gpus=2
```

the trained model save as afhqlabels_256.pkl and celebalabels_256.pkl.

**3. 3D-aware I2I translation**

```
python run_train_step2.py outdir=./output data=~/datasets/afhq3c_labels.zip spec=paper256 model=stylenerf_afhq_step2 resume=./pretrained/afhqlabels_256.pkl cond=True label_dim=3 gpus=2
```
```
python run_train_step2.py outdir=./output data=~/datasets/celeba2c_labels.zip spec=paper256 model=stylenerf_afhq_step2 resume=./pretrained/celebalabels_256.pkl cond=True label_dim=2 gpus=2
```

the trained model save as afhqadaptor_256.pkl and celebaadaptor_256.pkl.

## Rendering 3D-aware I2I translation results using the pretrained model

Example of 3D-aware I2I translation of dog into cat and wild on AFHQ $256^2$
```
python generate_3d23dt.py --network="{'stylenerf-3d23d': './pretrained/afhqlabels_256.pkl', 'adapted-layers': './pretrained/afhqadaptor_256.pkl'}" \
                          --class_label="[[1, 0, 0], [0, 1, 0], [0, 0, 1]]" --seed_nerf 1 --seed 1 --batch_size 16 --save_3dvideo 0 --batch_idx 15 \
                          --save_3dframes 1 --save_sgl_3dvideo 1 --save_sglframes 1 --class 1
```

Example of 3D-aware I2I translation of male into female on Celeba-HQ $256^2$
```
python generate_3d23dt.py  --network="{'stylenerf-3d23d': './pretrained/celebalabels_256.pkl', 'adapted-layers': './pretrained/celebaadaptor_256.pkl'}" \
                           --class_label="[[1, 0], [0, 1]]" --seed_nerf 2 --seed 2 --batch_size 13 --save_3dvideo 0 --batch_idx 12 \
                           --save_3dframes 1 --save_sgl_3dvideo 1 --save_sglframes 1 --class 1
```






