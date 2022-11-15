# 3DI2I: 3D-Aware Multi-Class Image-to-Image Translation with NeRFs</sub>

![Random Sample](./docs/random_sample.png)

**3DI2I: 3D-Aware Multi-Class Image-to-Image Translation with NeRFs**<br>

Abstract: *Recent advances in 3D-aware generative models (3D-aware GANs) combined with Neural Radiance Fields (NeRF) have achieved impressive results for novel view synthesis. However no prior works investigate 3D-aware GANs for 3D consistent multi-class image-to-image (3D-aware I2I) translation. Naively using 2D-I2I translation methods suffers from unrealistic shape/identity change. To perform 3D-aware multi-class I2I translation, we decouple this learning process into a multi-class 3D-aware GAN step and a 3D-aware I2I translation step.   In the first step, we propose two novel techniques: a new conditional architecture and a effective training strategy.  In the second step, based on the well-trained multi-class 3D-aware GAN architecture that preserves view-consistency,  we construct a 3D-aware I2I translation system. To further reduce the view-consistency problems, we propose several new techniques, including a U-net-like adaptor network design, a hierarchical representation constrain and a relative regularization loss.   In extensive experiments on two datasets, quantitative and qualitative results demonstrate  that we successfully perform  3D-aware I2I translation  with  multi-view  consistency.*

## Requirements
The codebase is tested on 
* Python 3.8
* PyTorch 1.7.0
* 2× Quadro RTX 3090 GPUs (24 GB VRAM) with CUDA version 11.0

For additional python libraries, please install by:

```
pip install -r requirements.txt
```

Please refer to https://github.com/NVlabs/stylegan2-ada-pytorch for additional software/hardware requirements.

## Dataset
We follow the same dataset format as [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch#preparing-datasets) supported, which can be either an image folder, or a zipped file.

You can also download the zipped datasets from [Hugging Face 🤗](https://huggingface.co/datasets/thomagram/StyleNeRF_Datasets).

## Pretrained Checkpoints
You can download the pre-trained checkpoints (used in our paper) and some recent variants trained with current codebase as follows:
| Dataset   | Resolution | #Params(M) | Config |                           Download                           |
| :-------- | :--------: | :--------: | :----: | :----------------------------------------------------------: |
| FFHQ      |    256     | 128        | Default |  [Hugging Face 🤗](https://huggingface.co/facebook/stylenerf-ffhq-config-basic/blob/main/ffhq_256.pkl) |
| FFHQ      |    512     | 148        | Default |  [Hugging Face 🤗](https://huggingface.co/facebook/stylenerf-ffhq-config-basic/blob/main/ffhq_512.pkl) |
| FFHQ      |    1024    | 184        | Default |  [Hugging Face 🤗](https://huggingface.co/facebook/stylenerf-ffhq-config-basic/blob/main/ffhq_1024.pkl) |

（I am slowly adding more checkpoints. Thanks for your very kind patience!)


## Train a new StyleNeRF model
```bash
python run_train.py outdir=${OUTDIR} data=${DATASET} spec=paper512 model=stylenerf_ffhq
```
It will automatically detect all usable GPUs.

Please check configuration files at ```conf/model``` and ```conf/spec```. You can always add your own model config. More details on how to use hydra configuration please follow https://hydra.cc/docs/intro/.

## Render the pretrained model
```bash
python generate.py --outdir=${OUTDIR} --trunc=0.7 --seeds=${SEEDS} --network=${CHECKPOINT_PATH} --render-program="rotation_camera"
```
It supports different rotation trajectories for rendering new videos.

## Run a demo page
```bash
python web_demo.py 21111
```
It will in default run a Gradio-powered demo on https://localhost:21111

[NEW]
The demo is also integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/facebook/StyleNeRF)

![Web demo](./docs/web_demo.gif)
## Run a GUI visualizer
```bash
python visualizer.py
```
An interative application will show up for users to play with.
![GUI demo](./docs/gui_demo.gif)
## Citation

```
@inproceedings{
    gu2022stylenerf,
    title={StyleNeRF: A Style-based 3D Aware Generator for High-resolution Image Synthesis},
    author={Jiatao Gu and Lingjie Liu and Peng Wang and Christian Theobalt},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=iUuzzTMUw9K}
}
```


## License

Copyright &copy; Facebook, Inc. All Rights Reserved.

The majority of StyleNeRF is licensed under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/), however, portions of this project are available under a separate license terms: all codes used or modified from [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) is under the [Nvidia Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).


