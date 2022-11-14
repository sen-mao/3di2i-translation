# Copyright (c) Computer school, NKU(Nankai University).

"""Generate images using pretrained stylenerf-3d23dt and adapted network pickle trained."""
import math
import os
import re
import time
import glob
from typing import List, Optional
import cv2
from einops import rearrange
import copy
from tqdm import tqdm
import ast

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import imageio
import legacy
from renderer import Renderer
from dnnlib.util import dividable, hash_func, EasyDict
from training.data_utils import save_image_grid
from training.utils import encode_image

def proc_img(img):
    return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def wvideo(img, name, outdir, grid_size):
    if isinstance(img, List):
        gw, gh = grid_size

        def stack_imgs(imgs):
            img = torch.stack(imgs, dim=2)
            return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)

        def reshape_imgs(imgs):
            B, H, W, C = imgs.shape
            imgs = imgs.reshape(gh, gw, H, W, C)
            imgs = imgs.permute(0, 2, 1, 3, 4)
            return imgs.reshape(gh * H, gw * W, C)

        imgs = [proc_img(i) for i in img]
        # write to video
        imgs = [reshape_imgs(imgs[k]).numpy() for k in range(len(imgs))]
        imageio.mimwrite(f'{outdir}/{name}.mp4', imgs, fps=30, quality=8)

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True, type=ast.literal_eval,
              default="{'stylenerf-3d23d': './pretrained/afhqlabels_256.pkl', 'adapted-layers': './pretrained/adaptedlayers_afhqlabels_256_wostylemix.pkl'}")
@click.option('--class_label', help='class lable', type=ast.literal_eval, required=True, default='[[1, 0, 0], [0, 1, 0], [0, 0, 1]]')
@click.option('--class_name', help='class name', type=ast.literal_eval, required=True, default='[\'cat\', \'dog\', \'wild\']')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=0)
# seed for latent code z
@click.option('--seed_nerf', type=int, help='List of random seeds for z_nerf', default=2022)
@click.option('--seed', type=int, help='List of random seeds for z', default=2022)
# rendering parameters
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--render-program', default='rotation_camera', show_default=True)
@click.option('--render-option', default=None, type=str, help="e.g. up_256, camera, depth")
@click.option('--relative_range_u_scale', default=1.0, type=float, help="relative scale on top of the original range u")
@click.option('--n_steps', default=50, type=int, help="number (n_steps*4) of steps for each seed")
# save parameters
@click.option('--batch_size', help='batch size (number of input images)', type=int, required=True, default=17)
@click.option('--save_3dvideo', help='if save 3d video', type=bool, required=True, default=True)
@click.option('--save_3dframes', help='if save 3d frames including jframes (opt), sgl_3dvideo (opt) and sglframes (opt)', type=bool, required=True, default=False)
@click.option('--save_jframes', help='if save 3d joined frames', type=bool, required=True, default=False)
@click.option('--batch_idx', help='batch index (seed index), 0~batch_size-1', type=int, required=True, default=29)
@click.option('--save_sgl_3dvideo', help='if save single 3d video', type=bool, required=True, default=False)
@click.option('--save_sglframes', help='if save singal 3d frames', type=bool, required=True, default=False)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR', default='./results/afhq')
@click.option('--sample_frames', help='if sample frames', type=bool, required=True, default=False)
@click.option('--save_fnimg', help='if save fake_nerf_image and D_images', type=bool, required=True, default=False)
# local images
@click.option('--input-source', type=click.Choice(['real', 'generated']), help='where the input images come from?', default='generated')
@click.option('--indir', help='directory of input images', type=str, required=True, metavar='DIR', default='/opt/data/private/senmao/data/afhq/val/cat')
@click.option('--img_res', help='image resolution', type=int, required=True, default=256)
# step2
@click.option('--step2', help='if use step2 to translate images', type=bool, required=True, default=True)
# encoder to reference latent code
@click.option('--inversion', help='use Image2StyleGAN to Embed Images Into w', type=int, required=True, default=False)
@click.option('--num_steps', help='iterations of predicting w', type=int, required=True, default=350)

def generate_images(
        ctx: click.Context,
        network_pkl: dict,
        seed_nerf: int,
        seed: int,
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        projected_w: Optional[str],
        class_label = None,
        class_name  = None,
        class_idx   = None,
        render_program=None,
        render_option=None,
        n_steps     = 50,
        relative_range_u_scale=1.0,
        # save 3d video and corresponding frames
        batch_size  = 4,
        save_3dvideo = True,
        save_3dframes = False,
        save_jframes = False,
        batch_idx   = 0, # 0, 1, 2, 3
        save_sgl_3dvideo = False,
        save_sglframes = False,
        sample_frames = False,
        # save fake_nerf_images and D_images
        save_fnimg=False,
        # step2
        step2=True,
        # encoder
        input_source = 'generated',
        indir = '',
        img_res = 256,
        inversion = False,
        num_steps = 300
):
    start_time = time.time()

    device = torch.device('cuda')
    if os.path.isdir(network_pkl['stylenerf-3d23d']) and os.path.isdir(network_pkl['adapted-layers']):
        network_pkl['stylenerf-3d23d'] = sorted(glob.glob(network_pkl['stylenerf-3d23d'] + '/*.pkl'))[-1]
        network_pkl['adapted-layers'] = sorted(glob.glob(network_pkl['adapted-layers'] + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)

    modules = {}
    for network_key in network_pkl:  # load network pkl
        with dnnlib.util.open_url(network_pkl[network_key]) as f:
            network = legacy.load_network_pkl(f)
            if network_key == 'stylenerf-3d23d':
                modules['G'] = network['G_ema'].to(device)
                modules['D'] = network['D'].to(device)
            elif network_key == 'adapted-layers':
                modules['Adapted_net'] = network['Adapted_net'].to(device)

    os.makedirs(outdir, exist_ok=True)
    transclass = ''.join([i for i in class_name if i!=class_name[class_idx]])
    outdir = os.path.join(outdir, f'{class_name[class_idx]}2{transclass}')
    os.makedirs(outdir, exist_ok=True)

    from training.networks import Generator
    from training.stylenerf import Discriminator
    from training.adaptednet import AdaptedNet
    from torch_utils import misc
    with torch.no_grad():
        # G
        G_init_kwargs = EasyDict(**modules['G'].init_kwargs)
        G = Generator(*modules['G'].init_args, **G_init_kwargs).to(device)
        misc.copy_params_and_buffers(modules['G'], G, require_all=False)
        # D
        D_init_kwargs = EasyDict(**modules['D'].init_kwargs)
        D_init_kwargs.step = 2
        D = Discriminator(*modules['D'].init_args, **D_init_kwargs).to(device)
        misc.copy_params_and_buffers(modules['D'], D, require_all=False)
        # Adapted_net
        Adapted_net = AdaptedNet(*modules['Adapted_net'].init_args, **modules['Adapted_net'].init_kwargs).to(device)
        misc.copy_params_and_buffers(modules['Adapted_net'], Adapted_net, require_all=False)
    G2 = Renderer(G, D, program=render_program)
    G2.set_random_seed(seed)

    z_nerf = torch.from_numpy(np.random.RandomState(seed_nerf).randn(batch_size, G.z_dim)).to(device)
    z = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, G.z_dim)).to(device)

    def c(index):
        return torch.tensor(class_label[index]).repeat(batch_size, 1).to(z.device)
    label = c(class_idx)
    bsize = math.sqrt(batch_size)
    if bsize-int(bsize) == 0:
        grid_size = (int(bsize), int(bsize))
    else:
        grid_size = (batch_size, 1)

    # step1: get input images for adaptor net.
    assert input_source in ['real', 'generated']
    imgs = []
    if input_source == 'real':
        for file in sorted(os.listdir(indir))[:batch_size]:
            img = cv2.imread(f'{indir}/{file}')  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
            img = cv2.resize(img, (img_res, img_res))
            # PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{file}')
            img = torch.from_numpy(img)
            imgs += [rearrange(img, 'h w c -> c h w')]
        imgs = torch.stack(imgs, 0)
        imgs = imgs.to(device).to(torch.float32) / 127.5 - 1
        # image to latent code (ws for synthesisblock)
        ws_nerf, ws = None, None
        if inversion == True:
            ws_nerf, ws = encode_image(batch_size, imgs, img_res, num_steps, G, D, Adapted_net, device)
    elif input_source == 'generated':
        # nerf-based
        with torch.no_grad():
            relative_range_u = [0.5 - 0.5 * relative_range_u_scale, 0.5 + 0.5 * relative_range_u_scale]
            outputs = G2(
                z_nerf=z_nerf,
                z=z,
                c=label,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
                render_option=render_option,
                n_steps=n_steps,
                relative_range_u=relative_range_u,
                return_cameras=True)
            if isinstance(outputs, tuple):
                imgs, cameras = outputs
            else:
                imgs = outputs
            # save sd video
            if save_3dvideo:
                wvideo(imgs, f'{class_name[class_idx]}_fs1_sdvideo', outdir, grid_size)
            # save 3d frames
            if save_3dframes:
                curr_out_dir = os.path.join(outdir, f'seed{seed}_{batch_idx}')
                os.makedirs(curr_out_dir, exist_ok=True)
                img_dir = os.path.join(curr_out_dir, f'{class_name[class_idx]}_fs1/step1')
                os.makedirs(img_dir, exist_ok=True)

                sgl_3dvideo = []
                for step, img in enumerate(imgs):
                    # single 3d video of batch_idx
                    if batch_idx != -1:
                        sgl_3dvideo.append(img[batch_idx, :, :, :].unsqueeze(dim=0))
                    # sample
                    if sample_frames and step not in np.linspace(1, n_steps*4, 8, endpoint=False).astype(np.int32)-1:
                        continue
                    # save joined frames
                    if save_jframes:
                        save_image_grid(img.cpu().numpy(), f'{img_dir}/join_{step:03d}.png', drange=[-1, 1], grid_size=grid_size)
                    # single 3d video and frames
                    pim = proc_img(img)
                    for n, im in enumerate(img):
                        if batch_idx == -1: pass
                        elif batch_idx != n: continue
                        # frames of batch_idx
                        if save_sglframes:
                            PIL.Image.fromarray(pim[n].detach().cpu().numpy(), 'RGB').save(f'{img_dir}/{n}_{step:03d}.png')
                if save_sgl_3dvideo:
                    wvideo(sgl_3dvideo, f'{class_name[class_idx]}_fs1', curr_out_dir, grid_size=(1, 1))
    print(f'| -------------------- step1-{class_name[class_idx]}: Done -------------------- |')


    if not step2:
        print(f'time: {(time.time() - start_time)  :.3f} seconds')
        return
    z_ada = torch.from_numpy(np.random.RandomState(seed+3).randn(batch_size, G.z_dim)).to(device)
    z_ada = z
    # step2: translate imgs to adapted_imgs
    with torch.no_grad():
        domain = []
        c_len = len(class_label)
        adapted_imgs = [[] for _ in range(c_len)]
        # D_imgs = []
        fake_imgs_nerf = []  # imgs outputed by adapted
        for gen_img in tqdm(imgs, desc='step2', ncols=80):
            b64_x = D(gen_img, step=2)
            # D_imgs.append(D(gen_img, c(label_index), step=1)['img_128'])
            fake_x_nerf, fake_img_nerf = Adapted_net(b64_x.to(dtype=torch.float32))
            if save_fnimg: fake_imgs_nerf.append(fake_img_nerf)
            lindex = class_idx
            for i in range(c_len):
                lindex = lindex + 1 if lindex + 1 < c_len else 0
                label = c(lindex)
                domain.append(lindex)
                if input_source == 'real' and inversion == True:  # input_source in ['real', 'generation']
                    adapted_img = G.get_final_output_adapted(styles=[ws_nerf, ws], fake_x_nerf=fake_x_nerf)
                elif input_source == 'generated':
                    adapted_img = G.get_final_output_adapted(z=z_ada, c=label, fake_x_nerf=fake_x_nerf, noise_mode='const')
                adapted_imgs[i].append(adapted_img)
        if save_fnimg:
            wvideo(fake_imgs_nerf, 'fake_imgs_nerf', outdir, grid_size)
            # wvideo(D_imgs, network_pkl['stylenerf-3d23d'], outdir, seed, i=-2)
        if save_3dvideo:
            for i in range(c_len):
                wvideo(adapted_imgs[i], f'2{class_name[domain[i]]}_3dvideo', outdir, grid_size)
        # save adapted images (frames)
        if save_3dframes:
            sgl_3dvideo = [[] for _ in range(c_len)]
            for n, adapted_img in enumerate(adapted_imgs):  # dog/cat/wild
                domian_dir = os.path.join(curr_out_dir, f'2{class_name[domain[n]]}')
                os.makedirs(domian_dir, exist_ok=True)
                for step, img in enumerate(adapted_img):
                    # single 3d video of batch_idx
                    if batch_idx != -1:
                        sgl_3dvideo[n].append(img[batch_idx, :, :, :].unsqueeze(dim=0))
                    # sample
                    if sample_frames and step not in np.linspace(1, n_steps*4, 8, endpoint=False).astype(np.int32)-1:
                        continue
                    # save joined frames
                    if save_jframes:
                        save_image_grid(img.cpu().numpy(), f'{domian_dir}/join_{n}_{step:03d}.png', drange=[-1, 1], grid_size=grid_size)
                    pim = proc_img(img)
                    for i, im in enumerate(pim):
                        if batch_idx == -1: pass
                        elif batch_idx != i: continue  # just save No.seed_idx image
                        # frames of batch_idx
                        if save_sglframes:
                            PIL.Image.fromarray(im.detach().cpu().numpy(), 'RGB').save(f'{domian_dir}/{n}_{i}_{step:03d}.png')
            if save_sgl_3dvideo:
                for i, s3dv in enumerate(sgl_3dvideo):
                    wvideo(s3dv, f'2{class_name[domain[i]]}', curr_out_dir, grid_size=(1, 1))
    print('| -------------------- step2: Done -------------------- |')

    print(f'time: {(time.time() - start_time)  :.3f} seconds')


if __name__ == "__main__":
    generate_images()