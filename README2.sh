# step0: finetune

# -------------------- afhq -------------------- #
# 1.2. results in StyleNeRF: /2022-10-19/22-30-25/, with 0.2*diveloss, which product StyleNeRF/pretrained/afhq_256_0.2dloss.pkl. √

# -------------------- celeba-hq -------------------- #
# 2.2. results in StyleNeRF: /2022-10-19/23-53-18/, with 0.2*diveloss, which product StyleNeRF/pretrained/celeba_256_0.2dloss.pkl. √



# step1: finetune
# -------------------- afhq -------------------- #
# 1.7. finetuning using afhq2c_labels.zip datasets and afhq_256_0.2dloss.pkl (from step0:1.2) pretrained model, as well as using labels at final layer of D following Hyper-Modulation (conditional stylenerf);
#      We use two mappingnetworks for nerf and synthesisblock respectively, in detail, nerf block uses unconditional mappingnetwork and synthesisblock uses conditional mappingnetwork;
#      We ws = (ws_nerf+ws) * 0.5;
#      We also use 0.2 * diveloss for step1.
nohup python -u run_train.py outdir=./output data=/opt/data/private/senmao/datasets/afhq3c_labels.zip spec=paper256 model=stylenerf_afhq  resume=/opt/data/private/senmao/StyleNeRF/pretrained/afhq_256_0.2dloss.pkl cond=True gpus=2 > out_labels.log 2>&1 &
# results in /2022-10-21/12-18-36/, which produce afhqlabels_256_0.2dloss.pkl. √

# -------------------- celeba-hq -------------------- #
# 2.4. finetuning using celeba2c_labels.zip datasets and celeba_256_0.2dloss.pkl (from step0:2.2) pretrained model, as well as using labels at final layer of D following Hyper-Modulation (conditional stylenerf);
#      We use two mappingnetworks for nerf and synthesisblock respectively, in detail, nerf block uses unconditional mappingnetwork and synthesisblock uses conditional mappingnetwork;
#      We ws = (ws_nerf+ws) * 0.5;
#      We also use 0.2 * diveloss for step1.
nohup python -u run_train.py outdir=./output data=/opt/data/private/senmao/datasets/celeba2c_labels.zip spec=paper256 model=stylenerf_afhq  resume=/opt/data/private/senmao/StyleNeRF/pretrained/celeba_256_0.2dloss.pkl cond=True gpus=2 > out_labels.log 2>&1 &
# results in /2022-10-21/09-30-44/, which produce celebalabels_256_0.2dloss.pkl. √



# step2: training adapted layers
# -------------------- afhq -------------------- #
# 1.8. training adapted layers with loss(fake_x_nerf, x_nerf) and loss(fake_img_nerf, img_nerf) of 32, 64, 128 and 256 resolution,
#      as well as ccpl_positive(fake_x_nerf, x_nerf) of 32, 64, 128 and 256 resolution;
#      remove tanh() of unet in the last layer;
#      use afhqlabels_256_0.2dloss.pkl produced by step1/1.7;
nohup python -u run_train_step2.py outdir=./output data=/opt/data/private/senmao/datasets/afhq3c_labels.zip spec=paper256 model=stylenerf_afhq_step2 resume=/opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/pretrained/afhqlabels_256_0.2dloss.pkl cond=True label_dim=3 gpus=2 > out_adapted.log 2>&1 &
# results in /2022-10-22/09-36-16/, which produce adaptedlayers_afhqlabels_256_0.2dloss01.pkl. √

# -------------------- celeba-hq -------------------- #
# 2.2. training adapted layers with loss(fake_x_nerf, x_nerf) and loss(fake_img_nerf, img_nerf) of 32, 64, 128 and 256 resolution,
#      as well as ccpl_positive(fake_x_nerf, x_nerf) of 32, 64, 128 and 256 resolution;
#      remove tanh() of unet in the last layer;
#      use celebalabels_256_0.2dloss.pkl produced by step1/2.4;
nohup python -u run_train_step2.py outdir=./output data=/opt/data/private/senmao/datasets/celeba2c_labels.zip spec=paper256 model=stylenerf_afhq_step2 resume=/opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/pretrained/celebalabels_256_0.2dloss.pkl cond=True label_dim=2 gpus=2 > out_adapted.log 2>&1 &
# results in /2022-10-22/09-40-46/, which produce adaptedlayers_celebalabels_256_0.2dloss01.pkl. √
