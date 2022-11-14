# -------------------- afhq -------------------- #

# 1. cat2dogwild/seed1_15/cat_fs1, 2cat, 2dog and 2wild.
# cd /StyleNeRF-2MappingNetwork-CCPL
python generate_3d23dt.py  --network="{'stylenerf-3d23d': './pretrained/afhqlabels_256.pkl', 'adapted-layers': './pretrained/adaptedlayers_afhqlabels_256_wostylemix.pkl'}" --class_label="[[1, 0, 0], [0, 1, 0], [0, 0, 1]]" \
                           --seed_nerf 1 --seed 1 --batch_size 16 --save_3dvideo 0 --batch_idx 15 --save_3dframes 1 --save_sgl_3dvideo 1 --save_sglframes 1 --class 0
# 2. cat2dogwild/seed1_15/2starganv2 with latent code z.
# cd /stargan-v2-transstep1.
python main.py --mode sample --num_domains 3 --resume_iter 100000 --w_hpf 0 --checkpoint_dir expr/checkpoints/afhq --domain_name="['cat', 'dog', 'wild']" \
               --result_dir /opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results/afhq/cat2dogwild/seed1_15/cat_fs1 \
               --src_dir /opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results/results/afhq/cat2dogwild/seed1_15/2starganv2 --seedfz 15

# step1 (cat_fs1) and step2 (2cat, 2dog and 2wild)
python calc_temporalloss.py --img_path=/opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results/cat2dogwild/seed1_15 --save_flow 0 --interval=16
python calc_temporalloss.py --img_path=/opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results/cat2dogwild/seed1_15/cat_fs1 --save_flow 0 --interval=16
# stargan-v2 (2cat, 2dog and 2wild)
python calc_temporalloss.py --img_path=/opt/data/private/senmao/stargan-v2-transstep1/expr/results/afhq/cat2dogwild/seed1_15/2starganv2 --save_flow 0 --interval=16


# -------------------- celeba-hq -------------------- #

# 1. female2male/seed2_12/female_fs1, 2female and 2male.
# cd /StyleNeRF-2MappingNetwork-CCPL
python generate_3d23dt.py  --network="{'stylenerf-3d23d': './pretrained/celebalabels_256.pkl', 'adapted-layers': './pretrained/adaptedlayers_celebalabels_256_wostylemix.pkl'}" --class_label="[[1, 0], [0, 1]]" \
                           --seed_nerf 1 --seed 1 --batch_size 14 --save_3dvideo 0 --batch_idx 13 --save_3dframes 1 --save_sgl_3dvideo 1 --save_sglframes 1 --class 0
# 2. female2male/seed2_12/2starganv2 with latent code z.
# cd /stargan-v2-transstep1.
python main.py --mode sample --num_domains 2 --resume_iter 100000 --w_hpf 1 --checkpoint_dir expr/checkpoints/celeba_hq --domain_name="['female', 'male']" \
               --result_dir /opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results/celeba-hq/female2male/seed2_12/2starganv2 \
               --src_dir /opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results/celeba-hq/female2male/seed2_12/female_fs1 --seedfz 15

# step1 (cat_fs1) and step2 (2cat, 2dog and 2wild)
python calc_temporalloss.py --img_path=/opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results/celeba-hq/female2male/seed2_12 --save_flow 0 --interval=16
python calc_temporalloss.py --img_path=/opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results/celeba-hq/female2male/seed2_12/female_fs1 --save_flow 0 --interval=16
# stargan-v2 (2cat, 2dog and 2wild)
python calc_temporalloss.py --img_path=/opt/data/private/senmao/stargan-v2-transstep1/expr/results/celeba-hq/female2male/seed2_12/2starganv2 --save_flow 0 --interval=16