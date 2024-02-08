# 100
# No Model
python test.py 100 --n_iter 100 --seed 0 --device cpu;
python test.py 100 --n_iter 100 --seed 1 --device cpu;
python test.py 100 --n_iter 100 --seed 2 --device cpu;

# RL
python train.py 100 --seed 0 --device cuda:1;
python train.py 100 --seed 1 --device cuda:1;
python train.py 100 --seed 2 --device cuda:1;
python test.py 100 -p "../pretrained/smtwtp/100/smtwtp100_sd0_rl_fromscratch/20.pt" --n_iter 100 --seed 0 --device cuda:1;
python test.py 100 -p "../pretrained/smtwtp/100/smtwtp100_sd1_rl_fromscratch/20.pt" --n_iter 100 --seed 0 --device cuda:1;
python test.py 100 -p "../pretrained/smtwtp/100/smtwtp100_sd2_rl_fromscratch/20.pt" --n_iter 100 --seed 0 --device cuda:1;

# GFN
python train.py 100 --gfn --seed 0 --device cuda:1;
python train.py 100 --gfn --seed 1 --device cuda:1;
python train.py 100 --gfn --seed 2 --device cuda:1;
python test.py 100 -p "../pretrained/smtwtp/100/smtwtp100_sd0_gfn30_fromscratch/20.pt" --gfn --n_iter 100 --seed 0 --device cuda:1;
python test.py 100 -p "../pretrained/smtwtp/100/smtwtp100_sd1_gfn30_fromscratch/20.pt" --gfn --n_iter 100 --seed 0 --device cuda:1;
python test.py 100 -p "../pretrained/smtwtp/100/smtwtp100_sd2_gfn30_fromscratch/20.pt" --gfn --n_iter 100 --seed 0 --device cuda:1;

# GFN-LS
python train.py 100 --gfn --train_w_ls --seed 0 --device cuda:2;
python test.py 100 -p "../pretrained/smtwtp/100/smtwtp100_sd0_gfn30-ls_fromscratch/20.pt" --gfn --train_w_ls --n_iter 100 --seed 0 --device cuda:2;
python train.py 100 --gfn --train_w_ls --seed 1 --device cuda:3;
python train.py 100 --gfn --train_w_ls --seed 2 --device cuda:3;
python test.py 100 -p "../pretrained/smtwtp/100/smtwtp100_sd1_gfn30-ls_fromscratch/20.pt" --gfn --train_w_ls --n_iter 100 --seed 0 --device cuda:3;
python test.py 100 -p "../pretrained/smtwtp/100/smtwtp100_sd2_gfn30-ls_fromscratch/20.pt" --gfn --train_w_ls --n_iter 100 --seed 0 --device cuda:3;
