export PYTHONPATH=./:$PYTHONPATH
model_dir=./models/EM_AT/
mkdir -p $model_dir
CUDA_VISIBLE_DEVICES=0 python fs_main.py \
    --resume \
    --adv_mode='feature_scatter' \
    --lr=0.1 \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --max_epoch=200 \
    --save_epochs=10 \
    --decay_epoch1=60 \
    --decay_epoch2=90 \
    --batch_size_train=120 \
    --dataset=cifar10

