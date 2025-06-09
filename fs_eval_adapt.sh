export PYTHONPATH=./:$PYTHONPATH
model_dir=./models/EM_AT/
CUDA_VISIBLE_DEVICES=0 python fs_eval_adapt.py \
    --model_dir=$model_dir \
    --init_model_pass=200 \
    --attack=True \
    --attack_method_list=fgsm-pgd-cw \
    --dataset=cifar10 \
    --batch_size_test=200 \
    --resume
