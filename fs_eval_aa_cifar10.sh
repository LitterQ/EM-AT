export PYTHONPATH=./:$PYTHONPATH
model_dir=./models/EM_AT/
CUDA_VISIBLE_DEVICES=0 python fs_eval_aa_cifar10.py \
    --model_dir=$model_dir \
    --init_model_pass=200 \
    --attack=True \
    --attack_method_list=fgsm \
    --dataset=cifar10\
    --batch_size_test=200 \
    --resume
