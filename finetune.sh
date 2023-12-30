CUDA_VISIBLE_DEVICES=0,1 python finetune.py --output_dir ./test_config \
    --dataset_name test \
    --train_file './dataset/test.fasta' \
    --remove_unused_columns False \
    # --do_train \
    # --overwrite_output_dir \
    # --per_device_train_batch_size 10 \
    # --fp16 True \
    # --num_train_epochs 10 \
    # --learning_rate 0.1