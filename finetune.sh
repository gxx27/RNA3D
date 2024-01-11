python finetune.py --output_dir ./result_config \
    --dataset_name test \
    --train_file './msa_folder/data.pkl' \
    --remove_unused_columns False \
    --do_train True \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
    --num_train_epochs 100 \
    --learning_rate 2e-5 \
    --weight_decay 1e-2 \
    --evaluation_strategy 'epoch' \
    --save_strategy 'epoch' \
    --fp16 True \
    --prediction_loss_only True \
    # --logging_strategy 'epoch' \
    # --logging_dir './logs' \
    # --num_alignments 3
