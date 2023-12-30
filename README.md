# RND 3D prediction project
目前进度：代码能跑通，但是输入数据映射过程有问题

主要改动：msadata.py collator按照原来的，MSADataSet自己写的

使用tokenizer来掩码那一块我觉得很怪，不过效果其实还好，由于T5没有\mask，所以我用<extra_id0>来代替，详见tokenizer.additional_special_tokens

finetune.py是运行的主函数 finetune.sh是运行脚本