# RND 3D prediction project
目前进度：24GB显存无法进行预训练

增加了数据的构造 包括pickle版本的和huggingface.dataset版本的

dataprocess是pickle版本的数据构造 构造出来的.pkl文件大小为5.78G 导入时间约为120s 同时json版本的.json文件大小为14.44G 导入时间约为150s

数据量大小为11709条 遵循原文利用30条MSA数据进行训练的方法 总共的数据量为351270

修改了预训练目标，从掩码学习到生成式预训练