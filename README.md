# GPLinker_pytorch
GPLinker_pytorch

# 介绍
这是pytorch版本的GPLinker，该代码执行效率可能有点慢，主要瓶颈应该在datagenerator部分，之后可能会使用dataloader加载数据。
本仓库主要参考了[苏神博客](https://kexue.fm/archives/8888)和[他的keras版本代码](https://github.com/bojone/bert4keras/tree/master/examples/task_relation_extraction_gplinker.py)

# 依赖
- transformers
- torch

# 运行
```bash
python run.py
```
可修改文件中的参数。
```python    
efficient = False # 是否使用EfficientGlobalpointer
epochs = 20
maxlen = 128
batch_size = 16
weight_decay = 0.01
lr = 3e-5
dict_path = "./chinese-roberta-wwm-ext/vocab.txt" # 预训练模型vocab.txt路径
model_name_or_path = "hfl/chinese-roberta-wwm-ext" # 预训练模型权重路径
```

# 结果
```bash
#Epoch 1 -- f1 : 0.6860628101678229, precision : 0.8071369146660334, recall : 0.5965740639068857
==================================================
#Epoch 2 -- f1 : 0.7958765733219821, precision : 0.8358021409757634, recall : 0.759591523004283
==================================================
#Epoch 3 -- f1 : 0.8108524322855335, precision : 0.836991841410104, recall : 0.7862962556275397
==================================================
#Epoch 4 -- f1 : 0.8135480049539608, precision : 0.798203719357566, recall : 0.8294937959811138
==================================================
#Epoch 5 -- f1 : 0.8235228089080464, precision : 0.8422611530778595, recall : 0.8056000878445156
==================================================
#Epoch 6 -- f1 : 0.825081844489883, precision : 0.8398234919479578, recall : 0.8108487976281985
==================================================
```