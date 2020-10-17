# NAIC_AI_Plus
AI+无线通信

## Encoder

使用两层3x3CNN提取特征。

## Decoder

使用多层CNN+残差连接对提取到的特征进行重构。

## Trainer 

优化器使用AdamW，学习率线性下降，训练到450个epoch。

