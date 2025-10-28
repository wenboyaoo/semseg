# 用 DINOv3 和 PSPNet 实现的图像语义分割算法

## 介绍
对原 PSPNet 进行了以下修改：
1. 根据 DINOv3 preprocessor 修改了数据预处理流程
2. 用 HuggingFace Transformers 加载的预训练模型 facebook/dinov3-convnext-tiny-pretrain-lvd1689m 作为骨干网络
3. 使用 FPN 降低输出步幅并提高通道数
4. 利用 DINOv3 的 CLS token 使用 FiLM 调节 PPM 中的通道

## 性能
- 数据集：PASCAL VOC 2012 Augmented
  - 训练集：10582张
  - 验证集：1449张
- 超参数：见 conig.yaml 
- 测试结果：mIoU/mAcc/allAcc 0.8116/0.8960/0.9582
- 训练和测试显卡: RTX 4090 x1
- 训练时长（含验证）：1 小时 39 分钟
  
## 复现
1. 克隆仓库
2. 安装依赖项
3. 将数据集放在 dataset/ 路径下
4. 登陆 HuggingFace 账户
5. 运行训练脚本 tool/train.sh
   
## 原始引用

```
@misc{semseg2019,
  author={Zhao, Hengshuang},
  title={semseg},
  howpublished={\url{https://github.com/hszhao/semseg}},
  year={2019}
}
@inproceedings{zhao2017pspnet,
  title={Pyramid Scene Parsing Network},
  author={Zhao, Hengshuang and Shi, Jianping and Qi, Xiaojuan and Wang, Xiaogang and Jia, Jiaya},
  booktitle={CVPR},
  year={2017}
}
@inproceedings{zhao2018psanet,
  title={{PSANet}: Point-wise Spatial Attention Network for Scene Parsing},
  author={Zhao, Hengshuang and Zhang, Yi and Liu, Shu and Shi, Jianping and Loy, Chen Change and Lin, Dahua and Jia, Jiaya},
  booktitle={ECCV},
  year={2018}
}
```
