# Paddle_3DResNet_UCF-101
百度论文复现_3DResNet

首先感谢百度能够发布论文复现营这种优秀的课程，感谢百度提供的算力卡、各位老师精彩的讲解还有班主任芮芮每天雷打不动的鼓励
这篇程序是根据百度 PaddlePaddle 的论文复现营仿写、复现出来的。主要是使用  Paddle  的库。
参加论文复现营的初衷目的是跟随课程想学东西，至于最终的奖励自己现在是不敢想的，
小目标就是希望能够达到 领取结业证书 的条件，算是自己对这一个月努力的一个交代。
已经完成了三次实践作业，就差最后跑通论文代码。


现在论文代码也已经能够跑通，自己已经完成10次迭代的训练，
训练集结果： Loss at epoch 9 step 73: [4.182964], acc: [0.0859375]
            Final loss: [4.182964]
测试集结果：验证集准确率为:0.015331747010350227


# 下面是在 Notebook 中执行的所有代码：

## 1. 解压数据集
!unzip -q -o /home/aistudio/data/data11460/UCF-101.zip -d data
print("数据集解压完成")

## 2. 将视频进行解压，分解为一张张的图片，存储在对应的_jpg文件夹下
## 源路径 data/data/UCF-101     目标路径  lyj/UCF-101_jpg
!python avi2jpg.py

## 3. 根据 trainlist01.txt 和 testlist01.txt   划分 训练集和测试集 ,  并生成对应的pkl文件(pkl文件很小，只是记录了名称)
## pkl 文件中存储的每一条记录，记录的是图片的路径:    视频类名/视频名/图片名
!python jpg2pkl.py

## 4. 针对指定的pkl文件夹(训练、测试) ,   生成数据列表:train.list、test.list
!python data_list_gener.py

## 5. 执行训练过程
!python train.py --use_gpu True --epoch 65

## 6. 执行测试过程
!python test.py --weights 'checkpoints_models/resnet_model' --use_gpu True 


## 下面是10次迭代最后一部分的展示：
Loss at epoch 9 step 58: [4.2755785], acc: [0.0625]
Loss at epoch 9 step 59: [4.1787696], acc: [0.0625]
Loss at epoch 9 step 60: [4.2209196], acc: [0.03125]
Loss at epoch 9 step 61: [4.141432], acc: [0.0625]
Loss at epoch 9 step 62: [4.247321], acc: [0.0625]
Loss at epoch 9 step 63: [4.2858725], acc: [0.0625]
Loss at epoch 9 step 64: [4.230678], acc: [0.0546875]
Loss at epoch 9 step 65: [4.253533], acc: [0.0859375]
Loss at epoch 9 step 66: [4.1572843], acc: [0.0546875]
Loss at epoch 9 step 67: [4.2979145], acc: [0.03125]
Loss at epoch 9 step 68: [4.135331], acc: [0.0546875]
Loss at epoch 9 step 69: [4.252095], acc: [0.078125]
Loss at epoch 9 step 70: [4.3616824], acc: [0.015625]
Loss at epoch 9 step 71: [4.164705], acc: [0.078125]
Loss at epoch 9 step 72: [4.233446], acc: [0.0703125]
Loss at epoch 9 step 73: [4.182964], acc: [0.0859375]
Final loss: [4.182964]


## 下面是测试的结果展示：
eval.py 中的参数 args :  Namespace(batch_size=1, config='configs/resnet.txt', filelist=None, infer_topk=1, log_interval=1, model_name='resnet', save_dir='checkpoints_models', use_gpu=True, weights='checkpoints_models/resnet_model')
[INFO: test.py:  133]: Namespace(batch_size=1, config='configs/resnet.txt', filelist=None, infer_topk=1, log_interval=1, model_name='resnet', save_dir='checkpoints_models', use_gpu=True, weights='checkpoints_models/resnet_model')
[INFO: config.py:   55]: ---------------- Valid Arguments ----------------
[INFO: config.py:   57]: MODEL:
[INFO: config.py:   59]:     name:resnet
[INFO: config.py:   59]:     format:pkl
[INFO: config.py:   59]:     num_classes:101
[INFO: config.py:   59]:     seg_num:1
[INFO: config.py:   59]:     seglen:16
[INFO: config.py:   59]:     image_mean:[0.485, 0.456, 0.406]
[INFO: config.py:   59]:     image_std:[0.229, 0.224, 0.225]
[INFO: config.py:   59]:     num_layers:50
[INFO: config.py:   57]: TRAIN:
[INFO: config.py:   59]:     epoch:1
[INFO: config.py:   59]:     short_size:240
[INFO: config.py:   59]:     target_size:112
[INFO: config.py:   59]:     num_reader_threads:1
[INFO: config.py:   59]:     buf_size:1024
[INFO: config.py:   59]:     batch_size:128
[INFO: config.py:   59]:     use_gpu:True
[INFO: config.py:   59]:     num_gpus:1
[INFO: config.py:   59]:     filelist:./lyj/train.list
[INFO: config.py:   59]:     learning_rate:0.01
[INFO: config.py:   59]:     learning_rate_decay:0.1
[INFO: config.py:   59]:     l2_weight_decay:0.0001
[INFO: config.py:   59]:     momentum:0.9
[INFO: config.py:   59]:     total_videos:80
[INFO: config.py:   57]: VALID:
[INFO: config.py:   59]:     seg_num:7
[INFO: config.py:   59]:     short_size:240
[INFO: config.py:   59]:     target_size:112
[INFO: config.py:   59]:     num_reader_threads:1
[INFO: config.py:   59]:     buf_size:1024
[INFO: config.py:   59]:     batch_size:1
[INFO: config.py:   59]:     filelist:./lyj/test.list
[INFO: config.py:   60]: -------------------------------------------------

W0905 19:17:46.177613  7921 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.0
W0905 19:17:46.182003  7921 device_context.cc:260] device: 0, cuDNN Version: 7.3.
验证集准确率为:0.015331747010350227
