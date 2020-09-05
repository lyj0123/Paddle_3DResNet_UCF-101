import os
# import sys
# import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid

from model import ResNet3D
from reader import KineticsReader
from config import parse_config, merge_configs, print_configs

## 记录跑网络时生成的log文件
logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

## 定义网络训练时需要的一些超参数
def parse_args():   ## parse_args 解析参数，ArgumentParser 参数解析器
    
    parser = argparse.ArgumentParser("Paddle Video train script")
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='resnet',
        help='name of model to train.')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/resnet.txt',                              ### 调用了 configs 文件夹下的 tsn.txt 文件，进行训练和测试的过程 
        help='path to config file of model')
        ########   tsn.txt   中只用到了训练和测试部分，验证和推理部分没有用到

    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.')

    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=False,
        help='default use gpu.')

    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='epoch number, 0 for read from config file')
        
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_models',
        help='directory name to save train snapshoot')

    args = parser.parse_args()
    return args


def train(args):
    # parse config   参数配置
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()    # 是否使用 GPU

    with fluid.dygraph.guard(place):
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))       # vars 函数，返回参数表达式的值
        print_configs( train_config, 'Train configs : ' )

        train_model = ResNet3D.ResNet3D('resnet',train_config['MODEL']['num_layers'],
                                    train_config['MODEL']['num_classes'],
                                    train_config['MODEL']['seg_num'],
                                    0.00002)

        #根据自己定义的网络，声明train_model
        # parameter_list 指明在训练的时候，哪些参数(  在此是 train_model.parameters()  )会被优化
        opt = fluid.optimizer.Momentum(0.001, 0.9, parameter_list=train_model.parameters())

        if args.pretrain:
            # 加载上一次训练的模型，继续训练
            model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/resnet_model')
            train_model.load_dict(model)

        # 创建一个保存模型的路径
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # get reader
        train_config.TRAIN.batch_size = train_config.TRAIN.batch_size # 两边完全一样啊？？？
        # KineticsReader().create_reader()  函数返回值是  batch_size 组 <img, label> 数据        
        train_reader = KineticsReader(args.model_name.upper(), 'train', train_config).create_reader()

        epochs = args.epoch or train_model.epoch_num()
        for i in range(epochs):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')

                ## 获取的img 是一个5维数据：batchbatch_size,提取多少片段(seg_num*seg_len)，通道数，长，宽
                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
        
                label.stop_gradient = True
                
                out, acc = train_model(img, label)
                
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()

                opt.minimize(avg_loss)
                train_model.clear_gradients()
                
                # 隔多少次训练，进行一次输出提示
                if batch_id % 1 == 0:
                    logger.info("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/resnet_model')

        logger.info("Final loss: {}".format(avg_loss.numpy()))
        print("Final loss: {}".format(avg_loss.numpy()))
                


if __name__ == "__main__":
    args = parse_args()     # 调用第一个自定义函数，进行参数配置

    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    train(args)             # 调用训练的函数，执行训练过程



