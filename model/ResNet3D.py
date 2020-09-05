# 3D-ResNet
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv3D, BatchNorm, Linear 
from model import Pool3D 

## resnet 的卷积块
class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_channels, num_filters, filter_size, stride=1, groups=1, act=None):
        super(ConvBNLayer,self).__init__(name_scope)
        # 卷积操作
        self._conv = Conv3D( num_channels=num_channels, num_filters=num_filters, filter_size=filter_size,
                            stride=stride, padding=(filter_size-1)//2, groups=groups, act=None, bias_attr=False)
        # 残差块中经典的操作，一个卷积后面跟一个batch_norm
        self._batch_norm = BatchNorm(num_filters,act=act)

    # 前向传播
    def forward(self,inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y

## resnet 中的残差块
class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock,self).__init__(name_scope)
        # 调用了上面自定义的卷积块
        self.conv0 = ConvBNLayer( self.full_name(), num_channels=num_channels, num_filters=num_filters, filter_size=1, act='relu')
        self.conv1 = ConvBNLayer( self.full_name(), num_channels=num_filters, num_filters=num_filters, filter_size=3, stride=stride, act='relu')
        self.conv2 = ConvBNLayer( self.full_name(), num_channels=num_filters, num_filters=num_filters * 4, filter_size=1, act=None)
        if not shortcut:
            self.short = ConvBNLayer( self.full_name(), num_channels=num_channels, num_filters=num_filters * 4, filter_size=1, stride=stride)
        self.shortcut = shortcut
        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1) 
        if self.shortcut:
            short = inputs              # 短路处理，相当于没有对图片进行处理
        else:
            short = self.short(inputs)
        y = fluid.layers.elementwise_add(x=short, y=conv2)
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)    

class ResNet3D(fluid.dygraph.Layer):   #定义网络结构，代码补齐
    def __init__(self, name_scope, layers=50, class_dim=102, seg_num=10, weight_devay=None):
        super(ResNet3D, self).__init__(name_scope)

        self.layers = layers
        self.seg_num = seg_num
        supported_layers = [50, 101, 152] 
        assert layers in supported_layers, "supported layers are {} but input layer is {}".format(supported_layers, layers)
        
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        # 在进入残差网络之前，先对图片进行 7*7*7 的卷积和 3*3 最大池化
        # self.conv = ConvBNLayer( self.full_name(), num_channels=3, num_filters=64, filter_size=7, stride=2, act='relu')
        self.conv = Conv3D( num_channels=3, num_filters=64, filter_size=(7,7,7), stride=(1,2,2), padding=( 7 // 2, 3, 3), bias_attr=False)
        self.bn  = BatchNorm( num_channels=64 ,act='relu' )

        # pool3d 没有动态图的实现，自定义Pool3D类 继承layers.Layer，然后传参，构建forward
        self.pool3d_max = Pool3D.Pool3D( pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        # 残差网络部分
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    "bb_%d_%d" % (block,i),
                    BottleneckBlock(
                        self.full_name(),
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block !=0 else 1,
                        shortcut=shortcut
                        )
                    )
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 残差网络之后的 平均池化
        self.pool3d_avg = Pool3D.Pool3D(pool_size=7, pool_type='avg', global_pooling=True)
        

        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        # 最后的 fc 
        self.fc = Linear(input_dim=num_channels, output_dim=class_dim, act='softmax',
                          param_attr=fluid.param_attr.ParamAttr( initializer=fluid.initializer.Uniform(-stdv,stdv)))


    def forward(self, inputs, label=None): 
        out = fluid.layers.reshape(inputs, [-1, inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4]]) # NCDHW    D 代表时间维度
        # 刚开始的  7*7*7 卷积 和 3*3*3 的最大池化
        y = self.conv(out)                      # 开始进行普通的resnet结构
        y = self.bn(y)
        y = self.pool3d_max(y)
        # 残差块部分
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)

        y = self.pool3d_avg(y)  # 残差块之后的平均池化  # 得到的是 batch*每个视频取多少帧 ,1, 1, N(通道数)   一个四维
        out = fluid.layers.reshape(x=y, shape=[-1, y.shape[1]])         
        y = self.fc(out)        # 最后的 fc
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label, k=1)
            return y,acc
        else:
            return y

if __name__ == '__main__':
    with fluid.dygraph.guard():

        network = ResNet3D('resnet', 50)
        img = np.zeros([1, 10, 3, 224, 224]).astype('float32')  # NDCHW
        img = fluid.dygraph.to_variable(img)
        
        outs = network(img).numpy()
        print('ResNet3D.py 中 网络输出为 ：outs', outs)

