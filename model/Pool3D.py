# from paddle.fluid.layers import pool3d
from paddle.fluid.dygraph import layers
from paddle.fluid.layers import utils
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.data_feeder import check_variable_and_dtype

class Pool3D( layers.Layer):
    def __init__(self,
                 pool_size=-1,
                 pool_type="max",
                 pool_stride=1,
                 pool_padding=0,

                 global_pooling=False,
                 use_cudnn=True,
                 ceil_mode=False,
                 exclusive=True):
        if pool_type not in ["max", "avg"]:
            raise ValueError(
                "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
                str(pool_type))

        if global_pooling is False and pool_size == -1:
            raise ValueError(
                "When the global_pooling is False, pool_size must be passed "
                "and be a valid value. Received pool_size: " + str(pool_size))

        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")

        super(Pool3D, self).__init__()

        self._pool_type = pool_type
        self._pool_size = utils.convert_to_list(pool_size, 3, 'pool_size')
        self._pool_padding = utils.convert_to_list(pool_padding, 3, 'pool_padding')
        self._pool_stride = utils.convert_to_list(pool_stride, 3, 'pool_stride')
        self._global_pooling = global_pooling
        self._use_cudnn = use_cudnn
        self._ceil_mode = ceil_mode
        self._exclusive = exclusive
        self._l_type = 'pool3d'

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('pooling_type', self._pool_type, 'ksize', self._pool_size,
                    'global_pooling', self._global_pooling, 'strides',
                    self._pool_stride, 'paddings', self._pool_padding,
                    'use_cudnn', self._use_cudnn, 'ceil_mode', self._ceil_mode,
                    'use_mkldnn', False, 'exclusive', self._exclusive)
            return core.ops.pool3d(input, *attrs)

        check_variable_and_dtype(input,'input',['int8','uint8','float16','float32','float64'],'Pool3D')

        attrs = {
            "pooling_type": self._pool_type,
            "ksize": self._pool_size,
            "global_pooling": self._global_pooling,
            "strides": self._pool_stride,
            "paddings": self._pool_padding,
            "use_cudnn": self._use_cudnn,
            "ceil_mode": self._ceil_mode,
            "use_mkldnn": False,
            "exclusive": self._exclusive,
        }
        inputs = {"X": [input]}

        pool_out = self._helper.create_variable_for_type_inference(self._dtype)

        self._helper.append_op(
            type=self._l_type,
            inputs={"X": input},
            outputs={"Out": pool_out},
            attrs=attrs)
        return pool_out


