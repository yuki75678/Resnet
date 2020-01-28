from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import Resnet_basic_module as Rb


def Resnet_layer(input,strides,conv_dim):

    strides_1 = strides[0]
    strides_2 = strides[1]
    strides_3 = strides[2]

    output_1 = Rb.Resnet_block(input,strides_1,strides_2,strides_3,conv_dim[0])

    strides_4 = strides[3]
    strides_5 = strides[4]
    strides_6 = strides[5]

    output_2 = Rb.Resnet_block(output_1,strides_4,strides_5,strides_6,conv_dim[1])

    return output_2
