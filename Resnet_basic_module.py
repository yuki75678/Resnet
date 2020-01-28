import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def Resnet_block(input_before_module,strides_1,strides_2,strides_3,conv_dim):
    Layer_block_1 = small_block(input_before_module,strides_1,conv_dim[0])
    Layer_block_2 = small_block(Layer_block_1,strides_2,conv_dim[1])
    Layer_block_3 = small_block(Layer_block_2,strides_3,conv_dim[2])
    output = input_before_module + Layer_block_3


    return output

def batch_normal(input):
    chanel_size = [input.get_shape().as_list()[3]]
    print(chanel_size)
    print(input.shape.as_list())

    mean_batch,Valiance_batch = tf.nn.moments(input,[0,1,2])

    variance_epsilon = 0.00001


    g_1 = tf.Variable(tf.truncated_normal(chanel_size, stddev=0.1))
    b_1 = tf.Variable(tf.truncated_normal(chanel_size, stddev=0.1))

    For_relu = tf.nn.batch_normalization(input,mean_batch,Valiance_batch,b_1,g_1,variance_epsilon)
    print(For_relu)
    return For_relu

def Relu_conv(input,strides,conv_dim):
    For_conv = tf.nn.relu(input)
    p_conv = tf.Variable(tf.truncated_normal(conv_dim, stddev=0.1))


    print(For_conv)
    print(p_conv)

    return tf.nn.conv2d(For_conv,p_conv,strides,padding='SAME')

def small_block(input,strides,conv_dim):
    print(input.shape)
    print(conv_dim)
    return Relu_conv(batch_normal(input),strides,conv_dim)
