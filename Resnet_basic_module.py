import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# combine small resnet block by this function and make one big module
def Resnet_block(input_before_module,strides_1,strides_2,strides_3,conv_dim):
    Layer_block_1 = small_block(input_before_module,strides_1,conv_dim[0])
    Layer_block_2 = small_block(Layer_block_1,strides_2,conv_dim[1])
    Layer_block_3 = small_block(Layer_block_2,strides_3,conv_dim[2])

    #shorten the output
    output = input_before_module + Layer_block_3
    return output


#batch normalization
def batch_normal(input):
    chanel_size = [input.get_shape().as_list()[3]]
    print(chanel_size)
    print(input.shape.as_list())

    #mean and valiance of input of this layer
    mean_batch,Valiance_batch = tf.nn.moments(input,[0,1,2])

    #learning rate
    variance_epsilon = 0.00001

    # set up initial parametor
    g_1 = tf.Variable(tf.truncated_normal(chanel_size, stddev=0.1))
    b_1 = tf.Variable(tf.truncated_normal(chanel_size, stddev=0.1))

    #return batch normalization resut for relu
    For_relu = tf.nn.batch_normalization(input,mean_batch,Valiance_batch,b_1,g_1,variance_epsilon)
    print(For_relu)
    return For_relu


#Relu function and conv layer
def Relu_conv(input,strides,conv_dim):
    #return relu
    For_conv = tf.nn.relu(input)

    #set initial parametor for conv layer
    p_conv = tf.Variable(tf.truncated_normal(conv_dim, stddev=0.1))


    print(For_conv)
    print(p_conv)

    #return conv layer output
    return tf.nn.conv2d(For_conv,p_conv,strides,padding='SAME')

#combine batch normal and conv layer
def small_block(input,strides,conv_dim):
    print(input.shape)
    print(conv_dim)
    return Relu_conv(batch_normal(input),strides,conv_dim)
