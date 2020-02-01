from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import Resnet_basic_module as Rb
import sys

#combine all resnet layer
def Resnet_layer(input,strides,conv_dim):

    length = len(strides)/3
    if float.is_integer(length):
        length = int(length)



        print(length)
        resnet_input_and_output = input

        for i in range(length):
            strides_1 = strides[i]
            strides_2 = strides[i+1]
            strides_3 = strides[i+2]
    #return first resnet module output
            output = Rb.Resnet_block(resnet_input_and_output,strides_1,strides_2,strides_3,conv_dim[i])

            resnet_input_and_output = output

    else:
        sys.exit()

    return resnet_input_and_output
