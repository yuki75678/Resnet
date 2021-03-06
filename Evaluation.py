from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import Resnet as Rs
import numpy as np


###model ###
#set values
training_loop = 1000
strides =[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
conv_dim =[[[3,3,1,8],[3,3,8,8],[3,3,8,16]],[[3,3,16,16],[3,3,16,16],[3,3,16,16]]]


#read data set for evaluation
mnist_set = input_data.read_data_sets("data/", one_hot=True)
train_images, train_labels = mnist_set.train.next_batch(50)

test_images = mnist_set.test.images
test_labels = mnist_set.test.labels




#reshape image data for tensor flow
x = tf.compat.v1.placeholder(tf.float32,[None, 784])
image_input = tf.reshape(x,[-1,28,28,1])
tf.compat.v1.summary.image("input_data",image_input,10)


#Resnet layer
Resnet_result = Rs.Resnet_layer(image_input,strides,conv_dim)
Resnet_result_reshape = tf.reshape(Resnet_result,[-1,28*28*16])


#NN layer
w = tf.Variable(tf.truncated_normal([28*28*16, 10], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))
out = tf.nn.softmax(tf.matmul(Resnet_result_reshape,w) + b)



#Caluculating loss
y = tf.placeholder(tf.float32,[None,10])
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(out+ 1e-5),axis=[1]))
#optimizing
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


###test ###
# test models by test data set
correct = tf.equal(tf.argmax(out,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))


###set up  ###
#initializing
init = tf.global_variables_initializer()

#set up config for CPU
config = tf.ConfigProto(log_device_placement=True)
config.graph_options.rewrite_options.layout_optimizer = 2  # RewriterConfig.OFF


### running graph###
#Run tf Graph
with tf.Session(config=config) as sess:
    # initializing
    sess.run(init)

    #train model
    for i in range(training_loop):

        train_images, train_labels = mnist_set.train.next_batch(50)
        sess.run(train_step, feed_dict={x:train_images, y:train_labels})
        steps = i+1

        #test model
        if steps % 10==0:
            accuracy_check = sess.run(accuracy, feed_dict={x:test_images,y:test_labels})
            print('%d step accuracy : %2f' % (steps, accuracy_check ))
