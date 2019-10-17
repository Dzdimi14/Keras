from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

#### https://www.tensorflow.org/guide/keras/overview #####

#making MNIST dataset

mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
#converts samples to integers
x_train, x_test = x_train / 255.0, x_test / 255.0


#multilayered perceptron
model = tf.keras.Sequential()

model.add(layers.Flatten(input_shape = (28, 28)))

#64 node layer 1
model.add(layers.Dense(64, activation= 'relu')) #activation = activation fn, relu = simple

#droupout layer
model.add(layers.Dropout(0.2))

#64 node layer 2
model.add(layers.Dense(64, activation= 'relu'))


#10node softmax layer with 10 outputs
model.add(layers.Dense(10, activation = 'softmax'))

#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers for info on optimizers

if __name__ == "__main__":
    
    
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    #saving weights for progressive training 
    model.save_weights('./weights/my_model')
    
    #restore model state
    #requires model with same architecture
    model.load_weights('./weights/my_model')
    
    #training model
    model.fit(x_train, y_train, epochs = 5)
    model.evaluate(x_test, y_test, verbose = 2)
    
    
    #TODO take in input and print something, visualize outputs
    
    
     





