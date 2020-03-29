
# coding: utf-8

# In[ ]:


import keras.models
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta, adam
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import regularizers
from keras.constraints import maxnorm
import matplotlib.pyplot as plt


# In[ ]:


X = np.load('influ_selected_model2_train_input.npy')
Y = np.load('influ_selected_model2_train_outputdim2.npy')


# In[ ]:


#X_test = np.load("/Users/TN/Desktop/influ_selected_model2_test_input.npy")
#Y_test = np.load("/Users/TN/Desktop/influ_selected_model2_test_outputdim2.npy")


# In[ ]:


K.image_data_format()


# In[ ]:


conv_layer = [    
    # convolution and then pooling
    Conv2D(20, (7, 7), input_shape=(1024,1360,1), name='first_conv_layer',padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(10, 10), padding='valid'),

    # convolution and then pooling
    Conv2D(25, (5, 5), name='second_conv_layer', padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(6, 6), padding='valid'),
    
    # convolution and then pooling
    Conv2D(30, (3, 3), name='third_conv_layer', padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(6, 6), padding='valid')
]

fc_layer = [
    # flatten and connect with three fully connected layer
    Flatten(),
    Dense(100, name='fc_layer_100_1'),
    Activation('sigmoid'),
    Dense(100, name='fc_layer_100_2',kernel_constraint= maxnorm(1.)),
    Activation('sigmoid'),
    Dense(100, name='fc_layer_100_3',kernel_regularizer=regularizers.l2(0.01)),
    Activation('sigmoid'),
 
    # conneted with smaller fully connected layer
    # with the same number of neurons as the number of classes
    Dense(2, name='fc_layer_2'),
    Activation('softmax')
]


# In[ ]:


model = Sequential(conv_layer + fc_layer)
model.compile(loss="binary_crossentropy",
              optimizer=adam(lr=0.0001),
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


filepath="influ_0816_filter-{epoch:02d}-acc_{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]


# In[ ]:


history = model.fit(X, Y, batch_size=15, epochs=1200, callbacks=callbacks_list, verbose=1, validation_split=0.2, shuffle=True)


# In[ ]:


model.save_weights('0816_influ_filter_model2.h5')
model.save('0816_influ_filter_model_2')


# In[ ]:


plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("model 2 loss")

plt.ylabel("loss")

plt.xlabel("epoch")

plt.legend(["train","test"],loc="upper left")

plt.savefig("model_2_loss")

plt.savefig("model_2_loss.pdf")

plt.close('all')

# In[ ]:


plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("model 2 acc")

plt.ylabel("acc")

plt.xlabel("epoch")

plt.legend(["train","test"],loc="upper left")

plt.savefig("model_2_acc")

plt.savefig("model_2_acc.pdf")

plt.close('all')




plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("model 2 ")

plt.ylabel("loss/acc")

plt.xlabel("epoch")

plt.legend(["train_loss","test_loss","train_acc","test_acc"],loc="upper left")

plt.savefig("model_2")

plt.savefig("model_2.pdf")

plt.close('all')




