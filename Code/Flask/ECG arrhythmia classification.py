#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the nueral network libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[2]:


#import the cnn layers
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten


# #image preprocessing (or) data Augmentation

# In[3]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[4]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)


# In[5]:


test_datagen=ImageDataGenerator(rescale=1./255)


# In[6]:


x_train =train_datagen.flow_from_directory("C:/Users/haris/OneDrive/Desktop/classification-of-Arrhythmia/Code/ECG-Dataset/Dataset/train",target_size=(64,64),batch_size=32,class_mode="categorical")


# In[7]:


x_test = test_datagen.flow_from_directory("C:/Users/haris/OneDrive/Desktop/classification-of-Arrhythmia/Code/ECG-Dataset/Dataset/test",target_size=(64,64),batch_size=32,class_mode="categorical")


# In[8]:


x_train.class_indices


# In[9]:


#initialize the model
model=Sequential()


# In[10]:


#convolutional model
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation="relu"))
#here 32 indiates no.of feature detectors and(3,3) is feature detector size


# In[11]:


#pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))


# In[12]:


#flatten layer
model.add(Flatten())


# #hidden layers

# In[13]:


model.add(Dense(units=200,activation="relu",kernel_initializer="random_uniform"))


# In[14]:


model.add(Dense(units=300,activation="relu",kernel_initializer="random_uniform"))


# #output layer
# 

# In[15]:


model.add(Dense(units=6,activation="softmax",kernel_initializer="random_uniform"))


# #compile model

# In[16]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# #train your  model

# In[17]:


tr=model.fit_generator(x_train,steps_per_epoch=480,epochs=25,validation_data=x_test,validation_steps=10)
#steps_per_epoch =>total trainging images/batch size
#validation_steps=>total testing images/batch size


# In[18]:


tr.history


# #to save the best accuracy got in the epoch we willn use this callback and checkpoint

# In[19]:


from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("best_model_{epoch:02d}.h5",monitor="val_accuracy",save_best_only=True,mode="Max")
lr = model.fit_generator(x_train,steps_per_epoch=480,callbacks=[checkpoint],validation_steps=10)


# #saving the model

# In[20]:


#for storing temporary
model.save('ECG.h5')


# In[21]:


#for storing permanent in drive
model.save("C:/Users/haris/OneDrive/Desktop/classification-of-Arrhythmia/Code/ECG-Dataset/ECG.h5")


# In[22]:


losses=tr.history['loss']
accuracy=tr.history['accuracy']
epochs=list(range(1,26))


# In[23]:


tr.history['loss'][5]


# In[28]:


import matplotlib.pyplot as plt
plt.plot(epochs,losses)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# In[29]:


plt.plot(epochs,accuracy)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()


# In[26]:


losses=tr.history['loss']
accuracy=tr.history['accuracy']
val_accuarcy=tr.history['val_accuracy']
epochs=list(range(1,26))


# In[27]:


plt.plot(epochs,val_accuarcy)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()

