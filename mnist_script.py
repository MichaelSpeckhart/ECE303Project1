# %% [markdown]
# <img src="./images/DLI_Header.png" style="width: 400px;">

# %% [markdown]
# ## Objectives

# %%
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import 
from tensorflow.keras.layers import Dense
import time

# %%
# the data, split between train and validation sets
start_time = time.time()
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

# %% [markdown]
# Using [Matplotlib](https://matplotlib.org/), we can render one of these grayscale images in our dataset:

# %%

image = x_train[0]
plt.imshow(image, cmap='gray')

# %% [markdown]
# In this way we can now see that this is a 28x28 pixel image of a 5. Or is it a 3? The answer is in the `y_train` data, which contains correct labels for the data. Let's take a look:

# %%
x_train = x_train.reshape(60000, 784)
x_valid = x_valid.reshape(10000, 784)

# %%
x_train = x_train / 255
x_valid = x_valid / 255 

# %%
num_categories = 10

y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)

# %% [markdown]
# Here are the first 10 values of the training labels, which you can see have now been categorically encoded:

# %%

model = Sequential()

# %%
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = 10, activation='softmax'))

# %%
model.summary()

# %% [markdown]
# ### Compiling the Model

# %%
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# %%
history = model.fit(
    x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid)
)

end_time = time.time()

total_time = end_time - start_time


import matplotlib.pyplot as plt
plt.figure(figsize=(16, 10))

num_images = 16
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]
    pred = model.predict(row)
    
    image = row.reshape(28,28)
    plt.subplot(1, num_images, i+1)
    plt.title(f"Actual: {label}, Pred: {pred}")
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    
plt.savefig(f"MNIST_predictions_{total_time}s.pdf", dpi = 800)
