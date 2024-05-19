# import numpy as np
# import os
# import tensorflow as tf
# import cv2
# from sklearn.model_selection import train_test_split

# my_dic = {
#     'Five Finger': 0,
#     'Scissors': 1,
#     'Rock': 2,
#     'Taxi': 3,
#     'Four Finger': 4,
#     'Devil': 5,
#     'Nice': 6,
#     'PePe': 7,
#     'Three Finger': 8,
#     'One Finger': 9,
#     'Stop': 10
# }
#
# def load_images_from_dir(directory, target_size=(64, 64)):
#     my_images = []
#     my_labels = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.png'):
#                 ih = os.path.join(root, file)
#                 label = my_dic[os.path.basename(root)]
#                 img = cv2.imread(ih)
#                 img = cv2.resize(img, target_size)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 my_images.append(img)
#                 my_labels.append(label)
#     return np.array(my_images), np.array(my_labels)
#
#
# images, labels = load_images_from_dir(R'C:\python\acquisitions')
#
# X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.4, random_state=42)
#
# np.save('X_train.npy', X_train)
# np.save('X_test.npy', X_test)
# np.save('y_train.npy', y_train)
# np.save('y_test.npy', y_test)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

img = "1-color.jpg"
img = np.uint8(np.dot(img, [0.33, 0.33, 0.34]))
print(img.shape)
# plt.imshow(img, cmap ='gray')
# plt.show()

l1 = 28 / img.shape[0]
l2 = 28 / img.shape[1]
new_img = np.zeros((28, 28))
for x in range(28):
    for y in range(28):
        new_img[x][y] = img[int(x / l1)][int(y / l2)]
new_img = 255 - new_img
# plt.imshow(new_img, cmap ='gray')
# plt.show()
z = np.array(new_img)
z = z.reshape((1, 28, 28))

# Load the data
x_train = np.load('X_train.npy')
x_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Normalize the data
x_train = (x_train / 255.0)
x_test = x_test / 255.0

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(11, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
predictions = model.predict([z])
# print(f'true value: {y_train[n]}')
print(predictions[0])
print(f'predicted value: {np.argmax(predictions[0])}')
plt.title()
plt.imshow(z[0], cmap='gray')
plt.show()
