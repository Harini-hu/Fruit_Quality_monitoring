# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# %%
# dataset count
good_bananas = next(os.walk('C:/Users/admin/Desktop/harini/ML/Banana_Good'))[2]
print('Good Bananas: ', len(good_bananas))
bad_bananas = next(os.walk('C:/Users/admin/Desktop/harini/ML/Banana_Bad'))[2]
print('Bad Bananas: ', len(bad_bananas))

# %%
def split_files(src_dir, train_file_count, test_dir, train_dir):
    # Get a list of all files in the source directory
    files = os.listdir(src_dir)

    # Shuffle the list of files
    random.shuffle(files)

    # Move the first `train_file_count` files to the train directory
    for i in range(train_file_count):
        file = files[i]
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(train_dir, file)
        os.rename(src_path, dst_path)

    # Move the remaining files to the test directory
    for file in files[train_file_count:]:
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(test_dir, file)
        os.rename(src_path, dst_path)

# %%
import tensorflow as tf
import os
labels = ['bad','good']
# Setup paths
train_path = 'C:/Users/admin/Desktop/harini/ML/Updated/train'
test_path = 'C:/Users/admin/Desktop/harini/ML/Updated/test'
# Define batch size
batch_size = 32
image_size = (240, 240)

# Define function to preprocess images
def preprocess_image(image, label):
    # Resize and rescale images
    image = tf.image.resize(image, image_size)
    image /= 255.0  # Rescale to [0,1]
    return image, label

# Create training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
)

# Create validation dataset
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
)

# Prefetching, caching, and shuffling
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# Preprocess images
train_dataset = train_dataset.map(preprocess_image)
val_dataset = val_dataset.map(preprocess_image)


# %%
def print_files(dir_path):
    # Get a list of all files in the directory
    files = os.listdir(dir_path)

    print(f'Found {len(files)} files')
    
    # Print the files
    for file in files:
        print(file)

# %%
import matplotlib.pyplot as plt

# Get a batch of images and labels from the training dataset
sample_images, sample_labels = next(iter(train_dataset))

# Plot the sample images
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(15, 12))

for i in range(3):
    for j in range(4):
        idx = i * 4 + j
        label = labels[sample_labels[idx].numpy()]
        ax[i, j].set_title(f"{label}")
        ax[i, j].imshow(sample_images[idx])
        ax[i, j].axis("off")

plt.suptitle("Sample Training Images", fontsize=21)
plt.show()
# %%
# Define model and compile
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(240, 240, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Assuming 2 classes for classification
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset,
                    epochs=12,
                    validation_data=val_dataset)

# %%
model.summary()
# %%
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import keras
print("Keras version:", keras.__version__)
# %%
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
)
# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy}")

# %%
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Generate confusion matrix
val_labels = []
val_preds = []
for images, labels in val_dataset:
    preds = model.predict(images)
    val_labels.extend(labels.numpy())
    val_preds.extend(np.argmax(preds, axis=1))

val_labels = np.array(val_labels)
val_preds = np.array(val_preds)
# Compute the confusion matrix
conf_matrix = confusion_matrix(val_labels, val_preds)
# Map the confusion matrix to true positives, false positives, true negatives, and false negatives
tn, fp, fn, tp = conf_matrix.ravel()
# Simplified confusion matrix
simplified_conf_matrix = np.array([[tn, fp],
                                   [fn, tp]])
# Plot simplified confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(simplified_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['True Negative', 'True Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.show()
# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
# Generate ROC curve
val_labels = []
val_preds = []

for images, labels in val_dataset:
    preds = model.predict(images)
    val_labels.extend(labels.numpy())
    val_preds.extend(preds)

val_labels = np.array(val_labels)
val_preds = np.array(val_preds)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(2):  # Assuming binary classification
    fpr[i], tpr[i], _ = roc_curve(val_labels == i, val_preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(10, 8))

for i in range(2):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
# %%
from sklearn.metrics import precision_score, recall_score

# Calculate precision and recall for each class
precision = precision_score(val_labels, np.argmax(val_preds, axis=1), average=None)
recall = recall_score(val_labels, np.argmax(val_preds, axis=1), average=None)

# Display precision and recall values in a table
print("Class\tPrecision\tRecall")
for i in range(2):  # Assuming binary classification
    print(f"{i}\t{precision[i]:.2f}\t\t{recall[i]:.2f}")

# %%
# Save the entire model with the HDF5 extension
model.save('my_model.h5')

# You can also save the model in the native Keras format
model.save('my_model.keras')

# %%
import cv2
import numpy as np
import tensorflow as tf
import time

# Load the model
loaded_model = tf.keras.models.load_model('my_model.h5')

# Access the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Press 'q' to close the camera feed and exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if it's time to predict
    if time.time() % 5 < 1:
        # Resize the frame to match the input size of the model
        resized_frame = cv2.resize(frame, (240, 240))

        # Normalize the frame
        normalized_frame = resized_frame / 255.0

        # Perform inference
        predictions = loaded_model.predict(np.expand_dims(normalized_frame, axis=0))

        # Get the predicted class
        predicted_class = np.argmax(predictions)

        # Determine the label
        label = "Bad" if predicted_class == 0 else "Good"
        label = f'Quality: {label}'

        # Display the label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with the label
        cv2.imshow('Predicted Image', frame)

        # Wait for a moment before proceeding to the next prediction
        time.sleep(1)

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
# %
import cv2
import numpy as np
import tensorflow as tf
# Load the model
loaded_model = tf.keras.models.load_model('my_model.h5')
image = cv2.imread('IMG_7992.jpg')
image = cv2.resize(image, (240, 240))
image = image / 255.0
predictions = loaded_model.predict(np.expand_dims(image, axis=0))
predicted_class = np.argmax(predictions)


label = "Bad" if predicted_class == 0 else "Good"

label = f'Quality: {label}'


cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()