import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

img_size = (224, 224)
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    "archive/train", validation_split=0.2, subset="training",
    seed=123, image_size=img_size, batch_size=batch_size)

val_ds = keras.utils.image_dataset_from_directory(
    "archive/train", validation_split=0.2, subset="validation",
    seed=123, image_size=img_size, batch_size=batch_size)

test_ds = keras.utils.image_dataset_from_directory(
    "archive/test", image_size=img_size, batch_size=batch_size)

class_names = train_ds.class_names
print("Kelas:", class_names)

# Normalisasi pixel ke rentang 0-1
normalization_layer = keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Visualisasi contoh data
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy())
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Membangun model CNN
num_classes = len(class_names)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Training
history = model.fit(train_ds, epochs=15, validation_data=val_ds)

# Evaluasi
test_loss, test_acc = model.evaluate(test_ds)
print(f'Akurasi Model: {test_acc:.2f}')

# Simpan model
model.save("model_kopi.h5")

# Confusion Matrix
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Prediksi")
plt.ylabel("Label Asli")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))