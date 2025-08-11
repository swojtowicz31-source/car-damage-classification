import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0, ResNet50,InceptionV3
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ścieżka do folderu z danymi
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# Parametry modelu
img_height = 150
img_width = 150
batch_size = 32
num_classes = 4  # back_part_car, car_windows, front_part_car, right_left_side_car

# Przygotowanie generatorów z odpowiednim skalowaniem i augmentacją tylko dla treningu
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # 70% trening, 30% walidacja/test

# Wczytanie danych z osobnych katalogów
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  # ważne do metryk i wykresów
)

# Tworzenie modelu CNN

# Wczytanie bazy InceptionV3 bez górnych warstw
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
#base_model.trainable = False  # Zamrażamy warstwy

# Budowa modelu
#model = Sequential([
 #   base_model,
  #  GlobalAveragePooling2D(),
   # Dense(128, activation='relu'),
    #Dropout(0.5),
    #Dense(num_classes, activation='softmax')  # klasy: 4 klasy uszkodzeń
#])

#resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
#model = Sequential([
    #resnet_base,
    #GlobalAveragePooling2D(),
   # Dense(128, activation='relu'),
  #  Dropout(0.5),
 #   Dense(num_classes, activation='softmax')
#])

#moj model na dole
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Trenowanie modelu
epochs = 100 #dla googlenet 40
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size
)

# Ewaluacja na zbiorze testowym
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predykcje na danych testowych
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Nazwy klas
class_names = list(test_generator.class_indices.keys())

# Macierz konfuzji
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Raport klasyfikacji
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Wykresy ROC dla każdej klasy
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true == i, y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.title('ROC Curves for Each Class')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Wizualizacja dokładności i strat
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show() 