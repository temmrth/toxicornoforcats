import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
)

val_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
)

# Save the exact class order used by the generator (important!)
# This ensures your Flask app maps model outputs -> correct class name.
class_indices = train_generator.class_indices  # e.g. {'aloe_vera':0, 'lily':1, ...}
classes_in_order = [k for k, _ in sorted(class_indices.items(), key=lambda x: x[1])]
with open("model_classes.json", "w", encoding="utf-8") as f:
    json.dump(classes_in_order, f, ensure_ascii=False, indent=2)

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(classes_in_order), activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_generator, validation_data=val_generator, epochs=5)

model.save("plant_model.h5")
print("Model saved: plant_model.h5")
print("Class order saved: model_classes.json ->", classes_in_order)
