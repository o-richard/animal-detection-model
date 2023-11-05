import os

import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)
print(np.__version__)
print(pd.__version__)

# The directory paths should match where the images are located
TRAIN_DIR = "train"
TEST_DIR = "test"
NUM_CLASSES = len(os.listdir(TRAIN_DIR))
IMAGE_SIZE = 600
BATCH_SIZE = 32
EPOCHS = 5
IMAGE_TARGET_SIZE = (IMAGE_SIZE, IMAGE_SIZE)

# Build the model
img_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1,
        ),
        tf.keras.layers.experimental.preprocessing.RandomFlip(),
        tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


def build_model() -> "tf.keras.Model":
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = img_augmentation(inputs)

    model = tf.keras.applications.EfficientNetB7(
        include_top=False,
        input_tensor=x,
        weights="imagenet",
    )

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="pred",
    )(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def unfreeze_model(model: "tf.keras.Model"):
    # unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )


# Fit the Model
model = build_model()
unfreeze_model(model)

train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2,
)
train_batches = train_image_generator.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=IMAGE_TARGET_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
)
valid_batches = train_image_generator.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=IMAGE_TARGET_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
)
test_batches = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
).flow_from_directory(
    directory=TEST_DIR,
    target_size=IMAGE_TARGET_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

history = model.fit(
    train_batches,
    epochs=EPOCHS,
    validation_data=valid_batches,
)


# Visualize the training results
epochs_range = range(EPOCHS)

training_acc = history.history["accuracy"]
validation_acc = history.history["val_accuracy"]
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
acc_df = pd.DataFrame(
    {"Training Accuracy": training_acc, "Validation Accuracy": validation_acc},
    index=epochs_range,
)
loss_df = pd.DataFrame(
    {"Training Loss": training_loss, "Validation Loss": validation_loss},
    index=epochs_range,
)
# Plot Accuracy
# acc_df.plot()

# Plot Loss
# loss_df.plot()

# Testing the Model
test_labels = test_batches.classes
print("Test Labels", test_labels)
print(test_batches.class_indices)
predictions = model.predict(test_batches, steps=len(test_batches))
acc = 0
for i in range(len(test_labels)):
    actual_class = test_labels[i]
    if predictions[i][actual_class] > 0.5:
        acc += 1
print("Accuarcy:", (acc / len(test_labels)) * 100, "%")

# Save the entire model (architecture, weights, and optimizer state)
model.save("animal_detection_model")
# Save only the weights of the model
model.save_weights("animal_detection_model_weights.h5")
