import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2  # Add this import
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Parameters
IMG_SIZE = (96, 96)
BATCH_SIZE = 32  # Increased for better stability
EPOCHS = 30
DATA_DIR = '../data'

# Custom data generator to handle BMP files properly
def custom_image_generator(directory, batch_size, img_size, is_training=True):
    """Custom generator that works with BMP files."""
    class_names = sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ])

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Get all image paths and labels
    image_paths = []
    labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg', '.tif')):
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(class_to_idx[class_name])
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Create one-hot encoded labels
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=8)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_one_hot))
    
    def load_and_preprocess(path, label):
        # Read file
        img = tf.io.read_file(path)
        
        # Decode based on file extension
        if tf.strings.regex_full_match(path, ".*\\.bmp"):
            # For BMP files, decode as BMP with 3 channels then convert to grayscale
            img = tf.image.decode_bmp(img, channels=3)
            img = tf.image.rgb_to_grayscale(img)
        else:
            # For PNG/JPG, decode normally
            img = tf.image.decode_image(img, channels=1, expand_animations=False)
            img.set_shape([None, None, 1])
        
        # Resize
        img = tf.image.resize(img, img_size)
        
        # Normalize to [0,1]
        img = tf.cast(img, tf.float32) / 255.0
        
        return img, label
    
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        # Add augmentation for training
        def augment(img, label):
            # Random rotation
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            # Random brightness
            img = tf.image.random_brightness(img, 0.1)
            # Random contrast
            img = tf.image.random_contrast(img, 0.9, 1.1)
            # Clip values
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, label
        
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, class_names

# Load datasets
print("Loading training data...")
train_ds, class_names = custom_image_generator(
    os.path.join(DATA_DIR, 'train'),
    BATCH_SIZE,
    IMG_SIZE,
    is_training=True
)

print("Loading validation data...")
val_ds, _ = custom_image_generator(
    os.path.join(DATA_DIR, 'validation'),
    BATCH_SIZE,
    IMG_SIZE,
    is_training=False
)

print(f"Found {len(class_names)} classes: {class_names}")

MODEL_PATH = '../trained_model/fingerprint_model.h5'


if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = keras.models.load_model(MODEL_PATH)
else:
    print("Creating new model...")
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4 - Additional layer for better feature extraction
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(8, activation='softmax')
    ])

# Compile Model with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint(
        '../trained_model/best_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
]

if not os.path.exists(MODEL_PATH):
    print("Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    model.save(MODEL_PATH)
    print("Model saved.")
else:
    print("Model already trained. Skipping training.")
    history = None


if history is not None:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history.history.get('precision', [0]), label='Precision')
    plt.plot(history.history.get('recall', [0]), label='Recall')
    plt.title('Precision & Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('../trained_model/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Evaluate model
print("\nEvaluating model on validation set...")
val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")