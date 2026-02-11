import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

# Define Constants
IMAGE_SIZE = (224,224) # Standard image size
BATCH_SIZE = 32
NUM_CLASSES = 4
LEARNING_RATE = 0.0001
EPOCHS = 20
SEED = 42  # For reproducibility


# Define Data Paths (Adjust these to your local paths)
TRAIN_DIR = '/content/drive/MyDrive/braintumorDataset/Training/'
TEST_DIR = '/content/drive/MyDrive/braintumorDataset/Testing/'




# Data Preprocessing
def preprocess_image(image_path):

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None

        # Check for blur (Laplacian variance).  Adjust threshold as needed.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_variance < 100:  # Threshold for blur detection
            print(f"Warning: Image {image_path} is blurry (variance: {laplacian_variance:.2f}).  Attempting to sharpen.")
            # Apply unsharp masking to try and sharpen (adjust parameters as needed)
            blurred = cv2.GaussianBlur(img, (0, 0), 3)
            img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0) #adjust values for best result

        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)  # Resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (important for models)
        img = img.astype('float32') / 255.0  # Normalize
        return img

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None



def data_generator(directory, batch_size, shuffle=True):

    class_names = sorted(os.listdir(directory))
    num_classes = len(class_names)
    image_paths = []
    labels = []

    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue  # Skip if it's not a directory
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image_paths.append(image_path)
            labels.append(i)  # Class index

    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Shuffle if requested
    if shuffle:
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)
        image_paths = image_paths[indices]
        labels = labels[indices]

    num_samples = len(image_paths)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_paths = image_paths[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]

            batch_images = []
            for path in batch_paths:
                img = preprocess_image(path)
                if img is not None:  # Only add if preprocessing was successful
                    batch_images.append(img)
                else:
                    # Handle the error: remove the corresponding label
                    index = np.where(image_paths == path)[0][0]
                    labels = np.delete(labels, index)  # Remove the corrupted images' label
                    image_paths = np.delete(image_paths, index)
            batch_images = np.array(batch_images)

            #One hot encode the batch labels
            batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=num_classes)


            yield batch_images, batch_labels


# Calculate Steps per Epoch
train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
test_samples = sum([len(files) for r, d, files in os.walk(TEST_DIR)])

train_steps = train_samples // BATCH_SIZE
test_steps = test_samples // BATCH_SIZE

print(f"Train samples: {train_samples}, Test samples: {test_samples}")
print(f"Train steps: {train_steps}, Test steps: {test_steps}")


# Model Creation Functions
def create_xception_model(num_classes):
    """Creates an Xception model with transfer learning."""
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model.trainable = False  # Freeze base model weights

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # Add a dense layer
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def create_efficientnet_model(num_classes):
    """Creates an EfficientNetB0 model with transfer learning."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model.trainable = False  # Freeze base model weights

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # Add a dense layer
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# Grad-CAM Visualization
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap for a given image and model."""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)  # Normalize
    return heatmap.numpy()


def visualize_gradcam(img_path, model, last_conv_layer_name,  alpha=0.4):
    """Overlays the Grad-CAM heatmap on the original image."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    img_array = preprocess_image(img_path)
    if img_array is None:
        print(f"Skipping GradCAM visualization for {img_path} due to preprocessing error.")
        return None

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Rescale heatmap to original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Convert to RGB for matplotlib


    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    return superimposed_img



# Training Function
def train_model(model, model_name, train_generator, validation_generator, train_steps, validation_steps):
    """Trains a given model."""

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'{model_name}_best_model.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint]
    )

    return history

# Classification and Visualization Function
def classify_and_visualize(image_path, model, last_conv_layer_name, class_names):
    """
    Classifies an image, predicts the tumor type, and visualizes with Grad-CAM.

    Args:
        image_path: Path to the input image.
        model: The trained Keras model.
        last_conv_layer_name: Name of the last convolutional layer for Grad-CAM.
        class_names: List of class names.
    """

    img = preprocess_image(image_path)
    if img is None:
        print(f"Error: Could not preprocess image {image_path}. Skipping classification.")
        return

    img_array = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    print(f"Predicted class: {predicted_class_name} with confidence {confidence:.4f}")

    # Generate and visualize Grad-CAM
    gradcam_output = visualize_gradcam(image_path, model, last_conv_layer_name)

    if gradcam_output is not None:
        plt.imshow(gradcam_output)
        plt.title(f"Grad-CAM: {predicted_class_name}")
        plt.show()
    else:
        print("Grad-CAM visualization failed.")


# Main Execution
if __name__ == '__main__':

  # Load Pre-trained Models
    # try:
    #     xception_model = load_model('xception_best_model.h5')  # Assuming the file is in the current directory
    #     print("Xception model loaded successfully.")
    # except Exception as e:
    #     print(f"Error loading Xception model: {e}")
    #     xception_model = None  # Set to None to avoid errors later

    # try:
    #     efficientnet_model = load_model('efficientnet_best_model.h5')  # Assuming the file is in the current directory
    #     print("EfficientNet model loaded successfully.")
    # except Exception as e:
    #     print(f"Error loading EfficientNet model: {e}")
    #     efficientnet_model = None  # Set to None to avoid errors later
    # Prepare Data Generators
    train_data_gen = data_generator(TRAIN_DIR, BATCH_SIZE, shuffle=True)
    test_data_gen = data_generator(TEST_DIR, BATCH_SIZE, shuffle=False) #No need to shuffle test set



    # Xception Training (or Loading)
    xception_model_filepath = '/content/drive/MyDrive/trainedModels/xception_best_model.keras'
    if os.path.exists(xception_model_filepath):
        print(f"Loading existing Xception model from {xception_model_filepath}")
       # xception_model.summary()
        xception_model = load_model(xception_model_filepath)
    else:
        print("Training Xception from scratch...")
        xception_model = create_xception_model(NUM_CLASSES)
        xception_history = train_model(xception_model, "xception", train_data_gen, test_data_gen, train_steps, test_steps, initial_epoch = 0)
        print("Xception training completed.")
        xception_model.save(xception_model_filepath)

    # EfficientNetB0 Training (or Loading)
    efficientnet_model_filepath = '/content/drive/MyDrive/trainedModels/efficientnet_best_model_20.keras'  # Filepath for saving/loading
    if os.path.exists(efficientnet_model_filepath):
        print(f"Loading existing EfficientNet model from {efficientnet_model_filepath}")
       # efficientnet_model.summary()
        efficientnet_model = load_model(efficientnet_model_filepath)
    else:
        print("Training EfficientNet from scratch...")
        efficientnet_model = create_efficientnet_model(NUM_CLASSES)
        efficientnet_history = train_model(efficientnet_model, "efficientnet", train_data_gen, test_data_gen, train_steps, test_steps, initial_epoch = 0)
        print("EfficientNet training completed.")
        efficientnet_model.save(efficientnet_model_filepath)


    #Xception Model
    # xception_model = create_xception_model(NUM_CLASSES)
    # print("Xception Model Summary:")
    # xception_model.summary()
    # xception_history = train_model(xception_model, "xception", train_data_gen, test_data_gen, train_steps, test_steps)



    # EfficientNetB0 Model
    # efficientnet_model = create_efficientnet_model(NUM_CLASSES)
    # print("\nEfficientNetB0 Model Summary:")
    # efficientnet_model.summary()
    # efficientnet_history = train_model(efficientnet_model, "efficientnet", train_data_gen, test_data_gen, train_steps, test_steps)

    # # Define Class Names
    class_names = sorted(os.listdir(TRAIN_DIR)) #Get the classes from the TRAIN_DIR


    # Example Usage: Classify and visualize a single image

    ##Xception example
    example_image_path ='/content/drive/MyDrive/braintumorDataset/Testing/glioma/Te-gl_0016.jpg' # Replace with a valid image path
    last_conv_layer_name_xception = "block14_sepconv2_act"
    print("\nXception Classification and Grad-CAM:")
    classify_and_visualize(example_image_path, xception_model, last_conv_layer_name_xception, class_names)


    ##Effecient net example
    example_image_path = '/content/drive/MyDrive/braintumorDataset/Testing/glioma/Te-gl_0016.jpg' # Replace with a valid image path
    last_conv_layer_name_efficientnet = "top_activation"
    print("\nEfficientNet Classification and Grad-CAM:")
    classify_and_visualize(example_image_path, efficientnet_model, last_conv_layer_name_efficientnet, class_names)


    print("Training and Grad-CAM visualization completed.")
