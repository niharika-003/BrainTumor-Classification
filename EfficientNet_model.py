import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
import matplotlib.pyplot as plt #import pyplot to display graph

# Define Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
LEARNING_RATE = 0.0001
EPOCHS = 20
SEED = 42
TRAIN_DIR = '/content/drive/MyDrive/braintumorDataset/Training/'
TEST_DIR = '/content/drive/MyDrive/braintumorDataset/Testing/'

# --- Image Processing (Blur Handling) ---
def handle_blur(img):
    """Detects and sharpens blurry images."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_32F).var()
    if laplacian_variance < 100:  # Threshold for blur detection
        print(f"Warning: Image is blurry (variance: {laplacian_variance:.2f}). Attempting to sharpen.")
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return img

# --- Data Augmentation with Custom Preprocessing ---
def preprocess_and_augment(img):
    """Combines blur handling with rescaling and converts to RGB."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB (important!)
    img = handle_blur(img)  # Apply blur handling
    img = img / 255.0  # Rescale
    return img

# --- Data Generators ---
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_and_augment # Apply custom preprocessing and augmentation
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: preprocess_and_augment(x) # process validation data.
)

validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED
)

# Calculate Steps per Epoch
train_samples = train_generator.samples
test_samples = validation_generator.samples

train_steps = train_samples // BATCH_SIZE
test_steps = test_samples // BATCH_SIZE

print(f"Train samples: {train_samples}, Test samples: {test_samples}")
print(f"Train steps: {train_steps}, Test steps: {test_steps}")

# --- Model Creation ---
def create_efficientnet_model(num_classes):
    """Creates an EfficientNetB0 model with transfer learning and dropout."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Unfreeze the last few layers of the base model
    for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers for example
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# --- Training Function ---
# def train_model(model, model_name, train_generator, validation_generator, train_steps, validation_steps,initial_epoch=0, class_weight=None):
#     """Trains a given model."""
#     model.compile(optimizer=Adam(learning_rate=LEARNING_RATE/10), #divide the learning rate to make it lower
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     model_checkpoint = ModelCheckpoint(f'{model_name}_best_model_20.keras', monitor='val_loss', save_best_only=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

#     history = model.fit(
#         train_generator,
#         steps_per_epoch=train_steps,
#         epochs=EPOCHS+initial_epochs,
#         validation_data=validation_generator,
#         validation_steps=validation_steps,
#         callbacks=[early_stopping, model_checkpoint, reduce_lr],
#         class_weight=class_weight
#     )
#     return history

def train_model(model, model_name, train_generator, validation_generator, train_steps, validation_steps, initial_epoch=0, class_weight=None):
    """Trains a given model."""
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE/10), #divide the learning rate to make it lower
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'{model_name}_best_model_20.keras', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(
        # train_generator,
        # steps_per_epoch=train_steps,
        # # Changed line below to use initial_epoch
        # epochs=EPOCHS ,
        # validation_data=validation_generator,
        # validation_steps=validation_steps,
        # callbacks=[early_stopping, model_checkpoint, reduce_lr],
        # class_weight=class_weight
        train_generator,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        initial_epoch=initial_epoch, # Start from here
        class_weight=class_weight
    )
    return history

# --- Main Execution ---
if __name__ == '__main__':
    efficientnet_model_filepath = '/content/drive/MyDrive/trainedModels/efficientnet_best_model_20.keras'

    #Calculate class weights
    class_names = sorted(os.listdir(TRAIN_DIR))
    num_classes = len(class_names)
    num_samples_per_class = []
    for class_name in class_names:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        num_samples_per_class.append(len(os.listdir(class_dir)))
    num_samples_per_class = np.array(num_samples_per_class)

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=np.concatenate([[i] * num_samples_per_class[i] for i in range(num_classes)])
    )
    class_weight_dict = dict(zip(range(num_classes), class_weights))
    print("Class Weights:", class_weight_dict)

    # efficientnet_model = create_efficientnet_model(NUM_CLASSES)
    # model_path = '/content/drive/MyDrive/BrainTumorModelCheckpoints/efficientnet_best_model.h5'  # Replace with the correct path
    # efficientnet_model = load_model(model_path)
    # print("Model loaded successfully.")
    # #Train Model:
    # efficientnet_history = train_model(efficientnet_model, "efficientnet", train_data_gen, test_data_gen, train_steps, test_steps, initial_epoch = 10)

    # if os.path.exists(efficientnet_model_filepath):
    #     print(f"Loading existing EfficientNet model from {efficientnet_model_filepath}")
    #     efficientnet_model = load_model(efficientnet_model_filepath)
    #     #Train Model:
    #     efficientnet_history = train_model(efficientnet_model, "efficientnet", train_generator, validation_generator, train_steps, test_steps,initial_epoch=10)
    # else:
    #     print("Training EfficientNet from scratch...")
    #     efficientnet_model = create_efficientnet_model(NUM_CLASSES)
    #     efficientnet_history = train_model(efficientnet_model, "efficientnet", train_generator, validation_generator, train_steps, test_steps, class_weight_dict)
    #     print("EfficientNet training completed.")
    #     efficientnet_model.save(efficientnet_model_filepath)

    if os.path.exists(efficientnet_model_filepath):
        print(f"Loading existing EfficientNet model from {efficientnet_model_filepath}")
        efficientnet_model = load_model(efficientnet_model_filepath)
        initial_epoch = 10
        #Train Model:
        efficientnet_history = train_model(efficientnet_model, "efficientnet", train_generator, validation_generator, train_steps, test_steps, initial_epoch=10, class_weight=class_weight_dict)
    else:
        print("Training EfficientNet from scratch...")
        efficientnet_model = create_efficientnet_model(NUM_CLASSES)
        efficientnet_history = train_model(efficientnet_model, "efficientnet", train_generator, validation_generator, train_steps, test_steps, class_weight_dict)
        print("EfficientNet training completed.")
        efficientnet_model.save(efficientnet_model_filepath)

    # Example Usage: Classify and visualize a single image (You may need to adapt this)
    from tensorflow.keras.preprocessing import image # import
    ##Effecient net example
    example_image_path = '/content/drive/MyDrive/braintumorDataset/Testing/glioma/Te-gl_0010.jpg' # Replace with a valid image path
    last_conv_layer_name_efficientnet = "top_activation"

    # Load and preprocess the image for classification
    img = image.load_img(example_image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0  # Rescale

    predictions = efficientnet_model.predict(img_array) # Predict
    class_names = sorted(os.listdir(TRAIN_DIR)) #get classes

    predicted_class_index = np.argmax(predictions[0]) #get index of the class
    predicted_class_name = class_names[predicted_class_index] #get the name
    confidence = predictions[0][predicted_class_index]  #confidence

    print(f"\nEfficientNet Classification:")
    print(f"Predicted class: {predicted_class_name} with confidence {confidence:.4f}")

    #Import Grad-CAM functions
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


        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0  # Rescale

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        # Rescale heatmap to original image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Convert to RGB for matplotlib


        superimposed_img = heatmap * alpha + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
        return superimposed_img


    gradcam_output = visualize_gradcam(example_image_path, efficientnet_model, last_conv_layer_name_efficientnet)

    plt.imshow(gradcam_output)
    plt.title(f"Grad-CAM: {predicted_class_name}")
    plt.show()

    print("Training and Grad-CAM visualization completed.")
