from CreateModelCovid import *

def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={"KerasLayer": hub.KerasLayer})
    return model


model1 = load_model("C:/Users//giorgos//Desktop//576013_1042828_compressed_COVID-19 Radiography Database (2)//-full-covid-model_teliko.h5")

# Evaluate the pre-saved model


# testing

# Load test image filenames (since we're using os.listdir(), these already have .jpg)


# Get custom image filepaths
custom_path = "C://Users//giorgos//Desktop//xray//"
custom_image_paths = [custom_path + fname for fname in os.listdir(custom_path)]

# Turn custom image into batch (set to test data because there are no labels)
custom_data = create_data_batches(custom_image_paths, test_data=True)

# Make predictions on the custom data
custom_preds = model1.predict(custom_data)

# Get custom image prediction labels
custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
print(custom_pred_labels)

# Get custom images (our unbatchify() function won't work since there aren't labels)
custom_images = []

# Loop through unbatched data
for image in custom_data.unbatch().as_numpy_iterator():
    custom_images.append(image)

# Check custom image predictions
plt.figure(figsize=(10, 10))
for i, image in enumerate(custom_images):
    plt.subplot(1, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(custom_pred_labels[i])
    plt.imshow(image)
 
plt.show()
