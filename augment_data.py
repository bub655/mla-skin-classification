import numpy as np
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import random
from PIL import Image
import matplotlib.pyplot as plt

img_width, img_height = 100, 100
metadata = pd.read_csv("HAM10000_metadata.csv")


max_images_per_class = metadata["dx"].value_counts().max()

# create an array to hold the images in the desired order
ordered_images = []
dx = []
lesion_id = []
images_id = []
dx_type = []
sex = []
age = []
localization = []

# create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

# iterate over the metadata DataFrame and load the images
for index, row in metadata.iterrows():
    image_id = row["image_id"]
    filename = image_id + ".jpg"
    found = False

    # try to find the file in both directories
    if os.path.exists(
        os.path.join("/Users/anavbo/Desktop/Personal/ASDRP/HAM10000_images", filename)
    ):
        filename = os.path.join("HAM10000_images", filename)
        found = True

    if not found:
        print(f"Could not find file {filename} for image {image_id}")
        print(
            os.path.exists(
                os.path.join(
                    "/Users/anavbo/Desktop/Personal/ASDRP/HAM10000_images", filename
                )
            )
        )
        continue

    # load the image as a PIL Image object
    image = Image.open(filename).convert("RGB")

    # resize the image to the desired dimensions and convert it to a NumPy array
    image = image.resize((img_width, img_height))
    image = np.asarray(image)

    # calculate the number of additional images to generate
    num_images_to_generate = (
        int(max_images_per_class / metadata["dx"].value_counts()[row["dx"]]) - 1
    )

    # use the ImageDataGenerator to generate additional images
    if num_images_to_generate > 0:
        images = np.expand_dims(image, axis=0)
        datagen.fit(images)

        # For every missing values generate a new image and data and add them to the dataset
        for i in range((num_images_to_generate)):
            generated_images = datagen.flow(images, batch_size=1, shuffle=False).next()
            ordered_images.append(generated_images[0])
            delta = random.randint(-10, 10)
            dx.append(row["dx"])
            dx_type.append(row["dx_type"])
            lesion_id.append(row["lesion_id"])
            images_id.append(row["image_id"])
            age.append(row["age"] + delta)
            sex.append(row["sex"])
            localization.append(row["localization"])

    # add the original data to the arrays
    ordered_images.append(image)
    dx.append(row["dx"])
    dx_type.append(row["dx_type"])
    lesion_id.append(row["lesion_id"])
    images_id.append(row["image_id"])
    age.append(row["age"])
    sex.append(row["sex"])
    localization.append(row["localization"])

# convert the array lists to a NumPy arrays
# ordered_images = np.asarray(ordered_images)
dx = np.array(dx)
dx_type = np.array(dx_type)
lesion_id = np.array(lesion_id)
images_id = np.array(image_id)
age = np.array(age)
sex = np.array(sex)
localization = np.array(localization)

df = pd.DataFrame(
    {
        "image_id": images_id,
        "dx": dx,
        "dx_type": dx_type,
        "lesion_id": lesion_id,
        "sex": sex,
        "age": age,
        "localization": localization,
    }
)

df.to_csv(
    "/Users/anavbo/Desktop/Personal/ASDRP/HAM10000_metadata_augmented.csv",
    index=False,
)

# # Save the images
# np.save("augment_images", ordered_images)
# # Get the value counts of the data
# counts = np.unique(dx, return_counts=True)

# # Create a bar chart of the value counts
# plt.bar(counts[0], counts[1])

# # Add text annotations to the bars
# for i, v in enumerate(counts[1]):
#     plt.text(i, v, str(v), ha="center", va="bottom")

# # Add labels and title
# plt.xlabel("Value")
# plt.ylabel("Count")
# plt.title("Value Count")

# # Show the chart
# plt.show()
