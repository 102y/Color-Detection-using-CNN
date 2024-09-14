import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# Load the color dataset
color_data = pd.read_csv(r'C:\Users\NITRO\Desktop\AI Proj\Detect colors in an image\color_names.csv')  # تأكد من أن الامتداد .csv

# Prepare the data
X = color_data[['Red (8 bit)', 'Green (8 bit)', 'Blue (8 bit)']].values
y = color_data['Name'].values

# Normalize the input RGB values
X = X / 255.0

# Encode the color names into numerical categories
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert to categorical (one-hot encoding)
y_categorical = to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Reshape the data to fit the CNN model (assuming input image size is 1x1x3 since we have RGB values)
X_train = X_train.reshape(-1, 1, 1, 3)
X_test = X_test.reshape(-1, 1, 1, 3)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(1, 1, 3)))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Function to predict the color of an input image
def predict_color(img_path):
    img = image.load_img(img_path, target_size=(1, 1))
    img_array = image.img_to_array(img) / 255.0
    img_array = img_array.reshape(1, 1, 1, 3)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    predicted_color = label_encoder.inverse_transform([predicted_label])
    return predicted_color[0]

# Example usage: Predict the color of an input image
image_path = r'C:\Users\NITRO\Desktop\AI Proj\Detect colors in an image\Red.jpg'  # Replace with your image path
predicted_color = predict_color(image_path)
print("Predicted Color:", predicted_color)

# Calculate average RGB from an image
img = Image.open(image_path)
img_array = np.array(img)
avg_rgb = np.mean(img_array, axis=(0, 1))

# Display the image with the predicted color
plt.imshow(img)
plt.title(f'Predicted Color: {predicted_color}')
plt.axis('off')  # Hide axes
plt.show()
