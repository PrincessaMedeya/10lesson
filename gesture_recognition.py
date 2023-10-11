import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Загрузка датасета с жестами
dataset_path = 'D:\DANINO\GB\Attestazia\leapGestRecog'
categories = os.listdir(dataset_path)
label_encoder = LabelEncoder()
label_encoder.fit_transform(categories)

data = []
target = []

for category in categories:
    category_path = os.path.join(dataset_path, category)
    images = os.listdir(category_path)
    for image_name in images:
        image_path = os.path.join(category_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))
        data.append(image)
        target.append(label_encoder.transform([category])[0])

data = np.array(data)
target = np.array(target)

# Разделение на обучающую и тестовую выборку
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)

# Создание и обучение модели
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data.reshape(-1, 100, 100, 1), train_target, epochs=10, batch_size=32, validation_split=0.1)

# Запуск видеопотока с веб-камеры и распознавание жестов
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.resize(frame_gray, (100, 100))
    prediction = model.predict(frame_gray.reshape(-1, 100, 100, 1))
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
