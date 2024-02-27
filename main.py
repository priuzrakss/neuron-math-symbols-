from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np

# Список символов: математические и нематематические
symbols = ['+', '-', '=', '(', ')', '*', '/', '^', '%', ':', 'a', 'b', 'c', 'd', 'e']

# Преобразование символов в числовые значения
encoder = LabelEncoder()
encoder.fit(symbols)

# Создание нейросети
model = Sequential()
model.add(Dense(16, input_dim=1, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 2 класса: математический символ или нет

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
X = np.array(range(len(symbols)))
y = to_categorical([1]*10 + [0]*5)  # Первые 10 символов - математические, остальные - нет
model.fit(X, y, epochs=100, batch_size=1)

# Функция для классификации символов
def classify_symbol(symbol):
    symbol_index = encoder.transform([symbol])
    prediction = model.predict(np.array([symbol_index]))
    return 'Математический символ' if np.argmax(prediction) == 1 else 'Не математический символ'

