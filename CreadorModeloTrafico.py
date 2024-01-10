import numpy as np
import pandas as pd 
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score

data = []
labels = []
clases = 43
ruta_actual = os.getcwd()

# Carga de datos de entrenamiento
for i in range(clases):
    ruta = os.path.join('Trafico','Train',str(i))
    imagenes = os.listdir(ruta)

    for a in imagenes:
        try:
            imagen = Image.open(ruta + '/'+ a)
            imagen = imagen.resize((30,30))
            imagen = np.array(imagen)
            data.append(imagen)
            labels.append(i)
        except:
            print("Error cargando imagen")

# CONVERSIÓN DE LISTAS A ARRAYS DE NUMPY
data = np.array(data)
labels = np.array(labels)

# PREPROCESAMIENTO DE DATOS
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Conversión a variables categóricas
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# DEFINICIÓN DE MODELO

## Inicializa un modelo secuencial de Keras, que es una pila lineal de capas
model = Sequential()

## Conv2D
### Aplican filtros convolucionales a la entrada para extraer características
### La primera capa tiene 32 filtros de tamaño (5,5) y la segunda capa tiene 64 filtros de tamaño (3,3)
### La función de activación en ambas ReLu

## MaxPool2D
### Capas de pooling que realiza el muestreo máximo sobre la entrada, reduciendo así las dimensiones espaciales
### Para las MaxPool utilizamos una ventana de (2,2)

## Dropout
## Ayudan a prevenir el sobreajuste apagando aleatoriamente algunas unidades durante el entrenamiento
## Se aplica un 25% y 50% de dropout después de las capas de convolución y antes de las capas totalmente conectadas

## Flatten
### Capas que aplanan la entrada para convertirla en vectores unidimensionales antes de pasar a las capas totalmente conectadas

## Dense
### Capas totalmente conectadas, la primera tiene 256 unidades con la función de activación Relu
### La segunda capa tiene 43 unidades (igual al número de clases) con la función de activación softmax
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#  COMPILACIÓN Y RESUMEN DEL MODELO
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ENTRENAMIENTO DEL MODELO POR GRÁFICA
with tf.device('/GPU:0'):
    epochs = 15
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

# GRÁFICAS DE RENDIMIENTO
plt.figure(0)
plt.plot(history.history['accuracy'], label='accuracy training')
plt.plot(history.history['val_accuracy'], label='val accuracy`')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='loss training')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# IMPORTACIÓN DEL CONJUNTO DE DATOS DE PRUEBA
y_test = pd.read_csv('Trafico/Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]

# CARGA DE IMAGENES DE PRUEBA
with tf.device('/GPU:0'):
    for img in imgs:
        imagen = Image.open('Trafico/'+img)
        imagen = imagen.resize([30, 30])
        data.append(np.array(imagen))

X_test=np.array(data)

# PREDICCIONES Y EVALUACIÓN DE MODELO
with tf.device('/GPU:0'):
    pred = np.argmax(model.predict(X_test), axis=-1)

print(accuracy_score(labels, pred))

# GUARDAR EL MODELO
model.save('clasificador-trafico.h5')