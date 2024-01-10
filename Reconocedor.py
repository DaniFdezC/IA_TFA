from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from TipoSenales import SenalTrafico

# CARGAR EL MODELO ENTRENADO
model = load_model('clasificador-trafico.h5')

# RUTA DE LA CARPETA DE LAS IMÁGENES QUE SE INTENTARÁN ADIVINAR
carpeta_imagenes = "ImagenesParaReconocer"

lista_imagenes = []

# ITERACIÓN POR LA CARPETA
for archivo in os.listdir(carpeta_imagenes):
    if archivo.endswith(".png"):
        ruta_imagen = os.path.join(carpeta_imagenes, archivo)
        try:
            # CARGAR IMAGEN
            new_image_path = ruta_imagen
            new_image = Image.open(new_image_path)

            # PREPROCESAMIENTO
            new_image = new_image.resize((30, 30))
            new_image_array = np.array(new_image)
            new_image_array = np.expand_dims(new_image_array, axis=0)  # Añadir dimensión extra para el lote

            # PREDICCIÓN
            with tf.device('/GPU:0'):
                predictions = model.predict(new_image_array)

            # INTERPRETACIÓN DE RESULTADO
            predicted_class = np.argmax(predictions)
            nombre_predecir = SenalTrafico.obtener_senal(predicted_class)
            print(f"La predicción es la clase: {nombre_predecir}, la imagen es {archivo}")

        except Exception as e:
            print(f"No se pudo abrir la imagen {archivo}: {e}")


