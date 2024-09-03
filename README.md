# TFM_Leandro_Lopez_Ezcurra
Este repositorio contiene el código desarrollado para el Trabajo de Fin de Máster (TFM) sobre la implementación de un sistema de Machine Learning para la clasificación de residuos reciclables mediante análisis de imágenes.

# Archivos principales
Codigo_modelo_Entrenado_TFM.py: Este script en Python utiliza el modelo ya entrenado para probar la clasificación de residuos. Está diseñado para usuarios que desean probar directamente el rendimiento del modelo sin necesidad de entrenarlo desde cero. Solo debes ejecutar el archivo y seguir las instrucciones para cargar tus imágenes de prueba.

Codigo_Final_TFM_Clasificación.ipynb: Este Jupyter Notebook contiene todo el código completo, incluyendo las funciones de entrenamiento y prueba del modelo. Está orientado para aquellos interesados en comprender el proceso completo, desde la preparación de datos, entrenamiento del modelo, hasta la evaluación final. Se recomienda este archivo para una visión detallada del desarrollo del proyecto.

model_b5_1.pt: Archivo que contiene el modelo ya entrenado. Este archivo debe ser utilizado junto con Codigo_modelo_Entrenado_TFM.py para ejecutar el modelo y realizar la clasificación de imágenes de residuos sin necesidad de volver a entrenarlo. Puedes descargarlo desde este enlace https://drive.google.com/drive/folders/1LfBJRI204JPho7_gPaSuEWC2YBIIw4X3?usp=drive_link
 
# Carpetas de Imágenes
train_crops/: Contiene las imágenes utilizadas para el entrenamiento del modelo. Esta carpeta incluye las imágenes de diversas clases de residuos, organizadas de manera que se puedan cargar fácilmente durante el proceso de entrenamiento.

test_crops/: Incluye las imágenes utilizadas para la prueba y evaluación del modelo. Estas imágenes permiten medir la precisión y efectividad del modelo en la clasificación de residuos una vez entrenado.

Estos archivos y carpetas permiten probar el modelo entrenado o explorar el proceso completo de entrenamiento y evaluación.
