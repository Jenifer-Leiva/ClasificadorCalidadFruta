#--------------------------------------------------------
# LIBRERIAS
#--------------------------------------------------------
import numpy as np # NumPy
import scipy.stats as stats # SciPy (estadística avanzada)
import pandas as pd # pandas (tablas y estadísticas descriptivas)
import matplotlib.pyplot as plt # Matplotlib (visualización)

import cv2 # OpenCV
from skimage import io, color, filters, feature, morphology # scikit-image
from PIL import Image # Pillow
import zipfile #archivo zip
import os

#función que divide el dataset en train/val/test.
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans
 # para copiar archivos entre carpetas
import shutil


#VER
print("LIBRERIAS")
print("Librerías importadas correctamente")