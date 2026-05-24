#--------------------------------------------------------
# LIBRERIAS
#--------------------------------------------------------
import numpy as np # NumPy
import scipy.stats as stats # SciPy (estadística avanzada)
import pandas as pd # pandas (tablas y estadísticas descriptivas)
import matplotlib.pyplot as plt # Matplotlib (visualización)
import mahotas as mh # Mahotas (haralick)
import seaborn as sns # Seaborn
from sklearn.feature_selection import SelectKBest, f_classif # Selección de características

import cv2 # OpenCV
from skimage import io, color, filters, feature, morphology # scikit-image
from PIL import Image # Pillow
import zipfile #archivo zip
import os

#función que divide el dataset en train/val/test.
from sklearn.model_selection import train_test_split 
from statsmodels.stats.multitest import multipletests 
from sklearn.cluster import KMeans
# para copiar archivos entre carpetas
import shutil
# para interfaz
import customtkinter as ctk

#VER
print("LIBRERIAS")
print("Librerías importadas correctamente")