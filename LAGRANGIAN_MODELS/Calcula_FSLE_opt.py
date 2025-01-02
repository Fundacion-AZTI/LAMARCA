#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:15:12 2024

@author: saracloux
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import cv2
import seaborn as sns
import pandas as pd
from scipy.spatial import KDTree
import time 
import os
import gc
import re


pwd_files = '/data/geo/lamarca/ROMS500/RESULTADOS/3D_15Days/backwards/depth_25/'
day_6 = 'BACK_2011-02-26T03:30:00_dx_0.0025_depht_-25_dt_1.nc'


ds = xr.open_dataset(pwd_files + day_6)
ds.load()

umbral = 0.05
dt_horas = float(day_6[-4:-3])
#dt_horas = 1.0
ids = np.transpose(np.reshape(ds.id.values,ds.reshape_dimensions))
dx = day_6[30:36]
dx = dx.replace('_','')
dx = float(dx)
#dx=0.0025
zeta = day_6[-11:-8]
zeta = zeta.replace('_','')

rows, cols = ids.shape

# Definir los desplazamientos de los vecinos
displacements = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Crear un diccionario para almacenar los vecinos de cada ID
neighbors = {}

# Buscar vecinos para cada punto en la matriz
for i in range(rows):
    for j in range(cols):
        id = ids[i, j]
        current_coords = (i, j)
        current_neighbors = []

        # Buscar vecinos usando los desplazamientos definidos
        for deltax, deltay in displacements:
            neighbor_coords = (i + deltax, j + deltay)
            if 0 <= neighbor_coords[0] < rows and 0 <= neighbor_coords[1] < cols:
                neighbor_id = ids[neighbor_coords]
                current_neighbors.append(neighbor_id)

        
        # Almacenar la tupla de vecinos en el diccionario
        neighbors[id] = tuple(current_neighbors)

filtered_neighbors = {key: value for key, value in neighbors.items() if len(value) == 4}
neighbors = [(clave,) + valores for clave,valores in filtered_neighbors.items()]
pepe = pd.DataFrame(neighbors)

ds.close()
del ds
gc.collect()

def uniq(iterable):
    already = set()
    for x in iterable:
        if x not in already:
            yield x
            already.add(x)

def tryint(s):
    """
    Return an int if possible, or `s` unchanged.
    """

    try:
        return int(s)
    except ValueError:
        return s
 

def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.
    >>> alphanum_key("z23a")
    ["z", 23, "a"]
    """

    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

 

def human_sort(l):
    """
    Sort a list in the way that humans expect.
    """

    l.sort(key=alphanum_key)

def spherical_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia esférica entre dos puntos en la Tierra dados
    sus latitudes y longitudes en grados.
    
    Args:
    lat1, lon1: Latitud y longitud del primer punto (en grados)
    lat2, lon2: Latitud y longitud del segundo punto (en grados)
    
    Returns:
    Distancia esférica entre los dos puntos en grados.
    """
    DEG2RAD = np.pi / 180.0
    lat1_rad = lat1 * DEG2RAD
    lon1_rad = lon1 * DEG2RAD
    lat2_rad = lat2 * DEG2RAD
    lon2_rad = lon2 * DEG2RAD
    
    
    xinterm = (np.cos(lat1_rad) * np.cos(lat2_rad) * np.cos((lon2_rad - lon1_rad)) +
               np.sin(lat1_rad) * np.sin(lat2_rad))
    if xinterm.any() >1.0:print('quepasa?9')
    elif xinterm.any() <-1.0:print('quepasa?9')
    
    dist = np.arccos(xinterm) * (1.0/ DEG2RAD)
    
    return dist 


def calcula_fsle(ds):
    
#    ds['time'] = ds.time.values.astype('float64') / 1e9
#    t_matrix = np.zeros(ds.reshape_dimensions)
#    dis_matrix = np.zeros(ds.reshape_dimensions)
    
    tuplas_resultado=([])
    primeros_valores=([])
    for t in range(len(ds.time)):
        dist_d = spherical_distance(ds.latitude[pepe[1],t].values, ds.longitude[pepe[1],t].values, ds.latitude[pepe[0],t].values, ds.longitude[pepe[0],t].values)
        dist_u = spherical_distance(ds.latitude[pepe[2],t].values, ds.longitude[pepe[2],t].values, ds.latitude[pepe[0],t].values, ds.longitude[pepe[0],t].values)
        dist_r = spherical_distance(ds.latitude[pepe[3],t].values, ds.longitude[pepe[3],t].values, ds.latitude[pepe[0],t].values, ds.longitude[pepe[0],t].values)
        dist_l = spherical_distance(ds.latitude[pepe[4],t].values, ds.longitude[pepe[4],t].values, ds.latitude[pepe[0],t].values, ds.longitude[pepe[0],t].values)

        df_dist_d = pd.DataFrame(dist_d, columns=['0'])

        df_dist_u = pd.DataFrame(dist_u, columns=['1'])

        df_dist_r = pd.DataFrame(dist_r, columns=['2'])

        df_dist_l = pd.DataFrame(dist_l, columns=['3'])
        
        result = pd.concat([df_dist_d, df_dist_u, df_dist_r, df_dist_l], axis=1)
        result.index = pepe[0].values
        result = result.drop(index = primeros_valores,errors='ignore')
        result['max_value'] = result.max(axis=1,skipna=True)
        
        filtered_df = result[result['max_value'] >= umbral]
#        index_to_drop.append(filtered_df.index.values)
#        index_to_drop = np.unique(index_to_drop)

        for idx, row in filtered_df.iterrows():
            
            tuplas_resultado.append((idx, row['max_value'], t))   
            print(idx,row['max_value'],t)
        primeros_valores = [tupla[0] for tupla in tuplas_resultado]
      

    return tuplas_resultado


def plot_and_save_fsle(tuplas):
    
    ratio = umbral/dx
    ides, dist, time_stamp = ([tupla[i] for tupla in tuplas] for i in range(3))
    
    tau = [ts * (dt_horas / 24.0) for ts in time_stamp]
#    log_dis = [np.log(dis/dx) for dis in dist]
#    fsle_final = [ld / t for ld, t in zip(log_dis, tau)]
    log_dis = np.log(ratio)    
    fsle_final = [log_dis / t for t in tau]
    df_fsle = pd.DataFrame(fsle_final, columns=['FSLE'])
    df_fsle.index = ides
    
    # Crear una nueva matriz llena de np.nan con las mismas dimensiones que ids
    fsle_matrix = np.transpose(np.full(ds.reshape_dimensions, np.nan))
    
    indices = df_fsle.index.values
    fsle_values = df_fsle['FSLE'].values
    
    # Rellenar fsle_matrix con los valores de FSLE correspondientes
    for idx, value in zip(indices, fsle_values):
        # Encontrar la posición del índice en la matriz ids
        position = np.where(ids == idx)
        if position[0].size > 0 and position[1].size > 0:  # Asegurar que se encuentra el índice
            fsle_matrix[position] = value
    
    
    lat_dimension = ds.latitude_init[0:ds.reshape_dimensions[1]]
    lon_dimension = ds.longitude_init[0::ds.reshape_dimensions[1]]
    
#    ratio = umbral/dx
    
    #fsle_matrix_zeroed = np.nan_to_num(fsle_matrix, nan=0.0)
    
    # Configurar el tamaño de las fuentes
    plt.rcParams.update({
        'font.size': 14,  # Ajusta el tamaño de la fuente general
        'axes.titlesize': 16,  # Título de los ejes
        'axes.labelsize': 14,  # Etiquetas de los ejes
        'xtick.labelsize': 12,  # Etiquetas de las marcas en el eje x
        'ytick.labelsize': 12,  # Etiquetas de las marcas en el eje y
        'legend.fontsize': 14,  # Tamaño de la fuente de la leyenda
        'figure.titlesize': 18  # Título de la figura
    })
    
    
    plt.figure(figsize=(10, 8))
    plt.imshow(((np.flipud(fsle_matrix[:-2,:-2]))), 
               vmin=0, 
               vmax=14,
               extent=[lon_dimension[1], lon_dimension[-2], lat_dimension[1], lat_dimension[-2]],
               cmap='binary')

    cbar = plt.colorbar(orientation='horizontal', pad=0.1)
    cbar.set_label('Days$^{-1}$ [s]')
    #plt.colorbar(label='Days$^{-1}$ [s]')
    plt.xlabel('Longitude [º]')
    plt.ylabel('Latitude [º]')

    plt.title(r'$\delta_f = {} \delta_0$ ; ({})'.format(ratio,  file_name[5:-31]), fontsize=16)
    
    os.makedirs(folder_path, exist_ok=True)
    output_path = os.path.join(folder_path, 'fsle_matrix_ratio_'+str(ratio)+'_'+file_name[:-3]+'.png')
    plt.savefig(output_path, dpi=300)  # Guardar con alta resolución (300 dpi)
    
    plt.show()
    
    return output_path

folder_path = pwd_files + '/fsle_images/'
files = os.listdir(pwd_files)
    
# Filtrar archivos por la extensión deseada, por ejemplo, .nc para archivos NetCDF
files = [file for file in files if file.endswith('.nc')]
human_sort(files)


for file in files:
    ds = xr.open_dataset(pwd_files+file)
    ds.load()
    print(file)
    file_name = file
    tuplas = calcula_fsle(ds)
    paco = pd.DataFrame(tuplas)
    paco.to_csv(pwd_files+'csv/'+file_name[5:-31]+'.csv')

    plot_and_save_fsle(tuplas)

    ds.close()
    del ds
