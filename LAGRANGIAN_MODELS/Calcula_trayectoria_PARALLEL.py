#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:24:56 2024

@author: saracloux
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
import pandas as pd
import csv
import datetime
import argparse
from datetime import datetime, timedelta


# Crear el analizador de argumentos
parser = argparse.ArgumentParser(description='Simulación con parámetros.')
parser.add_argument('--direction', choices=['-1', '1'], required=True, help='Direccion de la simulacion')
parser.add_argument('--date', required=True, help='Fecha de inicio en formato YYYY-MM-DDTHH:MM (2011-01-20T10:30)')



# Analizar los argumentos de la línea de comandos
args = parser.parse_args()


# Asignar el valor de 'dt_horas' a una variable
time_direction = int(args.direction)
empieza = args.date
k = 0.0

dt_horas = 1.0
tiempo_simulacion = 25



pwd = '/data/geo/lamarca/ROMS500/Mid_domain/20m_depth_resol/'
save_pwd = '/data/geo/lamarca/ROMS500/RESULTADOS/Traj_TESTS/Mid_domain/0025_resol_25days/'

ds = xr.open_dataset(pwd+'Mid_domain_20m_resol_20110202073000_20110331043000.nc')
ds.load()


R = 6370e3



lat_1 = ds.lat.values[170]
lat_2 = ds.lat.values[-1]
lon_1 = ds.lon.values[100]
lon_2 = ds.lon.values[-1]
#z = np.arange(-20,10,10)
dx =  0.0025



date = empieza+'.000000000'


dt = time_direction * 3600*dt_horas*(180./((np.pi)*R))
dt_z = time_direction *3600*dt_horas


longitude_ds_name = 'lon'
latitude_ds_name = 'lat'
depth_ds_name = 'z_new'
time_ds_name = 'time'
uo_ds_name = 'vel_u'
vo_ds_name = 'vel_v'
wo_ds_name = 'vel_w'


"""
longitude_ds_name = 'LONGITUDE'
latitude_ds_name = 'LATITUDE'
#depth_ds_name = 'z_w'
time_ds_name = 'TIME'
uo_ds_name = 'EWCT'
vo_ds_name = 'NSCT'
#wo_ds_name = 'vel_w'
"""

def interpolate_velocity_3D(ds, latitudes_deg, longitudes_deg, z_m, t):
    
    # Obtener las latitudes, longitudes y tiempos del conjunto de datos
    latitudes = ds[latitude_ds_name].values 
    longitudes = ds[longitude_ds_name].values
    times = ds[time_ds_name].values.astype('float64') / 1e9  # Convertir a segundos desde la época

    depth = ds[depth_ds_name].values
    
    u =  ds[uo_ds_name][:,:,:,:].values  #/ R
    v =  ds[vo_ds_name][:,:,:,:].values  #/ R
    w =  ds[wo_ds_name][:,:,:,:].values 
    
    # Crear interpoladores 3D para u y v
    interpolator_u = RegularGridInterpolator((times, latitudes, longitudes,depth), u, bounds_error=False, fill_value=np.nan)
    interpolator_v = RegularGridInterpolator((times, latitudes, longitudes,depth), v, bounds_error=False, fill_value=np.nan)
    interpolator_w = RegularGridInterpolator((times, latitudes, longitudes,depth), w, bounds_error=False, fill_value=np.nan)


    # Interpolar velocidades en la latitud, longitud y tiempo especificados
    interpolated_u = interpolator_u((t, latitudes_deg, longitudes_deg, z_m))
    interpolated_v = interpolator_v((t, latitudes_deg, longitudes_deg, z_m))
    interpolated_w = interpolator_w((t, latitudes_deg, longitudes_deg, z_m))


    return interpolated_u, interpolated_v, interpolated_w


def trajectory_ode_3D(t, state, ds,  _=None):
    # Verificar si t es NaN y detener la integración

    lat, lon, z = state
    correccion = np.cos(lat*(np.pi/180.))
    # Interpolate velocity at the current position and time (t)
    u_curr, v_curr, z_curr = interpolate_velocity_3D(ds, lat, lon, z, t)

    # Calculate new positions in radians
    new_lat = lat + (u_curr * dt)
    new_lon = lon + (v_curr * dt) / correccion
    new_z = z + (z_curr * dt_z)
    
    return new_lat, new_lon, new_z, u_curr, v_curr, z_curr




def simulate_trajectories_3D(ds, lat_1, lat_2, lon_1, lon_2, zeta, dx, fecha_inicio, tiempo_simulacion):
    
    # VARIABLES
    # ds :: Dataset que contiee campos de vedlocidades
    # lat_1,lat_2,lon_1,lon_2,zeta :: extermos de las particulas
    # dx :: separacion entre particulas
    # inicio :: fecha inicial
    # tiempo_simulacion :: cuanto dura (EN DIAS) la simulacion
    
    
    # Almacenar las posiciones y velocidades en listas
    latitude_deg = np.arange(lat_1, lat_2, dx)
    longitude_deg = np.arange(lon_1, lon_2, dx)
    lat_mesh, lon_mesh, z_mesh= np.meshgrid(latitude_deg, longitude_deg, zeta)
    latitudes = lat_mesh.flatten()
    longitudes = lon_mesh.flatten()
    depths =z_mesh.flatten()

    
    lat_init = latitudes
    lon_init = longitudes
    #depth_init = depths
    
    inicio = np.datetime64(fecha_inicio).astype('float64')/1e9
    tiempo_de_simulacion = tiempo_simulacion * 24. *3600.
    intervalo_de_tiempo = np.arange(inicio,inicio+tiempo_de_simulacion,dt_horas*3600.)
    
    # Convertir de vuelta a np.datetime64
    intervalo_de_tiempo_datetime64 = np.datetime64('1970-01-01') + intervalo_de_tiempo.astype('timedelta64[s]')

# Convertir a cadenas de texto en el formato deseado
    marcas_temporales = intervalo_de_tiempo_datetime64.astype(str)

    # Truncar la cadena para que solo tenga microsegundos
#    fecha_inicio_str_truncada = fecha_inicio[:26]
#    fecha_inicio_str_truncada = date[:26]
#    fecha_inicial = datetime.strptime(fecha_inicio_str_truncada, '%Y%m%d%H%M%S')

    # Convertir la fecha inicial truncada a un objeto datetime
#    fecha_inicial = datetime.fromisoformat(fecha_inicio+'00')


    # Tu paso temporal en horas y el tiempo de simulación en días



    
    if time_direction == -1:
        intervalo_de_tiempo = np.flip(intervalo_de_tiempo)
        marcas_temporales = np.flip(marcas_temporales)
#    if time_direction == 1:continue    
    results = []

    for t in intervalo_de_tiempo:
        print(t)
        new_lat, new_lon, new_z, u , v, w = trajectory_ode_3D(t, [latitudes, longitudes, depths], ds,time_direction)

#        new_lat, new_lon, new_z, u , v, w = trajectory_ode_3D(t, [latitudes, longitudes, zetas], test)

        # Actualizar las coordenadas para la siguiente iteración
        new_z[np.where(new_z>0.0)[0]] = 0.0

        latitudes, longitudes, depths = new_lat, new_lon, new_z
        
        # Agregar los resultados a la lista
        for i, (lat, lon, depth, uu, vv, ww) in enumerate(zip(new_lat, new_lon, new_z, u, v, w)):
            results.append({
                'id': i,
                'time': t,
                'latitude': lat,
                'longitude': lon,
                'depth': depth,
                'u': uu,
                'v': vv,
                'w': ww
            })

    # Organizar los diccionarios por ID
    diccionarios_por_id = {}
    for d in results:
        id_ = d['id']
        if id_ not in diccionarios_por_id:
            diccionarios_por_id[id_] = {'latitude': [], 'longitude': [], 'depth': [], 'u': [], 'v': [], 'w': [], 'time': []}
        diccionarios_por_id[id_]['latitude'].append(d['latitude'])
        diccionarios_por_id[id_]['longitude'].append(d['longitude'])
        diccionarios_por_id[id_]['depth'].append(d['depth'])
        diccionarios_por_id[id_]['time'].append(d['time'])
    
    # Crear el Dataset de xarray
    out_ds = xr.Dataset(
        {
            'latitude': (['id', 'time'], [diccionarios_por_id[id_]['latitude'] for id_ in sorted(diccionarios_por_id.keys())]),
            'longitude': (['id', 'time'], [diccionarios_por_id[id_]['longitude'] for id_ in sorted(diccionarios_por_id.keys())]),
            'depth': (['id', 'time'], [diccionarios_por_id[id_]['depth'] for id_ in sorted(diccionarios_por_id.keys())]),
        },
        coords={'id': sorted(diccionarios_por_id.keys()), 'time': marcas_temporales}
    )

# Guardar el conjunto de datos en un archivo NetCDF
    
    out_ds.attrs['latitude_init'] = lat_init
    out_ds.attrs['longitude_init'] = lon_init
    out_ds.attrs['reshape_dimensions'] = (len(np.arange(lon_1, lon_2, dx)),len(np.arange(lat_1, lat_2, dx)))
#    if time_direction == -1: out_ds.to_netcdf(save_pwd+'BACK_'+empieza+'_days_'+str(int(tiempo_simulacion))+'_depht_'+str(int(zeta))+'_dt_'+str(dt_horas)+'.nc')
#    if time_direction == 1: out_ds.to_netcdf(save_pwd+'FORW_'+empieza+'_days_'+str(int(tiempo_simulacion))+'_depht_'+str(int(zeta))+'_dt_'+str(dt_horas)+'.nc')
    if time_direction == -1: out_ds.to_netcdf(save_pwd+'BACK_'+empieza+'_dx_'+str(dx)+'_depht_'+str(int(k))+'_dt_'+str(dt_horas)+'.nc')
    if time_direction == 1: out_ds.to_netcdf(save_pwd+'FORW_'+empieza+'_dx_'+str(dx)+'_depht_'+str(int(k))+'_dt_'+str(dt_horas)+'.nc')

    return out_ds 

simulate_trajectories_3D(ds, lat_1, lat_2, lon_1, lon_2,k , dx, date, tiempo_simulacion)


