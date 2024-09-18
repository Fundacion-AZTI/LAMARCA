#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sara Cloux
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

#  Create the argument parser
# The user can choose the time direction (BACKWARD = -1 or FORWARD = 1) and the starting date (YYYY-MM-DDTHH:MM "(2011-01-20T10:30)" format)

parser = argparse.ArgumentParser(description='Simulación con parámetros.')
parser.add_argument('--direction', choices=['-1', '1'], required=True, help='Direccion de la simulacion')
parser.add_argument('--date', required=True, help='Fecha de inicio en formato YYYY-MM-DDTHH:MM (2011-01-20T10:30)')



# Parse the command line arguments

args = parser.parse_args()


# Assign the value of '--direction' and '--date' to a variable

time_direction = int(args.direction)
empieza = args.date


# Define the depth ('k'), time step ('dt_hours') and total time of simulation ('simulation_time')
k = 0.0
dt_hours = 1.0
simulation_time = 25


# Define the paths for read and save
pwd = '/data/geo/lamarca/ROMS500/Mid_domain/20m_depth_resol/'
save_pwd = '/data/geo/lamarca/ROMS500/RESULTADOS/Traj_TESTS/Mid_domain/0025_resol_25days/'

# Read and load the velocity field
ds = xr.open_dataset(pwd+'Mid_domain_20m_resol_20110202073000_20110331043000.nc')
ds.load()

# Define de Earth's radius
R = 6370e3



lat_1 = ds.lat.values[170]
lat_2 = ds.lat.values[-1]
lon_1 = ds.lon.values[100]
lon_2 = ds.lon.values[-1]
#z = np.arange(-20,10,10)
dx =  0.0025



date = empieza+'.000000000'


dt = time_direction * 3600*dt_hours*(180./((np.pi)*R))
dt_z = time_direction *3600*dt_hours

# Define the name of the velocity field variables

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



# Define the funtion for interpolate the velocity fields in 3D

def interpolate_velocity_3D(ds, latitudes_deg, longitudes_deg, z_m, t):
    
    # Retrieve latitudes, longitudes, and times from the dataset
    latitudes = ds[latitude_ds_name].values 
    longitudes = ds[longitude_ds_name].values
    times = ds[time_ds_name].values.astype('float64') / 1e9  # Convert to seconds since the epoch
    depth = ds[depth_ds_name].values
    
    u =  ds[uo_ds_name][:,:,:,:].values  # Zonal velocity (u)
    v =  ds[vo_ds_name][:,:,:,:].values  # Meridional velocity (v)
    w =  ds[wo_ds_name][:,:,:,:].values  # Vertical velocity (w)


    # Create 3D interpolators for u, v, and w velocities
    interpolator_u = RegularGridInterpolator((times, latitudes, longitudes, depth), u, bounds_error=False, fill_value=np.nan)
    interpolator_v = RegularGridInterpolator((times, latitudes, longitudes, depth), v, bounds_error=False, fill_value=np.nan)
    interpolator_w = RegularGridInterpolator((times, latitudes, longitudes, depth), w, bounds_error=False, fill_value=np.nan)

    # Interpolate velocities at the specified latitude, longitude, depth, and time
    interpolated_u = interpolator_u((t, latitudes_deg, longitudes_deg, z_m))
    interpolated_v = interpolator_v((t, latitudes_deg, longitudes_deg, z_m))
    interpolated_w = interpolator_w((t, latitudes_deg, longitudes_deg, z_m))

    # Return the interpolated velocity components
    return interpolated_u, interpolated_v, interpolated_w


def trajectory_ode_3D(t, state, ds,  _=None):

    # Unpack the current state (latitude, longitude, depth)
    lat, lon, z = state

    # Apply a correction factor based on the latitude (for longitude scaling)
    correction = np.cos(lat * (np.pi / 180.))

    # Interpolate velocity at the current position and time (t)
    u_curr, v_curr, z_curr = interpolate_velocity_3D(ds, lat, lon, z, t)

    # Calculate the new latitude, longitude, and depth based on the interpolated velocities
    new_lat = lat + (u_curr * dt)
    new_lon = lon + (v_curr * dt) / correction
    new_z = z + (z_curr * dt_z)

    # Return the updated position (latitude, longitude, depth) and current velocities
    return new_lat, new_lon, new_z, u_curr, v_curr, z_curr




def simulate_trajectories_3D(ds, lat_1, lat_2, lon_1, lon_2, zeta, dx, init_time, simulation_time):
    
    # VARIABLES
    # ds :: Dataset containing velocity fields
    # lat_1, lat_2, lon_1, lon_2, zeta :: Bounds for particle positions
    # dx :: Particle separation distance
    # init_time :: Start date of the simulation
    # simulation_time :: Duration of the simulation (in DAYS)
    
    # Store the initial particle positions and velocities in lists
    latitude_deg = np.arange(lat_1, lat_2, dx)
    longitude_deg = np.arange(lon_1, lon_2, dx)
    lat_mesh, lon_mesh, z_mesh = np.meshgrid(latitude_deg, longitude_deg, zeta)
    latitudes = lat_mesh.flatten()
    longitudes = lon_mesh.flatten()
    depths = z_mesh.flatten()

    lat_init = latitudes
    lon_init = longitudes
    
    # Convert the start date to seconds since epoch and define the simulation time in seconds
    start_time = np.datetime64(init_time).astype('float64') / 1e9
    simulation_duration = simulation_time * 24. * 3600.
    time_interval = np.arange(start_time, start_time + simulation_duration, dt_hours * 3600.)

    # Convert back to np.datetime64 format for output timestamps
    time_interval_datetime64 = np.datetime64('1970-01-01') + time_interval.astype('timedelta64[s]')

    # Convert timestamps to strings in the desired format
    timestamps = time_interval_datetime64.astype(str)

    # Reverse time intervals and timestamps if simulating backwards
    if time_direction == -1:
        time_interval = np.flip(time_interval)
        timestamps = np.flip(timestamps)

    # Initialize the results list to store trajectory data
    results = []

    # Loop over each time step in the simulation
    for t in time_interval:
        print(t)
        # Interpolate the velocity at the current particle positions and update them
        new_lat, new_lon, new_z, u, v, w = trajectory_ode_3D(t, [latitudes, longitudes, depths], ds, time_direction)

        # Ensure particles do not go above the surface (z <= 0)
        new_z[np.where(new_z > 0.0)[0]] = 0.0

        # Update positions for the next iteration
        latitudes, longitudes, depths = new_lat, new_lon, new_z
        
        # Store the results (positions and velocities) for each particle
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

    # Organize results by particle ID
    dict_by_id = {}
    for d in results:
        id_ = d['id']
        if id_ not in dict_by_id:
            dict_by_id[id_] = {'latitude': [], 'longitude': [], 'depth': [], 'u': [], 'v': [], 'w': [], 'time': []}
        dict_by_id[id_]['latitude'].append(d['latitude'])
        dict_by_id[id_]['longitude'].append(d['longitude'])
        dict_by_id[id_]['depth'].append(d['depth'])
        dict_by_id[id_]['time'].append(d['time'])
    
    # Create an xarray Dataset to store the simulation output
    out_ds = xr.Dataset(
        {
            'latitude': (['id', 'time'], [dict_by_id[id_]['latitude'] for id_ in sorted(dict_by_id.keys())]),
            'longitude': (['id', 'time'], [dict_by_id[id_]['longitude'] for id_ in sorted(dict_by_id.keys())]),
            'depth': (['id', 'time'], [dict_by_id[id_]['depth'] for id_ in sorted(dict_by_id.keys())]),
        },
        coords={'id': sorted(dict_by_id.keys()), 'time': timestamps}
    )

    # Save the dataset to a NetCDF file
    out_ds.attrs['latitude_init'] = lat_init
    out_ds.attrs['longitude_init'] = lon_init
    out_ds.attrs['reshape_dimensions'] = (len(np.arange(lon_1, lon_2, dx)), len(np.arange(lat_1, lat_2, dx)))
    
    # Determine the filename based on simulation direction
    if time_direction == -1:
        out_ds.to_netcdf(save_pwd + 'BACK_' + empieza + '_dx_' + str(dx) + '_depth_' + str(int(k)) + '_dt_' + str(dt_hours) + '.nc')
    if time_direction == 1:
        out_ds.to_netcdf(save_pwd + 'FORW_' + empieza + '_dx_' + str(dx) + '_depth_' + str(int(k)) + '_dt_' + str(dt_hours) + '.nc')

    return out_ds

# Run the simulation
simulate_trajectories_3D(ds, lat_1, lat_2, lon_1, lon_2, k, dx, date, simulation_time)


