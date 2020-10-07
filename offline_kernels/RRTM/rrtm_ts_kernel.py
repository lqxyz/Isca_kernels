from __future__ import print_function
import pyrrtm
import numpy as np
import scipy.io.netcdf
import matplotlib.pyplot as plt
import xarray as xar
from scipy import interpolate as ip
import logging
import time
import os
import itertools
import multiprocessing
import ctypes


def monthly_mean_for_3hourly_datas(dt):
    interval = 24 / 3 * 30
    shp = list(np.shape(dt))
    shp.remove(shp[0])
    shp.insert(0, 12)
    monthlydt = np.zeros(shp, dtype=np.double)
    for i in range(12):
        monthlydt[i,] = np.mean(dt[i*interval:(i+1)*interval,], axis=0)
    return monthlydt


def save_zonal_mean_data(dt, lev_type, filename, varname):
    if lev_type == 'phalf':
        dt = xar.DataArray(dt, [('index', range(ntime)), ('phalf', pz), ('lat', lats)])
    if lev_type == 'pfull':
        dt = xar.DataArray(dt, [('index', range(ntime)), ('pfull', pavel), ('lat', lats)])
    if lev_type == 'no_lev':
        dt = xar.DataArray(dt, [('index', range(ntime)), ('lat', lats)])
    # Save the data into netcdf file
    dt.name = varname
    dt.to_netcdf(filename)
    # Save the data into netcdf file
    dt.name = varname
    dt.to_netcdf(filename)


def toa_net_lw_flux(nt, disturb=False):
    lw_toa_lat = np.zeros(nlat)

    for i in range(nlat):
        lw = pyrrtm.LW(nlayers)
        lw.tz = tz[nt,:,i]
        lw.pz = pz

        # If disturb the surface temperature
        if disturb:
            lw.tz[0] = lw.tz[0] + 1.0

        lw.set_species('co2', co2[nt,:,i], 'vmr')
        lw.set_species('h2o', q_mmr[nt,:,i], 'mmr')
                
        # run the radiation code
        output = lw.run()
        # fnet(nlayers+1,)
        lw_fnet = output.fnet
        lw_toa_lat[i] = lw_fnet[nlayers]

    return lw_toa_lat


def run_kernel(nt):
    print(nt)    
    # Original net longwave and shortwave flux at TOA
    orig_lwflux = toa_net_lw_flux(nt, disturb=False)
    
    # Net longwave and shortwave flux at TOA after perturbation
    new_lwflux = toa_net_lw_flux(nt, disturb=True)

    lw_kernel[nt,:] = new_lwflux - orig_lwflux 


# ============================================================= #
# Read data begin
# ============================================================= # 

# Recording the running time
start_time = time.time()
logfilename = 'out.rrtm.parallel_ts_kernel.log'
if os.path.exists(logfilename):
    os.remove(logfilename)
logging.basicConfig(filename=logfilename,level=logging.DEBUG)

# Read the input dataset
input_dir = '../../input'
ds = xar.open_dataset(os.path.join(input_dir, 'three_hourly_avg_data_in_one_year_rrtm.nc'), decode_times=False)

p_full = np.array(ds.pfull) # Units: hPa
p_half = np.array(ds.phalf)
nlayers = len(p_full)

lats = np.array(ds.lat)
lons = np.array(ds.lon)
nlat = len(lats)
nlon = len(lons)

# Using 3-hourly data
ntime = len(ds.index)

T_in = ds.temp.mean(dim='lon')
sphum_in = ds.sphum.mean(dim='lon')

pavel = p_full[::-1]
pz = p_half[::-1]
pz[-1] = 1e-40

# Temperature
fT = ip.interp1d(p_full, T_in, kind='linear', axis=1, fill_value='extrapolate')
tavel = fT(pavel)
tz = fT(pz)

# Water vapour
q_mmr_in = sphum_in / (1 - sphum_in)
f = ip.interp1d(p_full, q_mmr_in, kind='linear', axis=1, fill_value='extrapolate')
q_mmr = f(pavel)

# CO2
co2 = np.ones(tavel.shape) * 360. / 1.0e6

# Ozone file: ozone_1990(time: 12, pfull: 59, lat: 64, lon: 1)
o3 = xar.open_dataset(os.path.join(input_dir, 'ozone_1990.nc'), decode_times=False) # mass mixing ratio
f = ip.interp1d(np.array(o3.pfull), np.array(o3.ozone_1990), kind='linear', axis=1) #, fill_value='extrapolate')
o3_avel = f(pavel)
o3_zm_tm = np.mean(o3_avel, axis=(0,3)) # Zonal and annual mean

# Insolation: zenith angle in degree
coszen = monthly_mean_for_3hourly_datas(ds.coszen.mean(dim='lon'))
zenith_angle_2d = np.rad2deg(np.arccos(coszen))

lw_kernel = np.zeros((ntime, nlat), dtype=np.double)

# Create shared memory among the processors
#shared_array_base_lw = multiprocessing.Array(ctypes.c_double, ntime*nlat)
#lw_kernel = np.ctypeslib.as_array(shared_array_base_lw.get_obj())
#lw_kernel = lw_kernel.reshape((ntime, nlat))

# ============================================================= #
# Read data end
# ============================================================= # 


logging.info("ds read and process cost: --- %s seconds --- " % (time.time() - start_time))

if __name__ == '__main__':

    now = time.time() 

    for nt in range(ntime):
        run_kernel(nt)

    out_dt_dir = '../../kernel_data/'
    if not os.path.exists(out_dt_dir):
        os.makedirs(out_dt_dir)

    save_zonal_mean_data(lw_kernel, 'no_lev', os.path.join(out_dt_dir, 'rrtm_sfc_ts_lw_kernel_3_hourly_zonalmean.nc'), 'lw_kernel')

    logging.info("Parallel cost: --- %s seconds --- " % (time.time() - now))
    logging.info("Now total cost: --- %s seconds --- " % (time.time() - start_time))
