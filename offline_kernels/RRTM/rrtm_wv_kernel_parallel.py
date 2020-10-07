from __future__ import print_function
import pyrrtm   # https://github.com/tomflannaghan/pyrrtm
import numpy as np
import scipy.io.netcdf
import matplotlib.pyplot as plt
import xarray as xr
from scipy import interpolate as ip
import logging
import time
import os
import itertools
import multiprocessing
from numba import jit
import ctypes


def monthly_mean_for_3hourly_datas(dt):
    interval = 24/3*30
    shp = list(np.shape(dt))
    shp.remove(shp[0])
    shp.insert(0,12)
    monthlydt = np.zeros(shp, dtype=np.double)
    for i in range(12):
        monthlydt[i,] = np.mean(dt[i*interval:(i+1)*interval,], axis=0)
    return monthlydt


def save_zonal_mean_data(dt, lev_type, filename, varname):
    if lev_type == 'phalf':
        dt = xr.DataArray(dt, [('index', range(ntime)), ('phalf', pz), ('lat', lats)])
    if lev_type == 'pfull':
        dt = xr.DataArray(dt, [('index', range(ntime)), ('pfull', pavel), ('lat', lats)])
    if lev_type == 'no_lev':
        dt = xr.DataArray(dt, [('index', range(ntime)), ('lat', lats)])
    # Save the data into netcdf file
    dt.name = varname
    dt.to_netcdf(filename)
    # Save the data into netcdf file
    dt.name = varname
    try:
        dt.phalf = pz
        dt.to_netcdf(filename)
    except:
        dt.to_netcdf(filename)


def es_from_clausius_clapeyron(T_in_K):
    e_0 = 6.1078 # hPa
    T_0 = 273.16 # Kelvin
    Lv =  2.500e6
    rvgas = 461.50
    es = e_0 * np.exp(Lv / rvgas * (1./T_0 - 1./T_in_K))
    return es


def sphum_level_perturb(specific_hum, t_tmp):
    sphum_tmp = es_from_clausius_clapeyron(t_tmp + 1.0) / es_from_clausius_clapeyron(t_tmp) * specific_hum
    return sphum_tmp


def toa_net_lw_flux(nt, nlev, mmr_or_sphum='mmr', disturb=False):
    lw_toa_lat = np.zeros(nlat)
    if 'mmr' in mmr_or_sphum:
        q_mmr_2d = q_mmr_avel[nt,:,:]
    else:
        sphum_2d = sphum_avel[nt,:,:]
        q_mmr_2d = sphum_2d / (1 - sphum_2d)

    for i in range(nlat):
        lw = pyrrtm.LW(nlayers)
        #lw.tavel = tavel_2d[:,i]
        #lw.pavel = pavel
        lw.tz = tz[nt,:,i]
        lw.pz = pz

        if disturb and 'sphum' in mmr_or_sphum:
            sphum_tmp =  sphum_level_perturb(sphum_2d[nlev, i], lw.tavel[nlev])
            q_mmr_2d[nlev, i] = sphum_tmp / (1 - sphum_tmp)

        lw.set_species('co2', co2[nt,:,i], 'vmr')
        lw.set_species('h2o', q_mmr_2d[:,i], 'mmr')
                
        # run the radiation code
        output = lw.run()
        lw_fnet = output.fnet
        lw_toa_lat[i] = lw_fnet[nlayers-1]

    return lw_toa_lat


def toa_net_sw_flux(nt, nlev, mmr_or_sphum='mmr', disturb=False):
    sw_toa_lat = np.zeros(nlat)
    if 'mmr' in mmr_or_sphum:
        q_mmr_2d = q_mmr_avel[nt,:,:]
    else:
        sphum_2d = sphum_avel[nt,:,:]
        q_mmr_2d = sphum_2d / (1 - sphum_2d)

    for i in range(nlat):
        sw = pyrrtm.SW(nlayers)
        #sw.tavel = tavel_2d[:,i]
        #sw.pavel = pavel
        sw.tz = tz[nt,:,i]
        sw.pz = pz
        
        if disturb and 'sphum' in mmr_or_sphum:
            sphum_tmp =  sphum_level_perturb(sphum_2d[nlev, i], sw.tavel[nlev])
            q_mmr_2d[nlev, i] = sphum_tmp / (1 - sphum_tmp)

        sw.set_species('co2', co2[nt,:,i], 'vmr')
        sw.set_species('h2o', q_mmr_2d[:,i], 'mmr')
        sw.set_species('o3', o3_avel_zm_tm[:,i], 'mmr')
                
        # Surface
        albedo = 0.3
        sw.semis = 1.0-albedo 

        #Solar Forcing Parameters
        sw.juldat = 90
        sw.sza = zenith_angle_2d[nt,i]

        # run the radiation code
        output = sw.run()
        sw_fnet = output.fnet
        sw_toa_lat[i] = sw_fnet[nlayers-1]

    return sw_toa_lat


def run_kernel(ntnlev):
    now = time.time() 
    #print(ntnlev)
    nt = ntnlev[0]
    nlev = ntnlev[1]
    
    orig_lwflux = toa_net_lw_flux(nt, nlev, mmr_or_sphum='mmr', disturb=False)
    orig_swflux = toa_net_sw_flux(nt, nlev, mmr_or_sphum='mmr', disturb=False)
    
    new_lwflux = toa_net_lw_flux(nt, nlev, mmr_or_sphum='sphum', disturb=True)
    new_swflux = toa_net_sw_flux(nt, nlev, mmr_or_sphum='sphum', disturb=True)

    lw_kernel[nt,nlev,:] = new_lwflux - orig_lwflux 
    sw_kernel[nt,nlev,:] = new_swflux - orig_swflux 
    logging.info("nt=%d, nlev=%d, cost: --- %s seconds --- " % (nt, nlev, time.time() - now))


####################################################################################


if __name__ == '__main__':

    # Recording the running time
    start_time = time.time()
    logfilename = 'out_rrtm_parallel_wv.log'
    if os.path.exists(logfilename):
        os.remove(logfilename)
    logging.basicConfig(filename=logfilename,level=logging.DEBUG)

    dir_input = '../../input'

    ds = xr.open_dataset(os.path.join(dir_input, 'three_hourly_avg_data_in_one_year_rrtm.nc'), decode_times=False)

    p_full = np.array(ds.pfull) # Units: hPa
    p_half = np.array(ds.phalf)
    nlayers = len(p_full)

    lats = np.array(ds.lat)
    lons = np.array(ds.lon)
    nlat = len(lats)
    nlon = len(lons)

    T_in = ds.temp.mean(dim='lon')
    sphum_in = ds.sphum.mean(dim='lon')
    #print(np.shape(T_in))
    #print(np.shape(p_full))
    pavel = p_full[::-1]
    pz = p_half[::-1]
    pz[-1] = 1e-40

    # Temperature
    fT = ip.interp1d(p_full, T_in, kind='linear', axis=1, fill_value='extrapolate')
    tavel = fT(pavel)
    tz = fT(pz)

    # Water vapour
    q_mmr = sphum_in/(1-sphum_in)
    f = ip.interp1d(p_full, q_mmr, kind='linear', axis=1, fill_value='extrapolate')
    q_mmr_avel = f(pavel)

    fq = ip.interp1d(p_full, sphum_in, kind='linear', axis=1, fill_value='extrapolate')
    sphum_avel = fq(pavel)

    # CO2
    co2 = np.ones(tavel.shape) * 360. / 1e6

    # Ozone
    o3_zm = xr.open_dataset(os.path.join(dir_input, 'ozone_1990.nc'), decode_times=False) # mass mixing ratio
    # ozone_1990 (time: 12, pfull: 59, lat: 64, lon: 1)
    f = ip.interp1d(np.array(o3_zm.pfull), np.array(o3_zm.ozone_1990), kind='linear', axis=1, fill_value='extrapolate')
    o3_avel_zm = f(pavel)
    o3_avel_zm_tm = np.mean(o3_avel_zm, axis=(0,3))

    # Insolation or the angle
    coszen = ds.coszen.mean(dim='lon')
    zenith_angle_2d = np.rad2deg(np.arccos(coszen)) # degree

    ntime = len(ds.index) 
    logging.info(ntime)

    #lw_kernel = np.zeros((ntime, nlayers, nlat), dtype=np.double)
    #sw_kernel = np.zeros((ntime, nlayers, nlat), dtype=np.double)

    shared_array_base_lw = multiprocessing.Array(ctypes.c_double, ntime*nlayers*nlat)
    shared_array_base_sw = multiprocessing.Array(ctypes.c_double, ntime*nlayers*nlat)
    lw_kernel = np.ctypeslib.as_array(shared_array_base_lw.get_obj())
    sw_kernel = np.ctypeslib.as_array(shared_array_base_sw.get_obj())
    lw_kernel = lw_kernel.reshape((ntime, nlayers, nlat))
    sw_kernel = sw_kernel.reshape((ntime, nlayers, nlat))

    logging.info("data read and process cost: --- %s seconds --- " % (time.time() - start_time))

    # ================ Run the kernel in paralle ================ #
    now = time.time() 
    #print ('begin parallel...')

    pool = multiprocessing.Pool(processes=8)
    ntnlevs = list(itertools.product(range(ntime), range(nlayers)))
    #print(ntnlevs)

    pool.map(run_kernel, ntnlevs)
    #for ntnlev in ntnlevs:
    #    run_kernel(ntnlev)
    
    out_dt_dir = '../../kernel_data/'
    if not os.path.exists(out_dt_dir)
        os.makedirs(out_dt_dir)

    save_zonal_mean_data(lw_kernel, 'pfull', os.path.join(out_dt_dir, 'rrtm_toa_wv_lw_kernel_3h_zonalmean.nc'), 'lw_kernel')
    save_zonal_mean_data(sw_kernel, 'pfull', os.path.join(out_dt_dir, 'rrtm_toa_wv_sw_kernel_3h_zonalmean.nc'), 'sw_kernel')
 
    logging.info("Parallel cost: --- %s seconds --- " % (time.time() - now))
    logging.info("Now total cost: --- %s seconds --- " % (time.time() - start_time))
