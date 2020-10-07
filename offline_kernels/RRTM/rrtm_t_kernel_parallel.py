from __future__ import print_function
import pyrrtm
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
import ctypes


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
    dt.to_netcdf(filename)


def toa_net_lw_flux(nt, nlev, disturb=False):
    lw_toa_lat = np.zeros(nlat)

    for i in range(nlat):
        lw = pyrrtm.LW(nlayers)
        lw.tz = tz[nt,:,i]
        lw.pz = pz

        # If disturb the temperature at certain level
        if disturb:
            lw.tavel[nlev] = lw.tavel[nlev] + 1.0

        lw.set_species('co2', co2[nt,:,i], 'vmr')
        lw.set_species('h2o', q_mmr[nt,:,i], 'mmr')

        # run the radiation code
        output = lw.run()
        # fnet(nlayers+1,)
        lw_fnet = output.fnet
        lw_toa_lat[i] = lw_fnet[nlayers]

    return lw_toa_lat


def toa_net_sw_flux(nt, nlev, disturb=False):
    sw_toa_lat = np.zeros(nlat)

    for i in range(nlat):
        sw = pyrrtm.SW(nlayers)
        sw.tz = tz[nt,:,i]
        sw.pz = pz

        # If disturb the temperature at certain level
        if disturb:
            sw.tavel[nlev] = sw.tavel[nlev] + 1.0

        sw.set_species('co2', co2[nt,:,i], 'vmr')
        sw.set_species('h2o', q_mmr[nt,:,i], 'mmr')
        sw.set_species('o3', o3_zm_tm[:,i], 'mmr')

        # Surface
        albedo = 0.3
        sw.semis = 1.0-albedo 

        # Solar Forcing Parameters
        sw.juldat = 90
        sw.sza = zenith_angle_2d[nt,i]

        # run the radiation code
        output = sw.run()
        sw_fnet = output.fnet
        sw_toa_lat[i] = sw_fnet[nlayers]

    return sw_toa_lat


def run_kernel(nt_nlev):
    now = time.time() 
    nt = nt_nlev[0]
    nlev = nt_nlev[1]
    
    # Original net longwave and shortwave flux at TOA
    orig_lwflux = toa_net_lw_flux(nt, nlev, disturb=False)
    orig_swflux = toa_net_sw_flux(nt, nlev, disturb=False)
    
    # Net longwave and shortwave flux at TOA after perturbation
    new_lwflux = toa_net_lw_flux(nt, nlev, disturb=True)
    new_swflux = toa_net_sw_flux(nt, nlev, disturb=True)

    lw_kernel[nt,nlev,:] = new_lwflux - orig_lwflux 
    sw_kernel[nt,nlev,:] = new_swflux - orig_swflux 
    logging.info("nt=%d, nlev=%d, cost: --- %s seconds --- " % (nt, nlev, time.time() - now))



if __name__ == '__main__':

    # Recording the running time
    start_time = time.time()
    logfilename = 'out.rrtm.parallel_t_kernel_v2_3h.log'
    if os.path.exists(logfilename):
        os.remove(logfilename)
    logging.basicConfig(filename=logfilename,level=logging.DEBUG)

    dir_input = '../../input'

    # Read the input dataset
    ds = xr.open_dataset(os.path.join(dir_input, 'three_hourly_avg_data_in_one_year_rrtm.nc'), decode_times=False)

    p_full = np.array(ds.pfull) # Units: hPa
    p_half = np.array(ds.phalf)
    nlayers = len(p_full)

    lats = np.array(ds.lat)
    lons = np.array(ds.lon)
    nlat = len(lats)
    nlon = len(lons)

    # Using 3h ds
    ntime = len(ds.index) 
    logging.info(ntime)

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
    q_mmr_in = sphum_in/(1-sphum_in)
    f = ip.interp1d(p_full, q_mmr_in, kind='linear', axis=1, fill_value='extrapolate')
    q_mmr = f(pavel)

    # CO2
    co2 = np.ones(tavel.shape) * 360. / 1.0e6

    # Ozone file: ozone_1990(time: 12, pfull: 59, lat: 64, lon: 1)
    o3 = xr.open_dataset(os.path.join(dir_input, 'ozone_1990.nc'), decode_times=False) # mass mixing ratio
    f = ip.interp1d(np.array(o3.pfull), np.array(o3.ozone_1990), kind='linear', axis=1, fill_value='extrapolate')
    o3_avel = f(pavel)
    o3_zm_tm = np.mean(o3_avel, axis=(0,3)) # Zonal and annual mean

    # Insolation: zenith angle in degree
    coszen = ds.coszen.mean(dim='lon')
    zenith_angle_2d = np.rad2deg(np.arccos(coszen))

    #lw_kernel = np.zeros((ntime, nlayers, nlat), dtype=np.double)
    #sw_kernel = np.zeros((ntime, nlayers, nlat), dtype=np.double)

    # Create shared memory among the processors
    shared_array_base_lw = multiprocessing.Array(ctypes.c_double, ntime*nlayers*nlat)
    shared_array_base_sw = multiprocessing.Array(ctypes.c_double, ntime*nlayers*nlat)

    lw_kernel = np.ctypeslib.as_array(shared_array_base_lw.get_obj())
    sw_kernel = np.ctypeslib.as_array(shared_array_base_sw.get_obj())

    lw_kernel = lw_kernel.reshape((ntime, nlayers, nlat))
    sw_kernel = sw_kernel.reshape((ntime, nlayers, nlat))

    logging.info("data read and process cost: --- %s seconds --- " % (time.time() - start_time))

    now = time.time() 

    # Using 8 cores to run
    ncores = 8
    pool = multiprocessing.Pool(processes=ncores)

    # Produce the parameters list
    nt_nlevs = list(itertools.product(range(ntime), range(nlayers)))
    # Run the scripts in parallel
    pool.map(run_kernel, nt_nlevs)
    #for nt_nlev in nt_nlevs:
    #    run_kernel(nt_nlev)

    out_dt_dir = '../../kernel_data/'
    if not os.path.exists(out_dt_dir):
        os.makedirs(out_dt_dir)

    save_zonal_mean_data(lw_kernel, 'pfull', os.path.join(out_dt_dir, 'rrtm_toa_t_lw_kernel_3h_zonalmean.nc'), 'lw_kernel')
    save_zonal_mean_data(sw_kernel, 'pfull', os.path.join(out_dt_dir, 'rrtm_toa_t_sw_kernel_3h_zonalmean.nc'), 'sw_kernel')

    logging.info("Parallel cost: --- %s seconds --- " % (time.time() - now))
    logging.info("Now total cost: --- %s seconds --- " % (time.time() - start_time))
