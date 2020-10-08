from __future__ import print_function
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
import os
from functions import add_datetime_info, saturation_specific_humidity

def read_kernel(filename, varname):
    ds = xr.open_dataset(filename, decode_times=False)
    pfull = ds.pfull
    lats = ds.lat
    lons = ds.lon

    try:
        months = ds.month
    except:
        months = ds.index

    kernel = ds[varname]
    coslats = np.cos(np.deg2rad(lats))

    # Crude tropopause estimate: 100hPa in the tropics, lowering with cosine to 300hPa at the poles.
    p_tropopause= 300 - 200 * coslats
    kernel = np.array(kernel.where(kernel.pfull>=p_tropopause))
    kernel[np.isnan(kernel)] = 0.0

    kernel = xr.DataArray(kernel, coords=[months, pfull, lats, lons], dims=['month', 'pfull', 'lat', 'lon'])
    global_mean_kernel = np.average(kernel.sum(dim='pfull').mean(('month','lon')), weights=coslats, axis=0)
    kernel.attrs['phalf'] = ds.phalf

    print(varname + ': Global mean = ' + str(global_mean_kernel) + ' W/m^2/K')
    kernel.attrs['global_mean'] = global_mean_kernel
    
    return kernel

def read_rad_scheme_kernels(rad_scheme):
    '''
    rad_scheme can be 'frierson', 'byrne' or 'rrtm'
    '''
    # Read zonal-mean kernel
    datadir = '../kernel_data'

    # Read temperature kernel
    kernel_fn = rad_scheme + '_t_kernel_monthly.nc'
    t_lw_kernel = -read_kernel(os.path.join(datadir, kernel_fn), 'lw_kernel') # Note the negative value for lw kernel
    t_sw_kernel =  read_kernel(os.path.join(datadir, kernel_fn), 'sw_kernel')

    # Read water vapor kernel
    if 'frierson' not in rad_scheme.lower():
        kernel_fn = rad_scheme + 'wv_kernel_monthly.nc'
        wv_lw_kernel = -read_kernel(os.path.join(datadir, kernel_fn), 'lw_kernel')
        wv_sw_kernel =  read_kernel(os.path.join(datadir, kernel_fn), 'sw_kernel')

    # Read surface temperature kernel
    ts_file_name = os.path.join(datadir, rad_scheme + '_sfc_ts_kernel_monthly.nc')
    ds = xr.open_dataset(ts_file_name, decode_times=False)
    ts_lw_kernel = -ds.lw_kernel

    # Try to rename time index to month
    try:
        ts_lw_kernel = ts_lw_kernel.rename({'index':'month'})
    except:
        print('Time index is month already.')

    # print global mean surface temperature kernel
    coslats = np.cos(np.deg2rad(ts_lw_kernel.lat))
    ts_lw_kernel_gm = np.average(ts_lw_kernel.mean(('month', 'lon')), weights=coslats, axis=0)
    print('ts_kerenl: Global mean = '+str(ts_lw_kernel_gm) + ' W/K/m^2.')

    if 'frierson' in rad_scheme.lower():
        lw_kernels = {'ts':ts_lw_kernel, 't':t_lw_kernel}
        sw_kernels = {'t':t_sw_kernel}

        total_kernels['t'] = lw_kernels['t'] + sw_kernels['t']
        total_kernels['ts'] = lw_kernels['ts'] #+ sw_kernels['ts']
    else: 
        # add water vapor kernel
        lw_kernels['wv'] = wv_lw_kernel
        sw_kernels['wv'] = wv_sw_kernel

        total_kernels['wv'] = lw_kernels['wv'] + sw_kernels['wv']
    
    return lw_kernels, sw_kernels, total_kernels

def calc_planck_feedback_from_monthly_kernel(ds_diff, t_kernel, ts_kerenl):

    lats = ds_diff.lat
    lons = ds_diff.lon
    pfull = ds_diff.pfull
    nlat = len(lats)
    nlon = len(lons)
    nlev = len(pfull)
    coslats = np.cos(np.deg2rad(lats))

    # get monthly mean data (need to add date and time info)
    ts_diff = ds_diff['t_surf'].groupby('month').mean('time')
    #ts_diff_tm = ts_diff.mean('month')

    ts_diff_vert = np.zeros((12, nlev, nlat, nlon), dtype=np.double)
    for tt in range(12):
        for nn in range(nlev):
            ts_diff_vert[tt,nn,:,:] = ts_diff[tt,:,:]  # uniform warming as surface
    ts_diff_vert = xr.DataArray(ts_diff_vert, coords=[range(12), pfull, lats, lons],
                     dims=['month','pfull','lat', 'lon'])

    tsk = ts_diff_vert * t_kernel
    sfc_response = ts_diff * ts_kernel
    # Add the surface and air temperature response
    planck_response = tsk.sum(dim='pfull') + sfc_response
    planck_feedback = planck_response / ts_diff

    # planck_response = tsk.sum(dim='pfull').mean(dim=('month')) + sfc_response.mean('month')
    # planck_feedback = planck_response / ts_diff_tm    # time averaged

    # Print global average
    planck_response_gm = np.average(planck_response.mean(('month', 'lon')), weights=coslats, axis=0)
    ts_diff_gm = np.average(ts_diff.mean(('month', 'lon')), weights=coslats, axis=0)
    planck_fb_gm = planck_response_gm / ts_diff_gm
    print('Global mean Planck feedback parameter is ' + str(planck_fb_gm))

    return planck_feedback  #(month, lat, lon)

def calc_lapse_rate_feedback_from_monthly_kernel(ds_diff, t_kernel):

    lats = ds_diff.lat
    lons = ds_diff.lon
    pfull = ds_diff.pfull
    nlat = len(lats)
    nlon = len(lons)
    nlev = len(pfull)
    coslats = np.cos(np.deg2rad(lats))

    # get monthly mean data (need to add date and time info)
    ta_diff = ds_diff['temp'].groupby('month').mean('time')
    ts_diff = ds_diff['t_surf'].groupby('month').mean('time')
    #ts_diff_tm = ts_diff.mean('month')

    ts_diff_vert = np.zeros((12, nlev, nlat, nlon), dtype=np.double)
    for tt in range(12):
        for nn in range(nlev):
            ts_diff_vert[tt,nn,:,:] = ts_diff[tt,:,:]  # uniform warming as surface
    ts_diff_vert = xr.DataArray(ts_diff_vert, coords=[range(12), pfull, lats, lons], 
                            dims=['month', 'pfull', 'lat', 'lon'])

    del_tair_ts = ta_diff - ts_diff_vert
    del_tair_ts = xr.DataArray(del_tair_ts, coords=[range(12), pfull, lats, lons], 
                            dims=['month', 'pfull', 'lat', 'lon'])

    tempk = del_tair_ts * t_kernel
    lapse_rate_feedback = tempk.sum(dim='pfull') / ts_diff

    # Print global average
    tempk_gm = np.average(tempk.mean(('month', 'lon')), weights=coslats, axis=0)
    ts_diff_gm = np.average(ts_diff.mean(('month', 'lon')), weights=coslats, axis=0)
    lapse_rate_fb_gm = tempk_gm / ts_diff_gm
    print('Global mean lapse rate feedback parameter is ' + str(lapse_rate_fb_gm))

    return lapse_rate_feedback  #(month, lat, lon)

def calc_water_vapor_feedback_from_monthly_kernel(ds1, ds_diff, wv_kernel): 
    '''
    Reference:
        Shell, K. M., Kiehl, J. T., & Shields, C. A. (2008). Using the radiative kernel 
        technique to calculate climate feedbacks in NCAR’s Community Atmospheric Model.
        Journal of Climate, 21(10), 2269-2282.
    ---------------------------------------------------------------------------------
        Sec 2 Methodology (P2271):
        
        For the case of the water vapor feedback, we multiply the kernel by 
        the change in the natural log of the water vapor, divided by the change
        in the natural log of water vapor using in the kernel calculation, 
        since absorption of radiation by water vapor is roughly proportional to ln(q).
    '''
    
    # get monthly mean data (need to add date and time info)
    #ta_diff = ds_diff['temp'].groupby('month').mean('time')
    ts_diff = ds_diff['t_surf'].groupby('month').mean('time')
    q_diff = ds_diff['sphum'].groupby('month').mean('time')
    q1 = ds1['sphum'].groupby('month').mean('time')

    qs1_t = saturation_specific_humidity(ds1.temp)
    qs2_t = saturation_specific_humidity(ds1.temp + 1.0)
    qs1 = qs1_t.groupby('month').mean('time')
    qs2 = qs2_t.groupby('month').mean('time')

    dT = 1.0
    dqs_dT = (qs2 - qs1) / dT # dT is 1K
    rh = q1 / qs1          # assuming RH is fixed when warming
    dq_dT = rh * dqs_dT       
    dlogq_dT = dq_dT / q1  # natural log of water vapor using in the kernel calculation
    dlogq = q_diff / q1    # change in the natural log of the water vapor in perturbed exps

    response = wv_kernel * dlogq / dlogq_dT
    wv_feedback = response.sum(dim='pfull')) / ts_diff

    return wv_feedback # (month, lat, lon)

# --------------- Main Program --------------- #
if __name__ == "__main__":

    rad_scheme = 'byrne'
    
    # Read kernels
    lw_kernels, sw_kernels, total_kernels = read_rad_scheme_kernels(rad_scheme)
    t_kernel = total_kernels['t']
    ts_kernel = total_kernels['ts']
    if 'frierson' not in rad_scheme.lower():
        wv_kernel = total_kernels['wv']

    # Read dataset
    ds1 = ... # xr.open_dataset()
    ds2 = ... # xr.open_dataset()

    # Add date and time info to dataset
    add_datetime_info(ds1)
    add_datetime_info(ds2)

    # difference dataset
    ds_diff = ds2 - ds1

    # ----------- PLANCK FEEDBACK ------------- #
    planck_feedback = calc_planck_feedback_from_monthly_kernel(ds_diff, t_kernel, ts_kernel)
    
    # ----------- LAPSERATE FEEDBACK ------------- #
    lapse_rate_feedback = calc_lapse_rate_feedback_from_monthly_kernel(ds_diff, t_kernel)
    
    # ----------- WATER VAPOR FEEDBACK ------------- #
    if 'frierson' not in rad_scheme:
        water_vapor_feedback = calc_water_vapor_feedback_from_monthly_kernel(ds1, ds_diff, wv_kernel)
