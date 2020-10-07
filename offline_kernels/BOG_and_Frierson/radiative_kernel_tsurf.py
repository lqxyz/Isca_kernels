"""Re-create two stream radiation code from GFDL-MiMA in offline python version"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.io import loadmat
import model_constants as mc
import sys, os
sys.path.insert(0, '/home/links/ql260/Documents/Exps_Analysis/read_write_file/')
from nc_write_vars_to_file import *
import logging
import time


def sw_fb():
    #Frierson handling of SW radiation
    
    #SW optical thickness
    sw_tau_0 = np.transpose(np.tile((1.0 - sw_diff*np.sin(lat_rad)**2)*atm_abs, (nlon,1) ))

    #compute optical depths for each model level
    sw_down   = xr.DataArray(np.zeros((nlev+1, nlat, nlon)), [('phalf', p_half), ('lat', lats), ('lon', lons)])
    sw_up     = xr.DataArray(np.zeros((nlev+1, nlat, nlon)), [('phalf', p_half), ('lat', lats), ('lon', lons)])
    sw_tau    = xr.DataArray(np.zeros((nlev+1, nlat, nlon)), [('phalf', p_half), ('lat', lats), ('lon', lons)])
    
    for k in range(0, nlev+1):
        sw_tau[k,:,:] = sw_tau_0 * (p_half[k]/mc.pstd_mks)**solar_exponent
    
    #compute downward shortwave flux
    for k in range(0, nlev+1):
        sw_down[k,:,:]   = insolation * np.exp(-sw_tau[k,:,:])
    
    for k in range(0, nlev+1):
        sw_up[k,:,:]   = albedo * sw_down[nlev,:,:]
        
    return sw_down, sw_up


def lw_down_byrne():
    #dtau/ds = a*mu + b*q
    #ref: Byrne, M. P. & O'Gorman, P. A.
    #Land-ocean warming contrast over a wide range of climates:
    #Convective quasi-equilibrium theory and idealized simulations.
    #J. Climate 26, 4000-4106 (2013).
    lw_down   = xr.DataArray(np.zeros((nlev+1, nlat, nlon)), [('phalf', p_half), ('lat', lats), ('lon', lons)])
    lw_dtrans = xr.DataArray(np.zeros((nlev,   nlat, nlon)), [('pfull', p_full), ('lat', lats), ('lon', lons)])

    # !!! NOTICE: q and b -> (lev, lat, lon), lw_down, lw_dtrans, lw_del_tau --> (lat, lon, lev) !!!
    for k in range(0, nlev):
        lw_del_tau    = (bog_a*bog_mu + 0.17 * np.log(carbon_conc/360.)  + bog_b*q[k,:,:]) * (( p_half[k+1]-p_half[k] ) / p_half[nlev])
        lw_dtrans[k,:,:] = np.exp( - lw_del_tau )
        
    #compute downward longwave flux by integrating downward
    
    for k in range(0, nlev):
        lw_down[k+1,:,:] = lw_down[k,:,:]*lw_dtrans[k,:,:] + b[k,:,:]*(1. - lw_dtrans[k,:,:])
    
    return lw_down, lw_dtrans
    

def lw_down_frierson():
    #longwave optical thickness function of latitude and pressure
    
    lw_down   = xr.DataArray(np.zeros((nlev+1, nlat, nlon)), [('phalf', p_half), ('lat', lats), ('lon', lons)])
    lw_tau    = xr.DataArray(np.zeros((nlev+1, nlat, nlon)), [('phalf', p_half), ('lat', lats), ('lon', lons)])
    lw_dtrans = xr.DataArray(np.zeros((nlev,   nlat, nlon)), [('pfull', p_full), ('lat', lats), ('lon', lons)])
    
    lw_tau_0 = np.transpose(np.tile(ir_tau_eq + (ir_tau_pole - ir_tau_eq)*np.sin(lat_rad)**2, (nlon,1)) )
    
    #compute optical depths for each model level
    for k in range(0, nlev+1):
        lw_tau[k,:,:] = (lw_tau_0 * ( linear_tau * p_half[k]/mc.pstd_mks
                     + (1.0 - linear_tau) * (p_half[k]/mc.pstd_mks)**wv_exponent ))
    
    #longwave differential transmissivity
    for k in range(0, nlev):
        lw_dtrans[k,:,:] = np.exp( -(lw_tau[k+1,:,:] - lw_tau[k,:,:]) )
    
    #compute downward longwave flux by integrating downward
    lw_down[0,:,:]      = 0.
    for k in range(0, nlev):
        lw_down[k+1,:,:] = lw_down[k,:,:]*lw_dtrans[k,:,:] + b[k,:,:]*(1. - lw_dtrans[k,:,:])
    
    return lw_down, lw_dtrans

def lw_up_fb(lw_dtrans, b_surf):
    #compute upward longwave flux by integrating upward
    lw_up     = xr.DataArray(np.zeros((nlev+1, nlat, nlon)), [('phalf', p_half), ('lat', lats), ('lon', lons)]) 
    lw_up[nlev,:,:]    = b_surf
    for k in range(nlev-1,-1,-1):
        lw_up[k,:,:]   = lw_up[k+1,:,:]*lw_dtrans[k,:,:] + b[k,:,:]*(1.0 - lw_dtrans[k,:,:])
    return lw_up

def temp_level_perturb(temperature, level_idx):
    t_tmp = xr.DataArray(np.zeros((nlev, nlat, nlon)), [('pfull', p_full), ('lat', lats), ('lon', lons)]) 
    for i in range(0,nlev):
        if i != level_idx:
            t_tmp[i,:,:] = temperature[i,:,:]
        else:
            t_tmp[i,:,:] = temperature[i,:,:] + 1.0
    return t_tmp

def toa_lw_sw_rad_flux(ts):
    
    b_surf = mc.stefan*ts**4

    # Functions used determined by scheme choice at beginning of script
    if rad_scheme == 'frierson':
        sw_down, sw_up = sw_fb()
        lw_down, lw_dtrans = lw_down_frierson()
        lw_up = lw_up_fb(lw_dtrans, b_surf)

    elif rad_scheme == 'byrne':
        sw_down, sw_up = sw_fb()
        lw_down, lw_dtrans = lw_down_byrne()

        lw_up = lw_up_fb(lw_dtrans, b_surf)    
    else:
        print "Invalid scheme choice"

    # net fluxes (positive up)
    lw_flux  = lw_up - lw_down
    sw_flux  = sw_up - sw_down
    rad_flux = lw_flux + sw_flux

    return lw_flux[0,:,:], sw_flux[0,:,:], rad_flux[0,:,:]

# 3 dimensional data
def monthly_mean_for_3hourly_datas(datas):
    newdatas = []
    interval = 24/3*30
    for dt in datas:
        shp = np.shape(dt)
        monthlydtsize = (12, shp[1], shp[2])
        monthlydt = np.zeros(monthlydtsize, dtype=np.double)
        for i in range(12):
            monthlydt[i,:,:] = np.mean(dt[i*interval:(i+1)*interval,:,:], axis=0)
        newdatas.append(monthlydt)

    return newdatas


def write_kernel_data(out_dir, rad_scheme):
    
    latbs = np.array(ds.latb)
    lonbs = np.array(ds.lonb)
    index_arr = np.array(ds.index)

    number_dict = {'nlat': nlat, 'nlon': nlon, 'nlatb':np.size(latbs), 
                   'nlonb':np.size(lonbs), 'nindex':np.size(index_arr)}

    datas = [diff_lw_flux, diff_sw_flux, diff_rad_flux]
    varnames = ['lw_kernel', 'sw_kernel', 'rad_kernel']

    vars_dim = {}
    vars_units = {}
    for var in varnames:
        vars_dim[var] = ('index', 'lat', 'lon',)
        vars_units[var] = 'W/m^2/K'

    file_name = os.path.join(out_dir, rad_scheme+'_toa_ts_kernel_3h.nc')
    write_vars_to_nc_file(file_name, varnames, datas, lats, lons, latbs, 
                        lonbs, None, None, None, None,  number_dict, 
                        vars_dim, index_arr=index_arr, vars_units=vars_units)

    ######## Radiative kernel for monthly mean value ############
    datas_monthly_mean = monthly_mean_for_3hourly_datas(datas)

    index_arr = range(12)
    number_dict = {'nlat': nlat, 'nlon': nlon, 'nlatb':np.size(latbs), 
                   'nlonb':np.size(lonbs), 'nindex':12}
    vars_dim = {}
    for var in varnames:
        vars_dim[var] = ('index', 'lat', 'lon')
        vars_units[var] = 'W/m^2/K'

    file_name = os.path.join(outdir, rad_scheme+'_toa_ts_kernel_monthly.nc')
    write_vars_to_nc_file(datas_monthly_mean, lats, lons, latbs, lonbs, None, None, None, None, 
                file_name, varnames, number_dict, vars_dim, index_arr, vars_units=vars_units, index_units='1')


#############################################################################################


if __name__ == "__main__":

    start_time = time.time()

    # Select choice of radiation scheme
    rad_scheme = 'byrne'

    logfilename = 'output_radiative_kernel_tsurf_'+rad_scheme+'.log'
    if os.path.exists(logfilename):
        os.remove(logfilename)

    logging.basicConfig(filename=logfilename,level=logging.DEBUG)

    # module constants
    solar_constant  = 1360.0
    del_sol         = 1.4
    del_sw          = 0.0
    ir_tau_eq       = 6.0
    ir_tau_pole     = 1.5
    atm_abs         = 0.0
    sw_diff         = 0.0
    linear_tau      = 0.1
    wv_exponent     = 4.0
    solar_exponent  = 4.0
    albedo          = 0.3
    carbon_conc     = 360.0

    # parameters for Byrne and OGorman radiation scheme
    bog_a = 0.1627 #0.8678
    bog_b = 1997.9
    bog_mu = 1.0

    input_dir = '../../input'
    out_dir = '../../kernel_data'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Input profiles 
    ds = xr.open_dataset(os.path.join(input_dir, 'three_hourly_avg_data_in_one_year_'+rad_scheme+'.nc'), decode_times=False)

    p_full = np.array(ds.pfull*1e2)
    p_half = np.array(ds.phalf*1e2)
    nlev = len(p_full)

    lats = np.array(ds.lat)
    lons = np.array(ds.lon)
    nlat = len(lats)
    nlon = len(lons)

    # Insolation
    lat_rad = np.deg2rad(lats) # lat in rad, but lats in degree
    #p2 = (1. - 3.*np.sin(lat_rad)**2)/4.
    # Insolation for Frierson and Byrne
    #insolation  = np.transpose(np.tile( 0.25 * solar_constant * (1.0 + del_sol * p2 + del_sw * np.sin(lat_rad)), (nlon,1)))

    # Read the diurnal coszen and calculate insolation
    coszen = ds.coszen
    insolation_3h = solar_constant * coszen
    t_surf_3h = ds.t_surf

    ntime = len(ds.index)

    diff_lw_flux_all  = np.zeros((ntime, nlat, nlon), dtype='double')
    diff_sw_flux_all  = np.zeros((ntime, nlat, nlon), dtype='double')
    diff_rad_flux_all = np.zeros((ntime, nlat, nlon), dtype='double')

    for hour_index in range(0,ntime):
        now = time.time()
        logging.info('Hour = '+str(hour_index))

        insolation = insolation_3h[hour_index,:,:]
        t_surf = t_surf_3h[hour_index,:,:]

        q = ds.sphum[hour_index,:,:,:]
        t = ds.temp[hour_index,:,:,:]
        b = mc.stefan*t**4
        toa_lw_old, toa_sw_old, toa_rad_old = toa_lw_sw_rad_flux(t_surf)
        toa_lw_new, toa_sw_new, toa_rad_new = toa_lw_sw_rad_flux(t_surf+1.0)
            
        diff_lw_flux_all[hour_index,:,:] = toa_lw_new - toa_lw_old
        diff_sw_flux_all[hour_index,:,:] = toa_sw_new - toa_sw_old
        diff_rad_flux_all[hour_index,:,:] = toa_rad_new - toa_rad_old
        logging.info("Each step cost: --- %s seconds --- " % (time.time() - now))
        logging.info("Now total cost: --- %s seconds --- " % (time.time() - start_time))

    logging.info(np.shape(diff_rad_flux_all))

    write_kernel_data(out_dir, rad_scheme)

    logging.info("Total time: --- %s seconds ---.\n" % (time.time() - start_time))
