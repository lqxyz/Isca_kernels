from __future__ import print_function
import numpy as np
import xarray as xr
import model_constants as mc
import os
import logging
import time
from functions import write_vars_to_nc_file


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


def lw_down_byrne(sph):
    #dtau/ds = a*mu + b*q
    #ref: Byrne, M. P. & O'Gorman, P. A.
    #Land-ocean warming contrast over a wide range of climates:
    #Convective quasi-equilibrium theory and idealized simulations.
    #J. Climate 26, 4000-4106 (2013).
    lw_down   = xr.DataArray(np.zeros((nlev+1, nlat, nlon)), [('phalf', p_half), ('lat', lats), ('lon', lons)])
    lw_dtrans = xr.DataArray(np.zeros((nlev,   nlat, nlon)), [('pfull', p_full), ('lat', lats), ('lon', lons)])

    # !!! NOTICE: q and b -> (lev, lat, lon), lw_down, lw_dtrans, lw_del_tau --> (lat, lon, lev) !!!
    for k in range(0, nlev):
        lw_del_tau    = (bog_a*bog_mu + 0.17 * np.log(carbon_conc/360.)  + bog_b*sph[k,:,:]) * (( p_half[k+1]-p_half[k] ) / p_half[nlev])
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

def lw_up_fb(lw_dtrans):
    #compute upward longwave flux by integrating upward
    lw_up     = xr.DataArray(np.zeros((nlev+1, nlat, nlon)), [('phalf', p_half), ('lat', lats), ('lon', lons)]) 
    lw_up[nlev,:,:]    = b_surf
    for k in range(nlev-1,-1,-1):
        lw_up[k,:,:]   = lw_up[k+1,:,:]*lw_dtrans[k,:,:] + b[k,:,:]*(1.0 - lw_dtrans[k,:,:])
    return lw_up

def es_from_clausius_clapeyron(T_in_K):
    return 611. * np.exp(17.67*(T_in_K-273.15) / (T_in_K-29.65))

def sphum_level_perturb(specific_hum, temperature, level_idx):
    sphum_tmp = xr.DataArray(np.zeros((nlev, nlat, nlon)), [('pfull', p_full), ('lat', lats), ('lon', lons)]) 
    for i in range(0,nlev):
        if i != level_idx:
            sphum_tmp[i,:,:] = specific_hum[i,:,:]
        else:
            t_tmp = temperature[i,:,:]
            sphum_tmp[i,:,:] = es_from_clausius_clapeyron(t_tmp+1)/es_from_clausius_clapeyron(t_tmp)*specific_hum[i,:,:]
    return sphum_tmp

def toa_lw_sw_rad_flux(sphum3d):
    # Functions used determined by scheme choice at beginning of script
    if rad_scheme == 'frierson':
        sw_down, sw_up = sw_fb()
        lw_down, lw_dtrans = lw_down_frierson()
        lw_up = lw_up_fb(lw_dtrans)

    elif rad_scheme == 'byrne':
        sw_down, sw_up = sw_fb()
        lw_down, lw_dtrans = lw_down_byrne(sphum3d)
        lw_up = lw_up_fb(lw_dtrans)    
    else:
        print "Invalid scheme choice"

    # net fluxes (positive up)
    lw_flux  = lw_up - lw_down
    sw_flux  = sw_up - sw_down
    rad_flux = lw_flux + sw_flux

    return lw_flux[0,:,:], sw_flux[0,:,:], rad_flux[0,:,:]

def monthly_mean_for_3hourly_datas(datas):
    newdatas = []
    interval = 24/3*30
    for dt in datas:
        shp = np.shape(dt)
        monthlydtsize = (12, shp[1], shp[2], shp[3])
        monthlydt = np.zeros(monthlydtsize, dtype=np.double)
        for i in range(12):
            monthlydt[i,:,:,:] = np.mean(dt[i*interval:(i+1)*interval,:,:,:], axis=0)
        newdatas.append(monthlydt)
    return newdatas


def write_kernel_data(out_dir, rad_scheme):
    
    latbs = np.array(ds.latb)
    lonbs = np.array(ds.lonb)

    pfull = np.array(ds.pfull)
    phalf = np.array(ds.phalf)

    index_arr = np.array(ds.index)
    number_dict = {'nlat': nlat, 'nlon': nlon, 'nlatb':np.size(latbs), 
                   'nlonb':np.size(lonbs), 'npfull':np.size(pfull), 
                   'nphalf':np.size(phalf), 'nindex':np.size(index_arr)}

    datas = [diff_lw_flux, diff_sw_flux, diff_rad_flux]
    varnames = ['lw_kernel', 'sw_kernel', 'rad_kernel']

    vars_dim = {}
    vars_units = {}
    for var in varnames:
        vars_dim[var] = ('index', 'pfull', 'lat', 'lon',)
        vars_units[var] = 'W/m^2/K/100hPa'

    file_name = os.path.join(out_dir, rad_scheme+'_toa_wv_kernel_3h.nc')
    write_vars_to_nc_file(file_name, varnames, datas, lats, lons, latbs, 
                        lonbs, pfull, phalf, None, None,  number_dict, 
                        vars_dim, index_arr, vars_units=vars_units)

    ######## Radiative kernel for monthly mean value ############
    datas_monthly_mean = monthly_mean_for_3hourly_datas(datas)

    index_arr = range(12)
    number_dict = {'nlat': nlat, 'nlon': nlon, 'nlatb':np.size(latbs), 'nlonb':np.size(lonbs), 
                'npfull':np.size(pfull), 'nphalf':np.size(phalf), 'nindex':12}
    vars_dim = {}
    for var in varnames:
        vars_dim[var] = ('index', 'pfull', 'lat', 'lon')
        vars_units[var] = 'W/m^2/K/100hPa'

    file_name = os.path.join(outdir, rad_scheme+'_toa_wv_kernel_monthly.nc')
    write_vars_to_nc_file(datas_monthly_mean, lats, lons, latbs, lonbs, pfull, phalf, None, None, 
                file_name, varnames, number_dict, vars_dim, index_arr, vars_units=vars_units, index_units='1')


#############################################################################################

if __name__ == "__main__":

    start_time = time.time()

    # Select choice of radiation scheme
    rad_scheme = 'byrne'

    logfilename = 'output_radiative_kernel_wv_'+rad_scheme+'.log'
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

    # Prescribe SST
    #yy = lats * 90/60 * np.pi/180
    #yy = np.deg2rad(lats) * 90/60.
    #sst = 27 * (1 - np.sin(yy)**2)
    #sst[abs(lats) >= 60] = 0
    #t_surf = sst + 273.15

    t_surf = ds.t_surf

    ntime = len(ds.index)

    diff_lw_flux  = np.zeros((ntime, nlev, nlat, nlon), dtype='double')
    diff_sw_flux  = np.zeros((ntime, nlev, nlat, nlon), dtype='double')
    diff_rad_flux = np.zeros((ntime, nlev, nlat, nlon), dtype='double')

    #b_surf = mc.stefan*t_surf**4
    #b_surf = np.transpose(np.tile(mc.stefan*t_surf**4, (nlon,1)))
    b_surf_3h = mc.stefan*t_surf**4

    for hour_index in range(0,ntime):
        now = time.time()
        #print 'Hour = '+str(hour_index)
        logging.info('Hour = '+str(hour_index))

        insolation = insolation_3h[hour_index, :, :]
        b_surf = b_surf_3h[hour_index, :, :]

        t = ds.temp[hour_index,:,:,:]
        b = mc.stefan*t**4

        q = ds.sphum[hour_index,:,:,:]
        toa_lw_old, toa_sw_old, toa_rad_old = toa_lw_sw_rad_flux(q)

        for lev_idx in range(0, nlev):
            #print ' level = '+str(lev_idx)
            logging.info('      level = '+str(lev_idx))
            sphum_perturb = sphum_level_perturb(q, t, lev_idx)
            toa_lw_new, toa_sw_new, toa_rad_new = toa_lw_sw_rad_flux(sphum_perturb)
            # Black body emission from atmos and surface
            
            dp = p_half[lev_idx+1] - p_half[lev_idx]
            diff_lw_flux[hour_index, lev_idx,:,:] = (toa_lw_new - toa_lw_old) / dp * 1e2
            diff_sw_flux[hour_index, lev_idx,:,:] = (toa_sw_new - toa_sw_old) / dp * 1e2
            diff_rad_flux[hour_index, lev_idx,:,:] = (toa_rad_new - toa_rad_old) / dp * 1e2

        logging.info("Each step cost: --- %s seconds --- " % (time.time() - now))
        logging.info("Now total cost: --- %s seconds --- " % (time.time() - start_time))

    logging.info(np.shape(diff_rad_flux))

    # Write the flux data into file
    write_kernel_data(out_dir, rad_scheme)

    logging.info("Total time: --- %s seconds ---.\n" % (time.time() - start_time))
