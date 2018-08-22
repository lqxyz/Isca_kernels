# -*- coding: utf-8 -*-
from netCDF4 import Dataset 

def write_vars_to_nc_file(file_name, variable_names, datas, lats, lons, 
                         latbs, lonbs, p_full, p_half, time_arr, time_units, 
                         number_dict, vars_dims, index_arr=None, vars_units=None):
    '''
    Function to write data to a netcdf file
    ''' 
    # Create output file to write to
    output_file = Dataset(file_name, 'w', format='NETCDF4_CLASSIC')
    
    # Identify whether we have pressure and time dimensions
    if p_full is None and p_half is None:
        pressure_dim = False
    else:
        pressure_dim = True
    
    if time_arr is None:
        time_dim = False
    else:
        time_dim = True

    if index_arr is None:
        index_dim = False
    else:
        index_dim = True

    if lons is None:
        zonal_dim = True
    else:
        zonal_dim = False

    # Create dimensions
    lat = output_file.createDimension('lat', number_dict['nlat'])
    latb = output_file.createDimension('latb', number_dict['nlatb'])

    if zonal_dim == False:
        lon = output_file.createDimension('lon', number_dict['nlon'])
        lonb = output_file.createDimension('lonb', number_dict['nlonb'])
    
    if pressure_dim:
        pfull = output_file.createDimension('pfull', number_dict['npfull'])
        phalf = output_file.createDimension('phalf', number_dict['nphalf'])

    if time_dim:
        time = output_file.createDimension('time', None) 

    if index_dim:
        index = output_file.createDimension('index', number_dict['nindex']) 
    
    # Create variables for dimensions
    latitudes = output_file.createVariable('lat', 'd', ('lat',))
    latitudebs = output_file.createVariable('latb', 'd', ('latb',))

    if zonal_dim == False:
        longitudes = output_file.createVariable('lon', 'd', ('lon',))
        longitudebs = output_file.createVariable('lonb', 'd', ('lonb',))
    
    if pressure_dim:
        pfulls = output_file.createVariable('pfull', 'd', ('pfull',))
        phalfs = output_file.createVariable('phalf', 'd', ('phalf',))
    
    if time_dim:
        times = output_file.createVariable('time', 'd', ('time',))

    if index_dim:
        index = output_file.createVariable('index', 'i', ('index',))
    
    #Create units for dimensions
    latitudes.units = 'degrees_N'
    latitudes.cartesian_axis = 'Y'
    latitudes.edges = 'latb'
    latitudes.long_name = 'latitude'

    latitudebs.units = 'degrees_N'
    latitudebs.cartesian_axis = 'Y'
    latitudebs.long_name = 'latitude edges'

    if zonal_dim == False:
        longitudes.units = 'degrees_E'
        longitudes.cartesian_axis = 'X'
        longitudes.edges = 'lonb'
        longitudes.long_name = 'longitude'

        longitudebs.units = 'degrees_E'
        longitudebs.cartesian_axis = 'X'
        longitudebs.long_name = 'longitude edges'
    
    if pressure_dim:
        pfulls.units = 'hPa'
        pfulls.cartesian_axis = 'Z'
        pfulls.positive = 'down'
        pfulls.long_name = 'full pressure level'
    
        phalfs.units = 'hPa'
        phalfs.cartesian_axis = 'Z'
        phalfs.positive = 'down'
        phalfs.long_name = 'half pressure level'
    
    if time_dim:
        times.units = time_units
        times.calendar = 'THIRTY_DAY_MONTHS'
        times.calendar_type = 'THIRTY_DAY_MONTHS'
        times.cartesian_axis = 'T'

    if index_dim:
        index.long_name = 'index of the data'
    
    # Create variable in output file
    output_array_netcdf = []
    for var_name in variable_names:
        var = output_file.createVariable(var_name, 'f4', vars_dims[var_name])
        var.units = vars_units[var_name]
        output_array_netcdf.append(var)

    # Fill NetCDF with data
    latitudes[:] = lats
    longitudes[:] = lons
    
    latitudebs[:] = latbs
    longitudebs[:] = lonbs
    
    if pressure_dim:
        pfulls[:] = p_full
        phalfs[:] = p_half
    
    if time_dim:
        times[:] = time_arr
    
    if index_dim:
        index[:] = index_arr

    for dt, output_array_file in zip(datas, output_array_netcdf):
        output_array_file[:] = dt
    
    # Save and close
    output_file.close()
