from __future__ import print_function
import numpy as np
import xarray as xr
import netCDF4

def add_datetime_info(data):
    time = data.time
    try:
        dates = netCDF4.num2date(time, time.units, time.calendar)
    except:
        print('No calendar attribute in time.')
        dates = netCDF4.num2date(time, time.units)

    years = []
    months = []
    seasons = []
    days = []
    hours = []
    for now in dates:
        years.append(now.year)
        seasons.append((now.month%12 + 3)//3)
        months.append(now.month)
        days.append(now.day)
        hours.append(now.hour)
    data.coords['month'] = ('time', months)
    data.coords['year'] = ('time', years)
    data.coords['season'] = ('time', seasons)
    data.coords['day'] = ('time', days)
    data.coords['hour'] = ('time', hours)

def saturation_specific_humidity(temp):
    '''
    https://unidata.github.io/MetPy/latest/_modules/metpy/calc/thermo.html

    Calculate the saturation water vapor (partial) pressure.
    Reference:
    Bolton, David, 1980: The computation of equivalent potential temperature 
    Monthly Weather Review, vol. 108, no. 7 (july),  p. 1047, eq.(10) 
    http://dx.doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2 

    The formula used is that from [Bolton1980] for T in degrees Celsius:
    6.112 e^\frac{17.67T}{T + 243.5}

    # Converted from original in terms of C to use kelvin. Using raw absolute values of C in
    # a formula plays havoc with units support.
    return sat_pressure_0c * np.exp(17.67 * (temperature - 273.15 * units.kelvin)
                                    / (temperature - 29.65 * units.kelvin))
    '''

    es = 6.112 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65))  # hPa
    es = xr.DataArray(es, coords=[temp.time, temp.pfull, temp.lat, temp.lon],
                      dims=['time', 'pfull', 'lat', 'lon'])
    ws = 0.622 * es / (temp.pfull - es)
    qs = ws / (1 + ws)
    qs = xr.DataArray(qs, coords=[temp.time, temp.pfull, temp.lat, temp.lon],
                    dims=['time', 'pfull', 'lat', 'lon'])
    return qs
