import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dask.diagnostics import ProgressBar

from .sub_data     import *

def get_filenames(vname, data_freq, years, prefix='atm_remapped', sep='_'):
    '''generate names of all files to be opened for a given variable (accumulate years)'''
    file_names = []
    
    
    if isinstance(years, (list, np.ndarray, range)):
        # years = [yr_start, yr_end]
        if isinstance(years, list) and len(years)==2: 
            year_in = range(years[0],years[1]+1)
            str_mtim = 'y:{}-{}'.format(str(years[0]), str(years[1]))
        # years = [year1,year2,year3....]            
        else:           
            year_in = years
            str_mtim = 'y:{}-{}'.format(str(years[0]), str(years[-1]))
        # loop over years to create filename list 
        for year in year_in:
            file_names.append(prefix + sep + data_freq + sep + vname + sep + data_freq + sep + str(year) + '-' + str(year) + '.nc')
    elif isinstance(years, int):
        year = years
        str_mtim = 'y:{}'.format(year)
        file_names.append(prefix + sep + data_freq + sep + vname + sep + data_freq + sep + str(year) + '-' + str(year) + '.nc')
    else:
        raise ValueError( " year can be integer, list, np.array or range(start,end). Got {}, namely {}".format(type(years), years))
    return file_names, str_mtim





def get_filepaths(data_path, file_names):
    assert os.path.isdir(data_path), "{} is not a directory".format(data_path)
    file_paths = [data_path + '/' + file_name for file_name in file_names]
    for file_path, file_name in zip(file_paths, file_names): assert os.path.isfile(file_path), "{} is not a file name in {}".format(file_name, data_path)
    return file_paths





def open_data(data_path, vname, data_freq, years, mon=None, day=None, record=None, height=None, heightidx=None,
              do_tarithm='mean', do_zarithm='mean', descript='', do_compute=False, do_load=True, do_persist=False,
              chunks={'time': 'auto', 'lon': 'auto', 'lat': 'auto'}, **kwargs):
    xr.set_options(keep_attrs=True)
    # Default values
    is_data = 'scalar'
    str_aheight, str_atim = '', '' # string for arithmetic
    str_lheight, str_ltim = '', '' # string for labels    
    
    # Open data
    file_names, str_ltim = get_filenames(vname, data_freq, years)
    file_paths = get_filepaths(data_path, file_names)#[data_path + '/' + file_name for file_name in file_names]
    data_set = xr.open_mfdataset(file_paths, parallel=True, chunks=chunks, **kwargs)
    
    # Rename time dimension
    data_set = data_set.rename({'time_counter': 'time'})
    
    # years are selected by the files that are open, need to select mon or day or record 
    data_set, mon, day, str_ltim = do_select_time(data_set, mon, day, record, str_ltim)
    
    # do time arithmetic on data
    if 'time' in data_set.dims:
        data_set, str_atim = do_time_arithmetic(data_set, do_tarithm)
    
    # Select height
    if height is not None: raise NotImplementedError('selecting height is not implemented')
    if heightidx is not None: raise NotImplementedError('selecting heightidx is not implemented')
        # return str_lheight, str_aheight
    
    # write additional attribute info
    str_lsave = str_ltim + str_lheight
    str_lsave = str_lsave.replace(' ','_').replace(',','').replace(':','')
    attr_dict = dict({'datapath':data_path, 'do_tarithm':str_atim, 'do_zarithm':str_aheight, 'descript':descript,
                     'year':years, 'mon':mon, 'day':day, 'record':record, 'height':height, 'heightidx':heightidx,
                     'str_ltim':str_ltim,'str_lheight':str_lheight,'str_lsave':str_lsave,
                     'is_data':is_data, 'do_compute':do_compute})
    data_set = do_additional_attrs(data_set, vname, attr_dict)

    # Return Data
    if do_compute:
        with ProgressBar(): data_set = data_set.compute()
    if do_load   :
        with ProgressBar(): data_set = data_set.load()
    if do_persist:
        with ProgressBar(): data_set = data_set.persist()
    return data_set




def open_multiple_data(data_paths, data_names, vname, data_freq, years, mon=None, day=None, record=None,
                       height=None, heightidx=None, do_tarithm='mean', do_zarithm='mean',
                       do_compute=False, do_load=True, do_persist=False, ref_path=None, do_reffig=False,
                       ref_year=None, ref_mon=None, ref_day=None, ref_record=None,
                       chunks={'time_counter': 'auto', 'lon': 'auto', 'lat': 'auto'}, **kwargs):
    '''for every path / experiment it opens a dataset with the variable'''
    assert len(data_paths) == len(data_names), "data_paths and data_names do not have the same length"
    data_sets = []
    for ii, (data_path, data_name) in enumerate(zip(data_paths, data_names)):
        yearsi, moni, dayi, recordi = years, mon, day, record
        if (ii==0) and (ref_path != None and ref_path != 'None'): yearsi, moni, dayi, recordi = ref_year, ref_mon, ref_day, ref_record
        data_set = open_data(data_path, vname, data_freq, yearsi, mon=moni, day=dayi, record=recordi, height=height, heightidx=heightidx,
                             do_tarithm=do_tarithm, do_zarithm=do_zarithm, descript=data_name, chunks=chunks, **kwargs)

        # create reference data if given 
        if (ii==0) and (ref_path != None and ref_path != 'None'):
            data_set_ref = data_set
            if do_reffig: data_sets.append(data_set_ref) 
            continue
            
        #__________________________________________________________________________________________________    
        # compute anomaly 
        if (ref_path != None and ref_path != 'None'):
            data_sets.append(do_anomaly(data_set, data_set_ref))  
        # compute absolute    
        else:
            data_sets.append(data_set)
        del(data_set)
    if (ref_path != None and ref_path != 'None'): del(data_set_ref)
    return data_sets   
