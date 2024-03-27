import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapefile as shp
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
              file_names=None, do_zweight=False, do_hweight=True,
              drop_vars=['time_centered_bounds', 'time_counter_bounds'],
              chunks={'time_counter': 'auto', 'lon': 'auto', 'lat': 'auto'}, **kwargs):
    """
    load OIFS data
    In case file_names is not None: vname and years is just used for metadata info, data_freq is meaningless
    """
# 'time_centered_bounds', 'time_counter_bounds' are variables
# time_centered is a coordinate that is not automatically chunked the same way that the data is

    xr.set_options(keep_attrs=True)
    # Default values
    is_data = 'scalar'
    str_aheight, str_atim = '', '' # string for arithmetic
    str_lheight, str_ltim = '', '' # string for labels    
    
    # Open data
    if file_names is None: file_names, str_ltim = get_filenames(vname, data_freq, years)
    else:
        # Do not change file_names but determine str_ltim
        if isinstance(years, (list, np.ndarray, range)):
            # years = [yr_start, yr_end]
            if isinstance(years, list) and len(years)==2:
                str_ltim = 'y:{}-{}'.format(str(years[0]), str(years[1]))                                                                                                                                          
            # years = [year1,year2,year3....]            
            else:
                str_ltim = 'y:{}-{}'.format(str(years[0]), str(years[-1]))
        elif isinstance(years, int):
            str_ltim = 'y:{}'.format(years)
        else:    
            raise ValueError( " year can be integer, list, np.array or range(start,end). Got {}, namely {}".format(type(years), years))
    file_paths = get_filepaths(data_path, file_names)#[data_path + '/' + file_name for file_name in file_names]
    data_set = xr.open_mfdataset(file_paths, parallel=True, chunks=chunks, **kwargs)
    data_set = data_set.drop_vars(drop_vars)
    if 'time_centered' in data_set.coords:
        if chunks['time_counter'] == 'auto':
            # data_set.time_centered.load() # needs to be loaded or deleted to avoid incosistent chunk sizes
            data_set = data_set.drop_vars('time_centered')
        else:
            data_set.time_centered.chunk({'time_counter': chunks['time_counter']})
    
    # Rename time dimension
    data_set = data_set.rename({'time_counter': 'time'})
    
    # add weights
    if (do_zarithm != None) and (do_zarithm != 'None'): raise NotImplementedError('zaveraging is not implemented yet')
    data_set = do_oifs_weights(data_set, do_zweight=do_zweight, do_hweight=do_hweight)

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
                             do_tarithm=do_tarithm, do_zarithm=do_zarithm, do_compute=do_compute, do_load=do_load, do_persist=do_persist,
                             descript=data_name, chunks=chunks, **kwargs)

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



def do_oifs_weights(data_set, do_zweight=False, do_hweight=True):
    if do_hweight:
        set_chunk = dict({'lat': data_set.chunksizes['lat']})#'lon': data_set.chunksizes['lon'], })
        w_cos = xr.DataArray(np.cos(np.deg2rad(data_set.lat)).astype('float32'), dims=['lat']).chunk(set_chunk)
        data_set = data_set.assign_coords(w_cos=w_cos)
        del(w_cos)
    if do_zweight: raise NotImplementedError('zweights is not supported yet')
    return(data_set)



def load_index_reg(data, box_list, boxname=None, do_harithm='wmean',
                   do_zarithm=None, do_outputidx=False,
                   do_compute=False, do_load=True, do_persist=False, do_checkbasin=False):
    xr.set_options(keep_attrs=True)
    index_list = []
    indexin_list = []
    cnt = 0

    #___________________________________________________________________________
    # loop over box_list
    for box in box_list:

        if not isinstance(box, shp.Reader):
            if len(box)==2: boxname, box  = box[1], box[0]
            if box is None or box=='global': boxname='global'
        else:
            boxname = os.path.basename(box.shapeName).replace('_',' ')

        if boxname != 'global': raise NotImplementedError('Only global selection supported as of now')

        #_______________________________________________________________________
        # compute  mask index
        idx_IN = xr.DataArray(do_boxmask_reg(box, data.lon, data.lat), dims=['lon', 'lat']).chunk({'lon': data.chunksizes['lon'], 'lat': data.chunksizes['lat']})

        #_______________________________________________________________________
        # check basin selection

        #_______________________________________________________________________
        # selected points in xarray dataset object and  average over selected points
        dim_name=['lon', 'lat']

        #_______________________________________________________________________
        # do horizontal
        index = do_horizontal_arithmetic_reg(data, do_harithm)

        index_list.append(index)
        indexin_list.append(idx_IN)
        del(index)

        if do_compute: index_list[cnt] = index_list[cnt].compute()
        if do_load   : index_list[cnt] = index_list[cnt].load()
        if do_persist: index_list[cnt] = index_list[cnt].persist()

        #_______________________________________________________________________
        vname = list(index_list[cnt].keys())    
        index_list[cnt][vname[0]].attrs['boxname'] = boxname
        
        #_______________________________________________________________________
        cnt = cnt + 1
        
    #___________________________________________________________________________
    if do_outputidx:
        return(index_list, idxin_list)
    else:
        return(index_list)






def do_boxmask_reg(box, mesh_lon, mesh_lat):

    #___________________________________________________________________________
    # a rectangular box is given --> translate into shapefile object
    if box == None or box == 'global': # if None do global
        idx_IN = np.ones((mesh_lon.shape[0], mesh_lat.shape[0]), dtype=bool)

    else:
        raise NotImplementedError("Only global boxes supported")
    
    return(idx_IN)





def do_horizontal_arithmetic_reg(data, do_harithm):
    if do_harithm == 'wmean':
        weights = data['w_cos']
        data = data.drop_vars('w_cos')
        #weights = weights.where(np.isnan(data)==False) # I don't know what this does
        weights = weights/weights.sum(dim='lat', skipna=True)#/len(data.lon)
        data = data * weights
        del weights
        #data = data.sum(dim=['lat', 'lon'], keep_attrs=True, skipna=True)
        data = data.sum(dim=['lat'], keep_attrs=True, skipna=True).mean(dim='lon', keep_attrs=True, skipna=True)
        data = data.where(data!=0)
    return(data)
