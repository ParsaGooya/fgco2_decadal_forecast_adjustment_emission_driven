import numpy as np
from xarray import DataArray
import xarray as xr
import math

def select_test_years(dataset, n_test_years=1, test_years=None):
    if test_years:
        years = np.sort([test_years]).flatten()
        if years[-1] > dataset.year[-1].item():
            raise ValueError(f"Test year(s) {years} not valid - last possible test year: {dataset.year[-1].item()}")
    else:
        years = dataset.year[- n_test_years:].to_numpy()
    return years


def create_train_mask(dataset, exclude_idx=0):
    mask = np.full((dataset.shape[0], dataset.shape[1]), False, dtype=bool)
    x = np.arange(0, 12*dataset.shape[0], 12)   
    y = np.arange(1, dataset.shape[1] + 1)
    idx_array = x[..., None] + y
    # mask[idx_array >= idx_array[-1, exclude_idx + 12]] = True
    mask[idx_array > idx_array[-1, exclude_idx + 11]] = True
    return mask


def get_coordinate_indices(data, area):
    lat_min, lat_max, lon_min, lon_max = area
    coord_indices = []
    coords = []
    for min_val, max_val, data in [(lat_min, lat_max, data.lat.data), (lon_min, lon_max, data.lon.data)]:
        x = data - min_val
        if np.min(x) > 0:
            min_idx = 0
        else:
            min_idx = np.abs(x[x <= 0]).argmin()
        coord_indices.append(min_idx)
        coords.append(data[min_idx])
        y = data - max_val
        if np.max(y) < 0:
            max_idx = len(data) - 1
        else:
            max_idx = np.argwhere(y == np.min(y[y >= 0])).flatten()[0]
        coord_indices.append(max_idx)
        coords.append(data[max_idx])
    return coord_indices, coords



def align_data_and_targets(data, targets, lead_years):
    if 'ensembles' in data.dims: 
        if lead_years*12 > data.shape[2]:
            raise ValueError(f'Maximum available lead years: {int(data.shape[2] / 12)}')
    else:
        if lead_years*12 > data.shape[1]:
            raise ValueError(f'Maximum available lead years: {int(data.shape[1] / 12)}')
        
    last_target_year = data.year[-1].item() + lead_years - 1
    last_years_diff = data.year[-1].item() - targets.year[-1].item()
    year_diff = targets.year[-1].item() - last_target_year
    if year_diff >= 0:
        if 'ensembles' in data.dims:
            ds = data[:,:,:(lead_years*12),...]
        else:
            ds = data[:,:(lead_years*12),...]

        obs = targets[:len(targets.year) - year_diff,...]

    else: 
        if (last_years_diff <= 0):
            if 'ensembles' in data.dims:
                ds = data[:,:,:(lead_years*12),...]
            else:
                ds = data[:,:(lead_years*12),...]
            obs = targets
        else:
            if 'ensembles' in data.dims:
                ds = data[:-last_years_diff,:,:(lead_years*12),...]
            else:
                ds = data[:-last_years_diff,:(lead_years*12),...]
            obs = targets

    return ds, obs


def reshape_obs_to_data(obs, data, return_xarray=False):

    lead_years = int(data.shape[1] / 12)
    ls = [obs[y :y + lead_years,...].stack(flat = ['year','month']).reset_index('flat', drop = True) for y in range(len(obs.year))]
    obs_reshaped = xr.concat([ds.assign_coords(flat = np.arange(1, len(ds.flat) + 1) ) for ds in ls], dim = 'year').assign_coords(year = obs.year)

    return obs_reshaped.rename({'flat' : 'month'}).transpose('year','month', ...).where((obs_reshaped.year >= data.year.min()) & (obs_reshaped.year <= data.year.max()) , drop = True)


def check_array_integrity(x, x_reshaped, size=50):
    rand_year = np.random.randint(0, x_reshaped.shape[0], size=size)
    rand_month = np.random.randint(0, x_reshaped.shape[1], size=size)
    for year_idx, month_idx in zip(rand_year, rand_month):
        year_overflow, month_overflow = np.divmod(month_idx, 12)
        x_year = year_idx + year_overflow
        x_month = month_overflow
        if not np.array_equal(x[x_year, x_month,...], x_reshaped[year_idx, month_idx]):
            print('ERROR')
    print('No errors found')


def batch_create_spatial_subsets(data, dim0_min, dim0_max, dim1_min, dim1_max):
    subsets = []
    if type(data) != list:
        data = [data]
    for array in data:
        subsets.append(array[..., dim0_min:dim0_max, dim1_min:dim1_max])
    return subsets
    

def compute_anomalies(data, anomalies_base, mask=None, axis=0):
    if mask is not None:
        mean = np.ma.array(anomalies_base.to_numpy(), mask=mask).mean(axis=axis).data
    else:
        mean = anomalies_base.to_numpy().mean(axis=axis)
    data_anomalies = data - mean
    return data_anomalies

def calculate_climatology(ds, mask = None):
    if mask is not None:
        ds = ds.where(~mask)
    return ds.mean('year')



def detrend(data, detrend_base, trend_dim, deg=1, mask=None, with_intercept=True):
    detrend_base[trend_dim] = np.arange(len(detrend_base[trend_dim]))
    if mask is not None:
        trend_coefs = detrend_base.where(~mask).polyfit(dim=trend_dim, deg=deg, skipna=True)
    else:
        trend_coefs = detrend_base.polyfit(dim=trend_dim, deg=deg, skipna=True)
    slope = trend_coefs['polyfit_coefficients'][0].to_numpy()
    intercept = trend_coefs['polyfit_coefficients'][1].to_numpy()
    trend_axis = int(np.where(np.array(data.dims) == trend_dim)[0])
    timesteps = np.expand_dims(np.arange(data.shape[trend_axis]), axis=[i for i in range(0, data.ndim) if i != trend_axis])
    slope = np.expand_dims(slope, axis=trend_axis)
    intercept = np.expand_dims(intercept, axis=trend_axis)
    if with_intercept:
        trend = timesteps * slope + intercept
    else:
        trend = timesteps * slope
    data_detrend = data - trend
    return data_detrend, slope, intercept


def standardize(data, base, mask=None):
    if mask is not None:
        marray = np.ma.array(base.to_numpy(), mask=mask)
        mean = marray.mean()
        std = marray.std()
    else:
        mean = base.to_numpy().mean()
        std = base.to_numpy().std()
    data_standardized = (data - mean) / std
    coeffs = {'mean': mean,
              'std': std}
    return data_standardized, coeffs


def normalize(data, base, mask=None):
    if mask is not None:
        marray = np.ma.array(base.to_numpy(), mask=mask)
        max_val = marray.max()
        min_val = marray.min()
    else:
        max_val = base.to_numpy().max()
        min_val = base.to_numpy().min()
    data_normalized = (data - min_val) / (max_val - min_val)
    coeffs = {'min': min_val,
              'max': max_val}
    return data_normalized, coeffs


def linear_debiasing(data, baseline_data, baseline_targets, mask=None):
    # See Kharin et al. 2012: Statistical adjustment of decadal predictions in a changing climate. Geophysical Research Letters 39
    if mask is not None:
        baseline_data_mean = np.ma.array(baseline_data.to_numpy(), mask=mask).mean(axis=0).data
    else:
        baseline_data_mean = baseline_data.to_numpy().mean(axis=0)
    baseline_target_mean = baseline_targets.to_numpy().mean(axis=0)
    lead_time_years = int(np.ceil(baseline_data_mean.shape[0] / 12))
    baseline_target_mean = np.concatenate([baseline_target_mean] * lead_time_years, axis=0)
    bias = baseline_data_mean - baseline_target_mean
    data_debiased = data - bias
    return data_debiased


class AnomaliesScaler:
    def __init__(self, axis=0) -> None:
        self.mean = None
        self.axis=axis
    
    def fit(self, data, mask=None):
        if mask is not None:
            self.mean = np.ma.array(data.to_numpy(), mask=mask).mean(axis=self.axis).data
        else:
            self.mean = data.to_numpy().mean(axis=self.axis)
        return self

    def transform(self, data):
        data_anomalies = data - self.mean
        return data_anomalies
    
    def inverse_transform(self, data):
        if data.shape[1] > 12 and self.mean.shape[0] <= 12:
            lead_years = int(data.shape[1] / 12)
            mean = np.concatenate([self.mean for _ in range(lead_years)], axis=0)
            data_raw = data + mean
        else:
            data_raw = data + self.mean
        return data_raw
    


 
class AnomaliesScaler_v2:
    def __init__(self, axis=0) -> None:
        self.mean = None
        self.axis=axis
    
    def fit(self, data, mask=None):
        
        if 'ensembles' in data.dims: ## PG: if ensemble exists in the dimentions. Note that we always pass a map like data to this function. Even if it is flattened, we first write back to maps.
            axis = (self.axis, 2) ## PG: Tell the object to average over both years and ensembles for calculating anomalies.
        else:
            axis = self.axis

        if mask is not None:
            
            self.mean = np.ma.array(data.to_numpy(), mask=mask).mean(axis=axis).data[0:12,...] #PG

        else:
            self.mean = data.to_numpy().mean(axis=axis)[0:12,...] #PG
        
        nly = int(data.shape[1]/12) #PG
        self.mean = np.concatenate([self.mean for _ in range(nly)], axis = 0) #PG

        return self

    def transform(self, data):

        shape = data.dims
        data_anomalies = data.copy()
        if 'ensembles' in data.dims:  ## PG: if ensemble exists in the dimentions.

            data_anomalies = data.transpose('ensembles', 'year', ...) ## PG: move ensemble dim to axis = 0 so that we can substract the mean that averaged over both years and ensembles
 

        data_anomalies = data_anomalies - self.mean

        return data_anomalies.transpose(*shape) ## PG: Move ensemble back to the original axis.
    
    def inverse_transform(self, data):
        
        shape = data.shape
        if data.shape[1] > 12 and self.mean.shape[0] <= 12:
            lead_years = int(data.shape[1] / 12)
            mean = np.concatenate([self.mean for _ in range(lead_years)], axis=0)

            try:
                data_raw = data + mean
            except: ## PG: if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                
                data_raw = np.transpose(data, (2,0,1,3,4,5))
                data_raw = data_raw + mean
                data_raw = np.transpose(data_raw, (1,2,0,3,4,5))
        else:
            try:
                data_raw = data + self.mean
            except: ## PG: if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                data_raw = np.transpose(data, (2,0,1,3,4,5))
                data_raw = data_raw + self.mean
                data_raw = np.transpose(data_raw, (1,2,0,3,4,5))
            
        return data_raw ## Move ensemble back to its original position
    
class AnomaliesScaler_v1:
    def __init__(self, axis=0) -> None:
        self.mean = None
        self.axis=axis
    
    def fit(self, data, mask=None):
        
        if 'ensembles' in data.dims: ## PG: if ensemble exists in the dimentions. Note that we always pass a map like data to this function. Even if it is flattened, we first write back to maps.
            self.large_ensemble = True
            axis = (self.axis, 2) ## PG: Tell the object to average over both years and ensembles for calculating anomalies.
        else:
            axis = self.axis

        if mask is not None:
            
            self.mean = np.ma.array(data.to_numpy(), mask=mask).mean(axis=axis).data#PG

        else:
            self.mean = data.to_numpy().mean(axis=axis) #PG
        
        
        return self

    def transform(self, data):

        shape = data.dims
        data_anomalies = data.copy()
        if 'ensembles' in data.dims:  ## PG: if ensemble exists in the dimentions.

            data_anomalies = data.transpose('ensembles', 'year', ...) ## PG: move ensemble dim to axis = 0 so that we can substract the mean that averaged over both years and ensembles
 

        data_anomalies = data_anomalies - self.mean
        
        return data_anomalies.transpose(*shape) ## PG: Move ensemble back to the original axis.
     
    def inverse_transform(self, data):
        
        shape = data.shape
        if data.shape[1] > 12 and self.mean.shape[0] <= 12:
            lead_years = int(data.shape[1] / 12)
            mean = np.concatenate([self.mean for _ in range(lead_years)], axis=0)

            try:
                data_raw = data + mean
            except: ## PG: if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                
                data_raw = np.transpose(data, (2,0,1,3,4,5))
                data_raw = data_raw + mean
                data_raw = np.transpose(data_raw, (1,2,0,3,4,5))
        else:
            try:
                data_raw = data + self.mean
            except: ## PG: if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                data_raw = np.transpose(data, (2,0,1,3,4,5))
                data_raw = data_raw + self.mean
                data_raw = np.transpose(data_raw, (1,2,0,3,4,5))
            
        return data_raw



class Spatialnanremove: ## PG


    def __init__(self):
        pass

    def fit(self, data, target): ## PG: extract common grid points based on trainig and target data


        self.reference_shape = xr.full_like(target[0,0,0,...], fill_value = np.nan)
        try: 
            self.reference_shape = self.reference_shape.drop(['month','year']) 
        except: 
            self.reference_shape = self.reference_shape.drop(['lead_time','year']) ## PG: Extract initial spatial shape of traiing data for later
        temp = target.stack(ref = ['lat','lon']).sel(ref =  data.stack(ref = ['lat','lon']).dropna(dim = 'ref').ref)  ## PG: flatten target in space and choose space points where data is not NaN.
        self.final_locations = temp.dropna('ref').ref ## PG: Extract locations common to target and training data by dropping the remaining NaN values
 
        return self

    def sample(self, data, mode = None, loss_area = None): ## PG: Pass a DataArray and sample at the extracted locations
        
           
        conditions = ['lat' in data.dims, 'lon' in data.dims]

        if all(conditions): ## PG: if a map get passeed
                
                sampled = data.stack(ref = ['lat','lon']).sel(ref = self.final_locations)

        else: ## PG: If a flattened dataset is passed (in space)
                sampled = data.sel(ref = self.final_locations)
    

        if mode == 'Eval':   ## PG: if we are sampling the test_set, remmeber the shepe of the test Dataset in a template
            self.shape = xr.full_like(sampled, fill_value = np.nan)


        return sampled

    def extract_indices(self, loss_area): ## PG: Extract indices of the flattened dimention over a specific region

            lat_min, lat_max, lon_min, lon_max = loss_area
            subregion_indices = self.final_locations.where((self.final_locations.lat < lat_max) &  (self.final_locations.lat > lat_min) )
            subregion_indice = subregion_indices.where((self.final_locations.lon < lon_max) &  (self.final_locations.lon > lon_min) )

            return ~ subregion_indices.isnull().values

    def to_map(self, data): ## PG: Write back the flattened data to maps
        if not isinstance(data, np.ndarray): ## PG: if you pass a numpy array (the output of the network)
            return data.unstack().combine_first(self.reference_shape) ## Unstack the flattened spatial dim and write back to the initial format as saved in self.reference_shape using NaN as fill value
        
        else:  ## PG: if you pass a numpy array (the output of the network), we use the test_set template that we saved to create a Datset.

            output = self.shape
            if output.shape[2] < data.shape[2]:
                output = xr.concat([output.isel(ensembles = 2).expand_dims('ensembles',axis = 2) for _ in range(data.shape[2])], dim = 'ensembles').assign_coords(ensembles = np.arange(1, data.shape[2] + 1))
            output[:] =  data[:]
            return output.unstack().combine_first(self.reference_shape)

       




class Detrender:
    def __init__(self, trend_dim='year', deg=1, remove_intercept=True, version = None) -> None:
        self.trend_dim = trend_dim
        self.deg = deg
        self.slope = None
        self.intercept = None
        self.trend_axis = None
        self.remove_intercept = remove_intercept

        self.version = version
        if version == 'None':
                self.lead_time_dim = False
        else:
                self.lead_time_dim = True

    def fit(self, data, mask=None):

        if self.version == 2:
            data = data[:,:12,...]
            mask = mask[:,:12,...]


        data[self.trend_dim] = np.arange(len(data[self.trend_dim]))
        if mask is not None:
            trend_coefs = data.where(~mask).polyfit(dim=self.trend_dim, deg=self.deg, skipna=True)
        else:
            trend_coefs = data.polyfit(dim=self.trend_dim, deg=self.deg, skipna=True)


        slope = trend_coefs['polyfit_coefficients'][0].to_numpy()
        intercept = trend_coefs['polyfit_coefficients'][1].to_numpy()

        self.trend_axis = int(np.where(np.array(data.dims) == self.trend_dim)[0])
        self.slope = np.expand_dims(slope, axis=self.trend_axis)
        self.intercept = np.expand_dims(intercept, axis=self.trend_axis)
        return self

    def transform(self, data, start_timestep=0, remove_intercept=None):
        if remove_intercept is None:
            remove_intercept = self.remove_intercept        
        timesteps = self._make_timesteps(data.shape[self.trend_axis], data.ndim, start_timestep=start_timestep)
        if self.lead_time_dim :
            if data.shape[1] > 12 and self.slope.shape[1] <= 12:
                lead_years = int(data.shape[1] / 12)
                trend = np.concatenate([self._compute_trend(timesteps + i, with_intercept=remove_intercept) for i in range(lead_years)], axis=1)
            else:
                trend = self._compute_trend(timesteps, with_intercept=remove_intercept)        
        else:
                trend = self._compute_trend(timesteps, with_intercept=remove_intercept)
        data_detrended = data - trend
        return data_detrended

    def inverse_transform(self, data, start_timestep=0, add_intercept=None):
        timesteps = self._make_timesteps(data.shape[self.trend_axis], data.ndim, start_timestep=start_timestep)
        if add_intercept is None:
            add_intercept = self.remove_intercept
        if self.lead_time_dim:
            if data.shape[1] > 12 and self.slope.shape[1] <= 12:
                lead_years = int(data.shape[1] / 12)
                trend = np.concatenate([self._compute_trend(timesteps + i, with_intercept=add_intercept) for i in range(lead_years)], axis=1)
            else:
                trend = self._compute_trend(timesteps, with_intercept=add_intercept)
        else:
                trend = self._compute_trend(timesteps, with_intercept=add_intercept)
        data_trended = data + trend
        return data_trended

    def get_trend(self, sequence_length, start_timestep=0, with_intercept=True):
        timesteps = self._make_timesteps(sequence_length, self.slope.ndim, start_timestep=start_timestep)
        trend = self._compute_trend(timesteps, with_intercept=with_intercept)
        return trend
    
    def get_trend_coeffs(self):
        return self.slope, self.intercept

    def _make_timesteps(self, sequence_length, ndims, start_timestep=0):
        timesteps = np.expand_dims(np.arange(sequence_length) + start_timestep, axis=[i for i in range(ndims) if i != self.trend_axis])
        return timesteps

    def _compute_trend(self, timesteps, with_intercept=True):
        if with_intercept:
            trend = timesteps * self.slope + self.intercept
        else:
            trend = timesteps * self.slope
        return trend

class Standardizer:

    def __init__(self, axis = None) -> None:
        self.mean = None
        self.std = None
        self.axis = axis

    def fit(self, data, mask=None):

        if mask is not None:
            marray = np.ma.array(data, mask=mask)
        else:
            marray = data.to_numpy()
        
        if self.axis is None:

            if np.isnan(marray.mean()):
                self.mean = np.ma.masked_invalid(marray).mean()
                self.std = np.ma.masked_invalid(marray).std()
            else:            
                self.mean = marray.mean()
                self.std = marray.std()
        else:

                self.mean = marray.mean(self.axis).data
                self.std = marray.std(self.axis).data + 1e-4

        return self

    def transform(self, data):
        data_standardized = (data - self.mean) / self.std
        if self.axis is None:
            data_standardized = data_standardized.where(~np.isnan(self.mean)  & (self.std != 0) , 0).where(~np.isnan(self.mean)) 
        return data_standardized

    def inverse_transform(self, data):
        data_raw = data * self.std + self.mean
        return data_raw


class Normalizer:

    def __init__(self) -> None:
        self.min = None
        self.max = None
    
    def fit(self, data, mask=None):
        if mask is not None:
            marray = np.ma.array(data, mask=mask)
            self.min = marray.min()
            self.max = marray.max()
        else:
            self.min = data.min()
            self.max = data.max()
        return self

    def transform(self, data):
        data_normalized = (data - self.min) / (self.max - self.min)
        return data_normalized

    def inverse_transform(self, data):
        data_raw = data * (self.max - self.min) + self.min
        return data_raw


class PreprocessingPipeline:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.steps = []
        self.fitted_preprocessors = []

    def fit(self, data, mask=None):
        data_processed = data
        for step_name, preprocessor in self.pipeline:
            preprocessor.fit(data_processed, mask=mask)
            data_processed = preprocessor.transform(data_processed)
            self.steps.append(step_name)
            self.fitted_preprocessors.append(preprocessor)
        return self

    def transform(self, data, step_arguments=None):
        if step_arguments is None:
            step_arguments = dict()
        for a in step_arguments.keys():
            if a not in self.steps:
                raise ValueError(f"{a} not in preprocessing steps!")
            
        data_processed = data
        for step, preprocessor in zip(self.steps, self.fitted_preprocessors):
            if step in step_arguments.keys():
                args = step_arguments[step]
            else:
                args = dict()
            data_processed = preprocessor.transform(data_processed, **args)
        return data_processed

    def inverse_transform(self, data, step_arguments=None):
        if step_arguments is None:
            step_arguments = dict()
        for a in step_arguments.keys():
            if a not in self.steps:
                raise ValueError(f"{a} not in preprocessing steps!")
            
        data_processed = data
        for step, preprocessor in zip(reversed(self.steps), reversed(self.fitted_preprocessors)):
            if step in step_arguments.keys():
                args = step_arguments[step]
            else:
                args = dict()
            data_processed = preprocessor.inverse_transform(data_processed, **args)
        return data_processed

    def get_preprocessors(self, name=None):
        if name is None:
            return self.fitted_preprocessors
        else:
            idx = np.argwhere(np.array(self.steps) == name).flatten()
            if idx.size == 0:
                raise ValueError(f"{name} not in preprocessing steps!")
            return self.fitted_preprocessors[int(idx)]
    
    def add_fitted_preprocessor(self, preprocessor, name, index=None):
        if index is None:
            self.fitted_preprocessors.append(preprocessor)
            self.steps.append(name)
        else:
            self.fitted_preprocessors.insert(index, preprocessor)
            self.steps.insert(index, name)



from itertools import product
import random
class config_grid:
    def __init__(self, hyperparameter_dict = None):
            self.hyperparameter_dict = hyperparameter_dict 
    def full_grid(self):
        if self.hyperparameter_dict is None:
            raise ValueError(f"Provide a range of possibilities for at least one parameter!")
        output = []
        ranges = list(self.hyperparameter_dict.values())
        combinations = list(product(*ranges))
        for combo in combinations:
            output.append(dict(zip(self.hyperparameter_dict.keys(), combo)))
        return output 
    def draw_random(self, num_samples, seed = None ):
        population = self.full_grid()
        if seed is not None:
            random.seed(seed)
        return random.sample(population, num_samples)