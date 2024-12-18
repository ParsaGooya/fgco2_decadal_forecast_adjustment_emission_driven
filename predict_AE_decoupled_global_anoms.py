import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import dask
import xarray as xr
from pathlib import Path
import glob
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models.autoencoder import Autoencoder, Autoencoder_mean
from models.unet import UNet
from models.cnn import CNN, SCNN, CNN_mean
from losses import WeightedMSE, WeightedMSESignLoss
from data_utils.datahandling import combine_observations
from preprocessing import align_data_and_targets, get_coordinate_indices, create_train_mask, reshape_obs_to_data
from preprocessing import AnomaliesScaler_v1, AnomaliesScaler_v2, Detrender, Standardizer, Normalizer, PreprocessingPipeline, Spatialnanremove
from torch_datasets import XArrayDataset
from subregions import subregions
from data_locations import LOC_FORECASTS_fgco2, LOC_OBSERVATIONS_fgco2

# specify data directories
data_dir_forecast = LOC_FORECASTS_fgco2
data_dir_obs = LOC_OBSERVATIONS_fgco2
unit_change = 60*60*24*365 * 1000 /12 * -1 ## Change units for ESM data to mol m-2 yr-1


def predict(fct:xr.DataArray , observation:xr.DataArray , params, lead_years, model_dir,  test_years, model_year = None, ensemble_list = None, ensemble_mode = 'Mean', btstrp_it = 200, save=True):
    if 'extra_predictors_mean' not in params.keys():
        params['extra_predictors_mean'] = []

    if model_year is None:
        model_year_ = np.min(test_years) - 1
    else:
        model_year_ = model_year

    if params["model_mean"] is not None:

        if params['version_mean'] == 1:

            forecast_preprocessing_steps_mean = [
            ('anomalies', AnomaliesScaler_v1(axis=0)), 
            ('standardize', Standardizer()) ]

            observations_preprocessing_steps_mean = [
            ('anomalies', AnomaliesScaler_v1(axis=0))  ]

        elif params['version_mean'] == 2:
                    
                    forecast_preprocessing_steps_mean = [
                    ('anomalies', AnomaliesScaler_v2(axis=0)), 
                    ('standardize', Standardizer()) ]

                    observations_preprocessing_steps_mean = [
                    ('anomalies', AnomaliesScaler_v2(axis=0))  ]
        else:

            forecast_preprocessing_steps_mean = [
            ('anomalies', AnomaliesScaler_v1(axis=0)), 
            ('standardize', Standardizer()) ]

            observations_preprocessing_steps_mean = [
            ('anomalies', AnomaliesScaler_v2(axis=0))  ]

    if ensemble_mode != 'Mean':
        std_axis = (0,1,2)
    else:
        std_axis = (0,1)

    if params['version_patterns'] == 3:

        forecast_preprocessing_steps_patterns =  [('anomalies', AnomaliesScaler_v1(axis=0)),  ('standardize', Standardizer()) ]
        observations_preprocessing_steps_patterns = [ ('anomalies', AnomaliesScaler_v2(axis=0))]

    elif params['version_patterns'] == 1:

        forecast_preprocessing_steps_patterns =  [('anomalies', AnomaliesScaler_v1(axis=0)),  ('standardize', Standardizer()) ]
        observations_preprocessing_steps_patterns = [('anomalies', AnomaliesScaler_v1(axis=0)) ]

    elif params['version_patterns'] == 2:

        forecast_preprocessing_steps_patterns =  [('anomalies', AnomaliesScaler_v2(axis=0)),  ('standardize', Standardizer()) ]
        observations_preprocessing_steps_patterns = [('anomalies', AnomaliesScaler_v2(axis=0)) ]

    else:
        
        forecast_preprocessing_steps_patterns =  [ ('standardize', Standardizer()) ]
        observations_preprocessing_steps_patterns = []


    print(f"Start run for test year {test_years}...")

    ############################################## load data ##################################

    time_features_mean = params["time_features_mean"]

    ensemble_features = params['ensemble_features']
    time_features_patterns = params["time_features_patterns"]
    model_patterns = params['model_patterns']
    hidden_dims = params['hidden_dims']

    try:
        batch_normalization = params["batch_normalization"]
        dropout_rate = params["dropout_rate"]
    except:
        interp_nan = params['interp_nan']
        obs_clim = params["obs_clim"]
        kernel_size = params["kernel_size"]
        decoder_kernel_size = params["decoder_kernel_size"]

    print("Load forecasts")
    if ensemble_list is not None: ## PG: calculate the mean if ensemble mean is none
        ds_in = fct.sel(ensembles = ensemble_list)['fgco2']
            
    else:    ## Load specified members
        ds_in = fct['fgco2'] 

    if ensemble_mode == 'Mean': ##
        ensemble_features = False
        ds_in = ds_in.mean('ensembles').load() ##
    else:
        ds_in = fct.load().transpose('year','lead_time','ensembles',...)
        print(f'Warning: ensemble_mode is {ensemble_mode}. Training for large ensemble ...')


    obs_in = observation.expand_dims('channels', axis=2)

    if 'ensembles' in ds_in.dims: ### PG: add channels dimention to the correct axis based on whether we have ensembles or not
        ds_in = ds_in.expand_dims('channels', axis=3).sortby('ensembles')
    else:
        ds_in = ds_in.expand_dims('channels', axis=2) 

    ds_in_ = ds_in.sel(year = test_years).isel(lead_time = np.arange(0,12 * lead_years ))

    ds_raw, obs_raw = align_data_and_targets(ds_in.where(ds_in.year <= model_year_, drop = True), obs_in, lead_years)  # extract valid lead times and usable years ## used to be np.min(test_years)

    if 'ensembles' in ds_raw.dims: ## PG: reorder dimensions in you have ensembles
        ds_raw_ensemble_mean = ds_raw.transpose('year','lead_time','ensembles',...)* unit_change
        ds_in_ = ds_in_.transpose('year','lead_time','ensembles',...)* unit_change
    else:
        ds_raw_ensemble_mean = ds_raw.transpose('year','lead_time',...)* unit_change
        ds_in_ = ds_in_.transpose('year','lead_time',...)* unit_change
    
    obs_raw = reshape_obs_to_data(obs_raw, ds_raw_ensemble_mean, return_xarray=True)

    if not ds_raw_ensemble_mean.year.equals(obs_raw.year):         
            ds_raw_ensemble_mean = ds_raw_ensemble_mean.sel(year = obs_raw.year)

    train_years = ds_raw_ensemble_mean.year[ds_raw_ensemble_mean.year <= model_year_].to_numpy()
    ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean,ds_in_ ], dim = 'year')

    #######################################################################################################################################
    nanremover = Spatialnanremove()## PG: Get an instance of the class
    nanremover.fit(ds_raw_ensemble_mean[:,:12,...], obs_raw[:,:12,...]) ## PG:extract the commong grid points between training and obs data
    ds_raw_ensemble_mean = nanremover.to_map(nanremover.sample(ds_raw_ensemble_mean)) ## PG: flatten and sample training data at those locations
    obs_raw = nanremover.to_map(nanremover.sample(obs_raw)) ## PG: flatten and sample obs data at those locations    
    #######################################################################################################################################

    weights = np.cos(np.ones_like(ds_raw_ensemble_mean.lon) * (np.deg2rad(ds_raw_ensemble_mean.lat.to_numpy()))[..., None])  # Moved this up
    weights = xr.DataArray(weights, dims = ds_raw_ensemble_mean.dims[-2:], name = 'weights').assign_coords({'lat': ds_raw_ensemble_mean.lat, 'lon' : ds_raw_ensemble_mean.lon}) # Create an DataArray to pass to Spatialnanremove() 
    weights = nanremover.to_map(nanremover.sample(weights))
    ds_raw_ensemble_mean_patterns = ds_raw_ensemble_mean - ((ds_raw_ensemble_mean * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
    obs_raw_patterns = obs_raw - ((obs_raw * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
    
    ######################################
    if params["model_mean"] is not None:
        obs_raw = ((obs_raw * weights).sum(['lat','lon'])/weights.sum(['lat','lon'])).expand_dims('ref', axis=-1)
    #####################
        if 'atm_co2' in params['extra_predictors_mean']:
            atm_co2 = xr.open_dataset('/home/rpg002/CMIP6_ssp245_xCO2atm_1982_2029.nc').ssp245
            atm_co2 = reshape_obs_to_data(atm_co2, ds_raw_ensemble_mean, return_xarray=True).rename({'month' : 'lead_time'})
            try:
                extra_predictors = atm_co2.sel(year = ds_raw_ensemble_mean.year).expand_dims('channels', axis=2)
            except:
                raise ValueError("Extra predictors not available at the same years as the predictors.")
        else:
            extra_predictors = None     
        
        
        if extra_predictors is not None:
            if params["model_mean"] in [Autoencoder_mean]:
                weights = np.cos(np.ones_like(extra_predictors.lon) * (np.deg2rad(extra_predictors.lat.to_numpy()))[..., None])  # Moved this up
                weights = xr.DataArray(weights, dims = extra_predictors.dims[-2:], name = 'weights').assign_coords({'lat': extra_predictors.lat, 'lon' : extra_predictors.lon}) 
                extra_predictors = (extra_predictors * weights).sum(['lat','lon'])/weights.sum(['lat','lon'])
            else:  
                if not all(['ensembles' not in extra_predictors.dims, 'ensembles' in ds_raw_ensemble_mean.dims]):              
                    ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, extra_predictors], dim = 'channels')
                else:
                    ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, extra_predictors.expand_dims(ensembles = ds_raw_ensemble_mean['ensembles'], axis = 2) ], dim = 'channels')
                extra_predictors = None

    subset_dimensions = params["subset_dimensions"]

    try:
        if params['obs_clim']:
                
                ls = []
                for yr in ds_raw_ensemble_mean.year.values[3:]:
                        
                        ref_base = obs_raw.where(obs_raw.year<yr, drop = True)
                        mask = create_train_mask(ref_base)
                        mask = np.broadcast_to(mask[...,None,None,None], ref_base.shape)
                        ls.append(calculate_climatology(ref_base,mask ).expand_dims('year', axis = 0).assign_coords(year = yr))
                clim = xr.concat(ls, dim = 'year')
                obs_raw = obs_raw[3:]
                ds_raw_ensemble_mean = ds_raw_ensemble_mean[3:]
                ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, clim], dim = 'channels')
                train_years = train_years[3:]
    except:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    
    n_train = len(train_years)
    train_mask = create_train_mask(ds_raw_ensemble_mean[:n_train,...])
    if params["model_mean"] is not None:
        ds_baseline_mean = ds_raw_ensemble_mean[:n_train,...]
        obs_baseline_mean = obs_raw[:n_train,...]

    ds_baseline_patterns = ds_raw_ensemble_mean_patterns[:n_train,...]
    obs_baseline_patterns = obs_raw_patterns[:n_train,...]

    train_mask = create_train_mask(ds_baseline_patterns)
    
    if 'ensembles' in ds_raw_ensemble_mean.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline_patterns.shape)
    else:
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline_patterns.shape)
    if params["model_mean"] is not None:
        preprocessing_mask_obs_mean = np.broadcast_to(train_mask[...,None,None], obs_baseline_mean.shape)
    preprocessing_mask_obs_patterns = np.broadcast_to(train_mask[...,None,None,None], obs_baseline_patterns.shape)



    # Data preprocessing
    if params["model_mean"] is not None:
        ds_pipeline_mean = PreprocessingPipeline(forecast_preprocessing_steps_mean).fit(ds_baseline_mean, mask=preprocessing_mask_fct)
        ds_mean  = ds_pipeline_mean.transform(ds_raw_ensemble_mean)

        obs_pipeline_mean  = PreprocessingPipeline(observations_preprocessing_steps_mean ).fit(obs_baseline_mean , mask=preprocessing_mask_obs_mean)
        # if 'standardize' in ds_pipeline_mean.steps:
        #     obs_pipeline_mean .add_fitted_preprocessor(ds_pipeline_mean.get_preprocessors('standardize'), 'standardize')
        obs_mean  = obs_pipeline_mean.transform(obs_raw)

    ds_pipeline_patterns = PreprocessingPipeline(forecast_preprocessing_steps_patterns).fit(ds_baseline_patterns, mask=preprocessing_mask_fct)
    ds_patterns = ds_pipeline_patterns.transform(ds_raw_ensemble_mean_patterns)

    obs_pipeline_patterns = PreprocessingPipeline(observations_preprocessing_steps_patterns).fit(obs_baseline_patterns, mask=preprocessing_mask_obs_patterns)
    if 'standardize' in ds_pipeline_patterns.steps:
        obs_pipeline_patterns.add_fitted_preprocessor(ds_pipeline_patterns.get_preprocessors('standardize'), 'standardize')
    obs_patterns = obs_pipeline_patterns.transform(obs_raw_patterns)

    year_max = ds_patterns[:n_train + 1].year[-1].values 
    

    # TRAIN MODEL

    lead_time = None
    if params["model_mean"] is not None:
        ds_train_mean = ds_mean[:n_train,...]
        obs_train_mean = obs_mean[:n_train,...]
        ds_test_mean = ds_mean[n_train:,...]

    ds_train_patterns = ds_patterns[:n_train,...]
    obs_train_patterns = obs_patterns[:n_train,...]
    ds_test_patterns = ds_patterns[n_train:,...]

        
    weights = np.cos(np.ones_like(obs_train_patterns.lon) * (np.deg2rad(obs_train_patterns.lat.to_numpy()))[..., None])  # Moved this up
    weights = xr.DataArray(weights, dims = obs_train_patterns.dims[-2:], name = 'weights').assign_coords({'lat': obs_train_patterns.lat, 'lon' : obs_train_patterns.lon}) # Create an DataArray to pass to Spatialnanremove() 
    #########################
    if params["model_mean"] is not None:
        if params["model_mean"] in [Autoencoder_mean]:
            ds_train_mean = nanremover.sample(ds_train_mean) ## PG: flatten and sample training data at those locations
        else: 
            ds_train_mean = ds_train_mean.fillna(0.0)
     ## PG: flatten and sample weighs at those locations
    
    ########################################################################

    if model_patterns in [UNet , CNN]: ## PG: If the model starts with a nn.Conv2d write back the flattened data to maps.
        if interp_nan is not None:
            ds_train_patterns = ds_train_patterns.interpolate_na(dim =interp_nan, method = 'linear' ) ## PG: fill NaN values with 0.0 for training
        else:
            ds_train_patterns = ds_train_patterns.fillna(0.0) ## PG: fill NaN values with 0.0 for training

        img_dim_patterns = ds_train_patterns.shape[-2] * ds_train_patterns.shape[-1] 
 
    else: ## PG: If you have a dense first layer keep the data flattened.

        ds_train_patterns = nanremover.sample(ds_train_patterns)
        weights = nanremover.sample(weights) 
        img_dim_patterns = ds_train_patterns.shape[-1] ## PG: The input dim is now the length of the flattened dimention.
    
    weights = weights.values
    if params["model_mean"] in [Autoencoder_mean]:
        img_dim_mean = ds_train_mean.shape[-1]
    
    if time_features_patterns is None:
        if ensemble_features: ## PG: We can choose to add an ensemble feature.
            add_feature_dim_patterns = 1
        else:
            add_feature_dim_patterns = 0
    else:
        if ensemble_features:
            add_feature_dim_patterns = len(time_features_patterns) + 1
        else:
            add_feature_dim_patterns = len(time_features_patterns)
    
    if params["model_mean"] is not None:
        if time_features_mean is None:
            if ensemble_features: ## PG: We can choose to add an ensemble feature.
                add_feature_dim_mean = 1
            else:
                add_feature_dim_mean = 0
        else:
            if ensemble_features:
                add_feature_dim_mean = len(time_features_mean) + 1
            else:
                add_feature_dim_mean = len(time_features_mean) 

        if extra_predictors is not None:
                add_feature_dim_mean = add_feature_dim_mean + len(params['extra_predictors'])
    ########################################### load the model ######################################
    try:
        if params['obs_clim']:
            n_channels_x = len(ds_train_patterns.channels) + 1
        else:
            n_channels_x = len(ds_train_patterns.channels)
    except:
        pass

    if model_patterns == Autoencoder:
        net_patterns = model_patterns(img_dim_patterns, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim_patterns, append_mode=params['append_mode_patterns'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, device = device)
    elif model_patterns == UNet:
        net_patterns = model_patterns(n_channels_x= n_channels_x+ add_feature_dim_patterns , bilinear = params['bilinear'])
    elif model_patterns == CNN: ## PG: Combining CNN encoder with dense decoder
        net_patterns = model_patterns(n_channels_x + add_feature_dim_patterns ,hidden_dims, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size )
    
    if params["model_mean"] is not None:
        if params["model_mean"] in [Autoencoder_mean]:
            net_mean = Autoencoder_mean(img_dim_mean, [360, 180, 90, 45], added_features_dim=add_feature_dim_mean, append_mode=params['append_mode_mean'], batch_normalization=batch_normalization, dropout_rate=dropout_rate)
        else:
            net_mean = params["model_mean"]( n_channels_x=len(ds_train_mean.channels) + add_feature_dim_mean  , dense_dims= [512,128,32] )


    ##################################################################################################################################
    
    ds_test_patterns = nanremover.sample(ds_test_patterns, mode = 'Eval')  ## PG: Sample the test data at the common locations
    
    if params["model_mean"] is not None:
        ds_test_mean = nanremover.sample(ds_test_mean, mode = 'Eval')
        if params["model_mean"] in [CNN_mean]:
            ds_test_mean = nanremover.to_map(ds_test_mean).fillna(0.0)

    if model_patterns in [UNet, CNN]:
        if interp_nan is not None:
                ds_test_patterns = nanremover.to_map(ds_test_patterns).interpolate_na(dim =interp_nan, method = 'linear' )  ## PG: Write back to map if the model starts with a nn.Conv2D
        else:
                ds_test_patterns = nanremover.to_map(ds_test_patterns).fillna(0.0)  ## PG: Write back to map if the model starts with a nn.Conv2D
    ##################################################################################################################################

    test_lead_time_list = np.arange(1, ds_test_patterns.shape[1] + 1)
    test_years_list = np.arange(1, ds_test_patterns.shape[0] + 1)  ## PG: Extract the number of years as well 
    
    test_set_patterns = XArrayDataset(ds_test_patterns, xr.ones_like(ds_test_patterns).rename({'lead_time':'month'}), lead_time=None, time_features=time_features_patterns,ensemble_features =ensemble_features,  in_memory=False, aligned = True, year_max = year_max,  model = model_patterns.__name__)
    test_results_patterns = np.zeros_like(ds_test_patterns)

    if params["model_mean"] is not None:
        test_set_mean = XArrayDataset(ds_test_mean, xr.ones_like(ds_test_mean).rename({'lead_time':'month'}), extra_predictors = extra_predictors, lead_time=None, time_features=time_features_mean,ensemble_features =ensemble_features,  in_memory=False, aligned = True, year_max = year_max, model = params['model_mean'].__name__ )
        test_results_mean = np.zeros_like(ds_test_mean.mean('ref').expand_dims('ref', axis = -1))


    print('Loading model patterns ....')
    net_patterns.load_state_dict(torch.load(glob.glob(model_dir + f'/*patterns*-{model_year_}*.pth')[0], map_location=torch.device('cpu'))) 
    net_patterns.to(device)
    net_patterns.eval()
    
    for i, (x, target) in enumerate(test_set_patterns): 
        if 'ensembles' in ds_test_patterns.dims:  ## PG: If we have large ensembles:

            ensemble_idx, j = np.divmod(i, len(test_lead_time_list) * len(test_years_list))  ## PG: find out ensemble index
            year_idx, lead_time_list_idx = np.divmod(j, len(test_lead_time_list)) 
            lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
            
            with torch.no_grad():
                if (type(x) == list) or (type(x) == tuple):
                    test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                else:
                    test_raw = x.unsqueeze(0).to(device)
                test_adjusted_patterns = net_patterns(test_raw)
                test_results_patterns[year_idx, lead_time_idx,ensemble_idx,] = test_adjusted_patterns.to(torch.device('cpu')).numpy()  ## PG: write back to test_results

        else:

            year_idx, lead_time_list_idx = np.divmod(i, len(test_lead_time_list))
            lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
            with torch.no_grad():
                if (type(x) == list) or (type(x) == tuple):
                    test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                else:
                    test_raw = x.unsqueeze(0).to(device)
                test_adjusted_patterns = net_patterns(test_raw)
                test_results_patterns[year_idx, lead_time_idx,] = test_adjusted_patterns.to(torch.device('cpu')).numpy()
    
    if params["model_mean"] is not None:

        print('Loading model mean ....')
        net_mean.load_state_dict(torch.load(glob.glob(model_dir + f'/*mean*-{model_year_}*.pth')[0], map_location=torch.device('cpu'))) 
        net_mean.to(device)
        net_mean.eval()
        
        for i, (x, target) in enumerate(test_set_mean): 
            if 'ensembles' in ds_test_mean.dims:  ## PG: If we have large ensembles:

                ensemble_idx, j = np.divmod(i, len(test_lead_time_list) * len(test_years_list))  ## PG: find out ensemble index
                year_idx, lead_time_list_idx = np.divmod(j, len(test_lead_time_list)) 
                lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
                
                with torch.no_grad():
                    if (type(x) == list) or (type(x) == tuple):
                        test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                    else:
                        test_raw = x.unsqueeze(0).to(device)
                    test_adjusted_mean = net_mean(test_raw)
                    test_results_mean[year_idx, lead_time_idx,ensemble_idx,] = test_adjusted_mean.to(torch.device('cpu')).numpy()  ## PG: write back to test_results

            else:

                year_idx, lead_time_list_idx = np.divmod(i, len(test_lead_time_list))
                lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
                with torch.no_grad():
                    if (type(x) == list) or (type(x) == tuple):
                        test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                    else:
                        test_raw = x.unsqueeze(0).to(device)
                    test_adjusted_mean = net_mean(test_raw)
                    test_results_mean[year_idx, lead_time_idx,] = test_adjusted_mean.to(torch.device('cpu')).numpy()

    ##################################################################################################################################

    if model_patterns in [UNet , CNN]:   ## PG: if the output is already a map
                    test_results_untransformed_patterns = obs_pipeline_patterns.inverse_transform(test_results_patterns)
                    result = xr.DataArray(test_results_untransformed_patterns, ds_test_patterns.coords, ds_test_patterns.dims, name='nn_adjusted')
    else:  
                    test_results_patterns = nanremover.to_map(test_results_patterns)  ## PG: If the output is spatially flat, write back to maps
                    test_results_untransformed_patterns = obs_pipeline_patterns.inverse_transform(test_results_patterns.values) ## PG: Check preprocessing.AnomaliesScaler for changes
                    result = xr.DataArray(test_results_untransformed_patterns, test_results_patterns.coords, test_results_patterns.dims, name='nn_adjusted')

    if params["model_mean"] is not None:
        test_results_untransformed_mean= obs_pipeline_mean.inverse_transform(test_results_mean) 
        results_mean = xr.DataArray(test_results_untransformed_mean, ds_test_mean.mean('ref').expand_dims('ref', axis = -1).coords, ds_test_mean.mean('ref').expand_dims('ref', axis = -1).dims, name='nn_adjusted')
        result = result + results_mean[...,0]

    if save:
        if model_year is not None:
            Path(model_dir + f'/{model_year}_model_predictions').mkdir(parents=True, exist_ok=True)
            model_dir = model_dir + f'/{model_year}_model_predictions'
        if ensemble_mode != 'Mean':
            if np.min(test_years) != np.max(test_years):
                result.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_{np.min(test_years)}-{np.max(test_years)}_LE.nc', mode='w'))
            else:
                result.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_{np.min(test_years)}_LE.nc', mode='w'))

        else:

            if np.min(test_years) != np.max(test_years):
                result.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_{np.min(test_years)}-{np.max(test_years)}.nc', mode='w'))
            else:
                result.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_{np.min(test_years)}.nc', mode='w'))

    return result



def extract_params(model_dir):
    params = {}
    path = glob.glob(model_dir + '/*.txt')[0]
    file = open(path)
    content=file.readlines()
    for line in content:
        key = line.split('\t')[0]
        try:
            value = line.split('\t')[1].split('\n')[0]
        except:
            value = line.split('\t')[1]
        try:    
            params[key] = eval(value)
        except:
            if key == 'ensemble_list':
                ls = []
                for item in value.split('[')[1].split(']')[0].split(' '):
                    try:
                        ls.append(eval(item))
                    except:
                        pass
                params[key] = ls
            else:
                params[key] = value
    return params

if __name__ == "__main__":

    ############################################## Set_up ############################################

    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/fgco2_ems/SOM-FFN/results/Autoencoder/run_set_4'
    out_dir    = f'{out_dir_x}/N35_v22_decoupled_LNone_arch2_batch64_e60_LE' 

    lead_years = 5
    bootstrap = False
    test_years = [2022]

    observation = combine_observations(data_dir_obs, two_dim=True) # 1961.01 - 2021.12
    fct = xr.open_mfdataset(str(Path(data_dir_forecast, "*.nc")), combine='nested', concat_dim='year').load()
    ##################################################################################################
    
    # params = extract_params(out_dir)
    # print(f'loaded configuration: \n')
    # for key, values in params.items():
    #     print(f'{key} : {values} \n')
        
    # version = out_dir.split('/')[-1].split('_')[1][1:]

    # params["version_patterns"] = int(version[0])
    # try:
    #     params["version_mean"] = int(version[1])
    # except:
    #       params["version_mean"] = ""
 
    # print( f'Version patterns: {params["version_patterns"]}')
    # print( f'Version mean: {params["version_mean"]}')

    
    # if bootstrap:

    #     result_list = []
    #     ensembles = np.arange(1,11)#[f'r{i}i1p2f1' for i in range(1,11)]

    #     for it in tqdm(range(200)):

    #         ensemble_list = [random.choice(ensembles) for _ in range(len(ensembles))]
    #         result_list.append(predict(fct, observation, params, lead_years, out_dir,  test_years, ensemble_list = ensemble_list,  save=False))
        
    #     output = xr.concat(result_list, dim = 'iteration')

    #     if np.min(test_years) != np.max(test_years):
    #         output.to_netcdf(path=Path(out_dir, f'saved_model_nn_adjusted_{np.min(test_years)}-{np.max(test_years)}_bootstrap.nc', mode='w'))
    #     else:
    #         output.to_netcdf(path=Path(out_dir, f'saved_model_nn_adjusted_{np.min(test_years)}_bootstrap.nc', mode='w'))
    
    # else:
    #     predict(fct, observation, params, lead_years, out_dir,  test_years, ensemble_mode='Mean',  save=True)

    # print(f'Output dir: {out_dir}')
    # print('Saved!')

    ############################################### LE for each year #########################################################
    # for year in range(1987,2023):

    #     test_years = [year]
    #     predict(fct, observation, params, lead_years, out_dir,  test_years, ensemble_mode='LE',  save=True)
    #     print('Saved!')

    ################################################ ensembling models #########################################################

    for ens in range(1,11):
        print( f'\n running for ensemble member {ens}\n' + 
              '-------------------------------------------------------------------------------------------------------------------\n')

        params = extract_params(out_dir + f'/E{ens}')
        print(f'loaded configuration: \n')
        for key, values in params.items():
            print(f'{key} : {values} \n')
            
        version = out_dir.split('/')[-1].split('_')[1][1:]
        params["version_patterns"] = int(version[0])
        try:
            params["version_mean"] = int(version[1])
        except:
            params["version_mean"] = ""
    
        print( f'Version patterns: {params["version_patterns"]}')
        print( f'Version mean: {params["version_mean"]}')
        
            
        predict(fct, observation, params, lead_years, out_dir + f'/E{ens}',  test_years, ensemble_mode='Mean',  save=True)

        print(f'Output dir: {out_dir}')
        print('Saved!\n' + 
              '-------------------------------------------------------------------------------------------------------------------\n')
