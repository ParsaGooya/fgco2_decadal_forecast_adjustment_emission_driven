import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import dask
import xarray as xr
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models.autoencoder import Autoencoder, Autoencoder_mean
from models.unet import UNet
from models.cnn import CNN, CNN_mean
from losses import WeightedMSE,WeightedMSESignLoss
from data_utils.datahandling import combine_observations
from preprocessing import align_data_and_targets, get_coordinate_indices, create_train_mask, reshape_obs_to_data, config_grid
from preprocessing import AnomaliesScaler_v1, AnomaliesScaler_v2, Detrender, Standardizer, Normalizer, PreprocessingPipeline, Spatialnanremove
from torch_datasets import XArrayDataset
from subregions import subregions
from data_locations import LOC_FORECASTS_fgco2, LOC_OBSERVATIONS_fgco2

# specify data directories
data_dir_forecast = LOC_FORECASTS_fgco2
data_dir_obs = LOC_OBSERVATIONS_fgco2
unit_change = 60*60*24*365 * 1000 /12 * -1 ## Change units for ESM data to mol m-2 yr-1


def HP_congif(params, data_dir_obs, lead_years):

     
    ### load data
    print("Start training")
    print("Load observations")
    obs_in = combine_observations(data_dir_obs, two_dim=True) # 1961.01 - 2021.12

    if params['ensemble_list'] is not None: ## PG: calculate the mean if ensemble mean is none
        print("Load forecasts")
        ds_in = xr.open_mfdataset(str(Path(data_dir_forecast, "*.nc")), combine='nested', concat_dim='year').sel(ensembles = params['ensemble_list']).load()['fgco2']
        if params['ensemble_mode'] == 'Mean': ##
            ds_in = ds_in.mean('ensembles') ##
        else:
            print(f'Warning: ensemble_mode is {params["ensemble_mode"]}. Training for large ensemble ...')
    else:    ## Load specified members
        print("Load forecasts") 
        ds_in = xr.open_mfdataset(str(Path(data_dir_forecast, "*.nc")), combine='nested', concat_dim='year').mean('ensembles').load()['fgco2']

    obs_in = obs_in.expand_dims('channels', axis=2)

    if 'ensembles' in ds_in.dims: ### PG: add channels dimention to the correct axis based on whether we have ensembles or not
        ds_in = ds_in.expand_dims('channels', axis=3).sortby('ensembles')
    else:
        ds_in = ds_in.expand_dims('channels', axis=2) 

    ds_raw, obs_raw = align_data_and_targets(ds_in, obs_in, lead_years)  # extract valid lead times and usable years

    if 'ensembles' in ds_raw.dims: ## PG: reorder dimensions in you have ensembles
        ds_raw_ensemble_mean = ds_raw.transpose('year','lead_time','ensembles',...)* unit_change
    else:
        ds_raw_ensemble_mean = ds_raw.transpose('year','lead_time',...)* unit_change

    obs_raw = reshape_obs_to_data(obs_raw, ds_raw_ensemble_mean, return_xarray=True)
    
    if not ds_raw_ensemble_mean.year.equals(obs_raw.year):
            
            ds_raw_ensemble_mean = ds_raw_ensemble_mean.sel(year = obs_raw.year)

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
    ######################
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
            params['extra_predictor_ds'] = (extra_predictors * weights).sum(['lat','lon'])/weights.sum(['lat','lon'])
        else:  
            if not all(['ensembles' not in extra_predictors.dims, 'ensembles' in ds_raw_ensemble_mean.dims]):              
                ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, extra_predictors], dim = 'channels')
            else:
                ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, extra_predictors.expand_dims(ensembles = ds_raw_ensemble_mean['ensembles'], axis = 2) ], dim = 'channels')
            params['extra_predictor_ds'] = None 
    else:
        params['extra_predictor_ds'] = None 

    params['nanremover'] = nanremover

    return  (ds_raw_ensemble_mean,ds_raw_ensemble_mean_patterns), (obs_raw, obs_raw_patterns), params



def smooth_curve(list, factor = 0.9):
    smoothed_list = []
    for point in list:
        if smoothed_list:
            previous = smoothed_list[-1]
            smoothed_list.append(previous* factor + point * (1- factor))
        else:
            smoothed_list.append(point)
    return smoothed_list



def training_hp(hyperparamater_grid: dict, params:dict, ds_tuple: XArrayDataset ,obs_tuple: XArrayDataset , test_year, n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None, sub_model = 'Patterns'):
    assert sub_model in ['Patterns','Mean']
    if sub_model == 'Mean':
        if  params["model_mean"] is None:
            raise RuntimeError("Model_mean cannot be None in sub-model = 'Mean'.")
    else:
        params["model_mean"] = None

    if sub_model == 'Patterns':
        if  params["model_patterns"] is None:
            raise RuntimeError("Model_patterns cannot be None in sub-model = 'Patterns'.")
    else:
        params["model_patterns"] = None
    
    ds_raw_ensemble_mean, ds_raw_ensemble_mean_patterns = ds_tuple
    obs_raw, obs_raw_patterns = obs_tuple

    
    if params["model_mean"] != None :
        assert params['version_mean'] in [1,2,3]
    if params["model_patterns"] != None :
        assert params['version_patterns'] in [0,1,2,3]

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
    
    if params["model_patterns"] is not None:

        if params['ensemble_mode'] != 'Mean':
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
  

    for key, value in hyperparamater_grid.items():
            params[key] = value 


    ##### PG: Ensemble members to load 
    ensemble_list = params['ensemble_list']
    ###### PG: Add ensemble features to training features
    ensemble_mode = params['ensemble_mode'] ##
    model_patterns = params["model_patterns"]
    epochs = params["epochs"]
    batch_shuffle = params["batch_shuffle"]
    optimizer = params["optimizer"]
    loss_region = params["loss_region"]
    subset_dimensions = params["subset_dimensions"]
    Biome = params['Biome']
    num_val_years = params['num_val_years']
    lr = params["lr"]
    batch_size = params["batch_size"]
    l2_reg = params["L2_reg"]

    if params['model_mean'] is None:

        forecast_preprocessing_steps_mean = []
        observations_preprocessing_steps_mean = []

    if params['model_patterns'] is None:

        forecast_preprocessing_steps_patterns = []
        observations_preprocessing_steps_patterns = []

    test_years = test_year

    if n_runs > 1:
        numpy_seed = None
        torch_seed = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    print(f"Start run for test year {test_year}...")

    train_years = ds_raw_ensemble_mean.year[ds_raw_ensemble_mean.year < test_year-num_val_years].to_numpy()

    n_train = len(train_years)
    train_mask = create_train_mask(ds_raw_ensemble_mean[:n_train,...])
    
    ds_baseline_mean = ds_raw_ensemble_mean[:n_train,...]
    obs_baseline_mean = obs_raw[:n_train,...]
    ds_baseline_patterns = ds_raw_ensemble_mean_patterns[:n_train,...]
    obs_baseline_patterns = obs_raw_patterns[:n_train,...]
    train_mask = create_train_mask(ds_baseline_patterns)

    if 'ensembles' in ds_raw_ensemble_mean.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline_patterns.shape)
    else:
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline_patterns.shape)
    
    preprocessing_mask_obs_mean = np.broadcast_to(train_mask[...,None,None], obs_baseline_mean.shape)
    preprocessing_mask_obs_patterns = np.broadcast_to(train_mask[...,None,None,None], obs_baseline_patterns.shape)


    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    # Data preprocessing
    
    ds_pipeline_mean = PreprocessingPipeline(forecast_preprocessing_steps_mean).fit(ds_baseline_mean, mask=preprocessing_mask_fct)
    ds_mean = ds_pipeline_mean.transform(ds_raw_ensemble_mean)
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

    year_max = ds_patterns[:n_train + num_val_years].year[-1].values
    # TRAIN MODEL

    lead_time = None

    ds_train_mean = ds_mean[:n_train,...]
    obs_train_mean = obs_mean[:n_train,...]
    ds_validation_mean = ds_mean[n_train:n_train + num_val_years,...]
    obs_validation_mean = obs_mean[n_train:n_train + num_val_years,...]
    ds_train_patterns = ds_patterns[:n_train,...]
    obs_train_patterns = obs_patterns[:n_train,...]

    ds_validation_patterns = ds_patterns[n_train:n_train + num_val_years,...]
    obs_validation_patterns = obs_patterns[n_train:n_train + num_val_years,...]

    ##### PG: The ocean carbon flux has NaN values over land in both forecast and obs data and these are not necessarily in the excat same grid points. ###
    ##### We need to extract the common grid points where both obs and model data exist. That said, we need to flatten both the training and target data
    ##### I defined a Nanremover class. See preprocessing.py.
        
    weights = np.cos(np.ones_like(ds_train_patterns.lon) * (np.deg2rad(ds_train_patterns.lat.to_numpy()))[..., None])  # Moved this up
    weights = xr.DataArray(weights, dims = ds_train_patterns.dims[-2:], name = 'weights').assign_coords({'lat': ds_train_patterns.lat, 'lon' : ds_train_patterns.lon}) # Create an DataArray to pass to Spatialnanremove() 
    weights_val = weights.copy()


    ### Increase the weight over some biome :
    if Biome is not None:
        if type(Biome) == dict:
            for ind, sc in Biome.items():
                    weights = weights + (sc-1) * weights.where(biomes == ind).fillna(0)  
        elif type(Biome) == list:
            for ind in Biome:
                    weights = weights + weights.where(biomes == ind).fillna(0) 
        else:
                weights = weights + weights.where(biomes == Biome).fillna(0)
##########################################            
    # nanremover = Spatialnanremove() ## PG: Get an instance of the class
    # nanremover.fit(ds_train_patterns[:,:12,...], obs_train_patterns[:,:12,...]) ## PG:extract the commong grid points between training and obs data
    nanremover = params['nanremover']
    
    ds_train_mean = nanremover.sample(ds_train_mean) 
    if params["model_mean"] in [CNN_mean]:
            ds_train_mean = nanremover.to_map(ds_train_mean).fillna(0.0) ## PG: flatten and sample training data at those locations

    ds_train_patterns = nanremover.sample(ds_train_patterns)## PG: flatten and sample training data at those locations
    obs_train_patterns = nanremover.sample(obs_train_patterns) ## PG: flatten and sample obs data at those locations
    ########################################################################

    if model_patterns in [UNet , CNN]: ## PG: If the model starts with a nn.Conv2d write back the flattened data to maps.

        ds_train_patterns = nanremover.to_map(ds_train_patterns).fillna(0.0) ## PG: fill NaN values with 0.0 for training
        obs_train_patterns = nanremover.to_map(obs_train_patterns).fillna(0.0) ## PG: fill NaN values with 0.0 for training

        img_dim_patterns = ds_train_patterns.shape[-2] * ds_train_patterns.shape[-1] 
        if loss_region is not None:
            loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region)
        
        else:
            loss_region_indices = None
    
    else: ## PG: If you have a dense first layer keep the data flattened.
        
        weights = nanremover.sample(weights) ## PG: flatten and sample weighs at those locations
        weights_val = nanremover.sample(weights_val)
        img_dim_patterns = ds_train_patterns.shape[-1] ## PG: The input dim is now the length of the flattened dimention.
        if loss_region is not None:
    
            loss_region_indices = nanremover.extract_indices(subregions[loss_region]) ## PG: We need 1D index list rather than 2D lat:lon.

        else:
            loss_region_indices = None

    weights = weights.values
    weights_val = weights_val.values
    if params["model_mean"] in [Autoencoder_mean]:
        img_dim_mean = ds_train_mean.shape[-1]


################################################## pattern training ##################################################################

    hyperparam = params['hyperparam']
    reg_scale = params["reg_scale"]
    hidden_dims = params["hidden_dims"]
    time_features_patterns = params["time_features_patterns"].copy() if params["time_features_patterns"] is not None else None
    batch_normalization = params["batch_normalization"]
    dropout_rate = params["dropout_rate"]
    
    if params["model_patterns"] is not None:
        if time_features_patterns is None:

                add_feature_dim_patterns = 0
        else:

                add_feature_dim_patterns = len(time_features_patterns)
    
        if model_patterns == Autoencoder:
            net_patterns = model_patterns(img_dim_patterns, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim_patterns, append_mode=params['append_mode_patterns'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, device = device)
        elif model_patterns == UNet:
            net_patterns = model_patterns(n_channels_x= len(ds_train_patterns.channels) + add_feature_dim_patterns , bilinear = params['bilinear'])
        elif model_patterns == CNN: ## PG: Combining CNN encoder with dense decoder
            net_patterns = model_patterns(len(ds_train_patterns.channels) + add_feature_dim_patterns ,hidden_dims, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size )
    
        net_patterns.to(device)
        optimizer = torch.optim.Adam(net_patterns.parameters(), lr=lr, weight_decay = l2_reg)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        train_set_patterns = XArrayDataset(ds_train_patterns, obs_train_patterns, mask=train_mask, lead_time_mask = params['lead_time_mask'], in_memory=True, lead_time=lead_time, time_features=time_features_patterns, aligned = True, year_max=year_max, model = model_patterns.__name__) 
        dataloader_patterns = DataLoader(train_set_patterns, batch_size=batch_size, shuffle=batch_shuffle)

        if reg_scale is None: ## PG: if no penalizing for negative anomalies
            criterion_patterns = WeightedMSE(weights=weights, device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices)
        else:
            criterion_patterns = WeightedMSESignLoss(weights=weights, device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, min_val=0, max_val=None)

        # EVALUATE MODEL
        ds_validation_patterns = nanremover.sample(ds_validation_patterns, mode = 'Eval')  ## PG: Sample the test data at the common locations
        obs_validation_patterns = nanremover.sample(obs_validation_patterns)
        if model_patterns in [UNet,CNN]:
            ds_validation_patterns = nanremover.to_map(ds_validation_patterns).fillna(0.0)  ## PG: Write back to map if the model starts with a nn.Conv2D
            obs_validation_patterns = nanremover.to_map(obs_validation_patterns).fillna(0.0)

        val_mask = create_train_mask(ds_validation_patterns)
        validation_set_patterns = XArrayDataset(ds_validation_patterns, obs_validation_patterns, mask=val_mask, lead_time=None,lead_time_mask = params['lead_time_mask'] , time_features=time_features_patterns,  in_memory=False, aligned = True, year_max=year_max, model =  model_patterns.__name__) 
        dataloader_val_patterns = DataLoader(validation_set_patterns, batch_size=batch_size, shuffle=True)   

        # criterion_eval_global = GlobalLoss(device=device, scale=1, weights=weights_val, loss_area=None)
        criterion_eval_mean_patterns = WeightedMSE(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
        if reg_scale is None: ## PG: if no penalizing for negative anomalies
                criterion_eval_patterns = WeightedMSE(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
        else:
                criterion_eval_patterns = WeightedMSESignLoss(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, min_val=0, max_val=None)
    
        epoch_loss_train_patterns = []
        epoch_loss_val_patterns = []
        # epoch_loss_val_global_patterns = []
        epoch_loss_val_mean_patterns = []

        for epoch in tqdm.tqdm(range(epochs)):
            net_patterns.train()
            batch_loss = 0
            
            for batch, (x, y) in enumerate(dataloader_patterns):
                
                if (type(x) == list) or (type(x) == tuple):
                    x = (x[0].to(device), x[1].to(device))
                else:
                    x = x.to(device)
                if (type(y) == list) or (type(y) == tuple):
                    y, m = (y[0].to(device), y[1].to(device))
                else:
                    y = y.to(device)
                    m  = None
                optimizer.zero_grad()
                adjusted_forecast_patterns = net_patterns(x)
                loss_patterns = criterion_patterns(adjusted_forecast_patterns, y, mask = m) #/ (torch.max(y) - torch.min(y))
                batch_loss += loss_patterns.item()
                loss_patterns.backward()
                optimizer.step()
                
            epoch_loss_train_patterns.append(batch_loss / len(dataloader_patterns))

            if epoch<=30: ###
                scheduler.step() ####
                
            net_patterns.eval()
            val_loss = 0
            # val_loss_global = 0
            val_loss_mean = 0
            
            for batch, (x, target) in enumerate(dataloader_val_patterns):         
                with torch.no_grad():            
                    if (type(x) == list) or (type(x) == tuple):
                        # test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                        test_raw = (x[0].to(device), x[1].to(device))
                    else:
                        # test_raw = x.unsqueeze(0).to(device)
                        test_raw = x.to(device)
                    # test_obs = target.unsqueeze(0).to(device)
                    if (type(target) == list) or (type(target) == tuple):
                        test_obs, m = (target[0].to(device), target[1].to(device))
                    else:
                        test_obs = target.to(device)
                        m = None
                    test_adjusted_patterns = net_patterns(test_raw)
                    loss__patterns = criterion_eval_patterns(test_adjusted_patterns, test_obs, mask = m) #/ (torch.max(test_obs) - torch.min(test_obs))
                    val_loss += loss__patterns.item()
                    # loss_global = criterion_eval_global(test_adjusted, test_obs)
                    # val_loss_global += loss_global.item()
                    loss_mean = criterion_eval_mean_patterns(test_adjusted_patterns, test_obs, mask = m)
                    val_loss_mean += loss_mean.item()


            epoch_loss_val_patterns.append(val_loss / len(dataloader_val_patterns))
            # epoch_loss_val_global.append(val_loss_global / len(validation_set))
            epoch_loss_val_mean_patterns.append(val_loss_mean / len(dataloader_val_patterns))
            # Store results as NetCDF            
################################################## mean training ##################################################################

    
    time_features_mean = params["time_features_mean"].copy()

    if params["model_mean"] is not None:
        if time_features_mean is None:

                add_feature_dim_mean = 0
        else:

                add_feature_dim_mean = len(time_features_mean) 
        if params['extra_predictor_ds'] is not None:
                add_feature_dim_mean = add_feature_dim_mean + len(params['extra_predictors_mean'])



        if params["model_mean"] == Autoencoder_mean:
            net_mean = Autoencoder_mean(img_dim_mean, [360, 180, 90, 45], added_features_dim=add_feature_dim_mean, append_mode=params['append_mode_mean'], batch_normalization=batch_normalization, dropout_rate=dropout_rate)
        else:
            net_mean = params["model_mean"](  n_channels_x=len(ds_train_mean.channels) + add_feature_dim_mean  , dense_dims= [512,128,32] )
    
    

        net_mean.to(device)
        optimizer = torch.optim.Adam(net_mean.parameters(), lr=lr, weight_decay = l2_reg)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        train_set_mean = XArrayDataset(ds_train_mean, obs_train_mean, mask=train_mask,extra_predictors = params['extra_predictor_ds'], in_memory=True, lead_time=lead_time ,lead_time_mask = params['lead_time_mask'], time_features=time_features_mean, aligned = True, year_max=year_max, model = params['model_mean'].__name__ ) 
        dataloader_mean= DataLoader(train_set_mean, batch_size=batch_size, shuffle=batch_shuffle)
        criterion_mean = WeightedMSE(weights=np.ones([1]), device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices)


        # EVALUATE MODEL
        ds_validation_mean = nanremover.sample(ds_validation_mean, mode = 'Eval')  ## PG: Sample the test data at the common locations
        if params['model_mean'] in [CNN_mean]:
            ds_validation_mean = nanremover.to_map(ds_validation_mean).fillna(0.0) 
        val_mask = create_train_mask(ds_validation_mean)
        validation_set_mean = XArrayDataset(ds_validation_mean, obs_validation_mean, mask=val_mask,extra_predictors = params['extra_predictor_ds'], lead_time=None, lead_time_mask = params['lead_time_mask'] , time_features=time_features_mean,  in_memory=False, aligned = True, year_max=year_max, model = params['model_mean'].__name__ ) 
        dataloader_val_mean = DataLoader(validation_set_mean, batch_size=batch_size, shuffle=True)   

        criterion_eval_mean =  WeightedMSE(weights=np.ones([1]), device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
    
        epoch_loss_train_mean  = []
        epoch_loss_val_mean  = []


        for epoch in tqdm.tqdm(range(epochs)):
            net_mean.train()
            batch_loss = 0
            
            for batch, (x, y) in enumerate(dataloader_mean ):
                
                if (type(x) == list) or (type(x) == tuple):
                    x = (x[0].to(device), x[1].to(device))
                else:
                    x = x.to(device)
                if (type(y) == list) or (type(y) == tuple):
                    y, m = (y[0].to(device), y[1].to(device))
                else:
                    y = y.to(device)
                    m  = None
                optimizer.zero_grad()
                adjusted_forecast_mean  = net_mean (x)
                loss_mean  = criterion_mean (adjusted_forecast_mean , y, mask = m) #/ (torch.max(y) - torch.min(y))
                batch_loss += loss_mean .item()
                loss_mean.backward()
                optimizer.step()
                
            epoch_loss_train_mean .append(batch_loss / len(dataloader_mean ))

            if epoch<=30: ###
                scheduler.step() ####
                
            net_mean .eval()
            val_loss = 0

            
            for batch, (x, target) in enumerate(dataloader_val_mean ):         
                with torch.no_grad():            
                    if (type(x) == list) or (type(x) == tuple):
                        # test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                        test_raw = (x[0].to(device), x[1].to(device))
                    else:
                        # test_raw = x.unsqueeze(0).to(device)
                        test_raw = x.to(device)
                    # test_obs = target.unsqueeze(0).to(device)
                    if (type(target) == list) or (type(target) == tuple):
                        test_obs, m = (target[0].to(device), target[1].to(device))
                    else:
                        test_obs = target.to(device)
                        m = None
                    test_adjusted_mean  = net_mean(test_raw)
                    loss__mean  = criterion_eval_mean(test_adjusted_mean , test_obs, mask = m) #/ (torch.max(test_obs) - torch.min(test_obs))
                    val_loss += loss__mean.item()



            epoch_loss_val_mean.append(val_loss / len(dataloader_val_mean))

############################################################################################################################################
    if sub_model == 'Patterns':

        epoch_loss_val =  np.array(epoch_loss_val_patterns) 
        epoch_loss_val_mean =  np.array(epoch_loss_val_mean_patterns) 
        epoch_loss_train = np.array(epoch_loss_train_patterns)
    ############################################################################################################################################
    else:

        epoch_loss_val =  np.array(epoch_loss_val_mean)  
        epoch_loss_val_mean = np.array(epoch_loss_val_mean)
        epoch_loss_train = np.array(epoch_loss_train_mean)
    ############################################################################################################################################
    epoch_loss_val = smooth_curve(epoch_loss_val)
    epoch_loss_val_mean = smooth_curve(epoch_loss_val_mean)

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(2,epochs+1), epoch_loss_train[1:], label = 'Train')
    plt.plot(np.arange(2,epochs+1), epoch_loss_val[1:], label = 'Validation')
    plt.title(f'{hyperparamater_grid}')
    plt.legend()
    plt.ylabel('MSE')

    plt.xlabel('Epoch')
    
    plt.grid()
    plt.show()
    plt.savefig(results_dir+f'/{sub_model}/val-train_loss_1982-{test_year}-{hyperparamater_grid}.png')
    plt.close()


    with open(Path(results_dir+f'/{sub_model}/', "Hyperparameter_training.txt"), 'a') as f:
        f.write(

            f"{hyperparamater_grid} ---> val_loss at best epoch: {min(epoch_loss_val)} at {np.argmin(epoch_loss_val)+1}  (MSE : {epoch_loss_val_mean[np.argmin(epoch_loss_val)]})\n" +  ## PG: The scale to be passed to Signloss regularization
            f"-------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n" 
        )
    return epoch_loss_val_mean[np.argmin(epoch_loss_val)], epoch_loss_val,epoch_loss_val_mean, epoch_loss_train

                                 #########         ##########

    
def run_hp_tunning( ds_tuple: XArrayDataset ,obs_tuple: XArrayDataset,  hyperparameterspace, params:dict, y_start: int, out_dir_x , n_runs=1, numpy_seed=None, torch_seed=None, sub_model = 'both' ):
  
    y_end = ds_tuple[0].year[-1].values +1 

    if sub_model == 'both':

            if type(hyperparameterspace) == tuple:
                hyperparameterspace_patterns = hyperparameterspace[0]
                hyperparameterspace_mean = hyperparameterspace[1]
            else:
                hyperparameterspace_patterns = hyperparameterspace_mean = hyperparameterspace
                config_dict_patterns = config_dict_mean = config_dict

            losses = (np.zeros(len(hyperparameterspace_patterns)), np.zeros(len(hyperparameterspace_mean)))
            val_losses = (np.zeros([y_end - y_start + 1 , len(hyperparameterspace_patterns), params['epochs']]), np.zeros([y_end - y_start + 1 , len(hyperparameterspace_mean), params['epochs']])) 
            val_losses_mean = (np.zeros([y_end - y_start + 1 , len(hyperparameterspace_patterns), params['epochs']]), np.zeros([y_end - y_start + 1 , len(hyperparameterspace_mean), params['epochs']])) 
            train_losses = (np.zeros([y_end - y_start + 1, len(hyperparameterspace_patterns), params['epochs']]), np.zeros([y_end - y_start + 1, len(hyperparameterspace_mean), params['epochs']]))
    else:
            val_losses = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
            val_losses_mean = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
            train_losses = np.zeros([y_end - y_start + 1, len(hyperparameterspace), params['epochs']]) ####
            losses = np.zeros(len(hyperparameterspace))
  
    for ind_, test_year in enumerate(range(y_start,y_end+1)):
    
        if len(params["extra_predictors_mean"]) > 0:
            out_dir_x = out_dir_x + '_extra_predictors'

        out_dir    = f'{out_dir_x}/_{test_year}' 
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        with open(Path(out_dir, "Hyperparameter_training.txt"), 'w') as f:
            f.write(
                f"model_patterns\t{params['model_patterns']}\n" +
                f"model_mean\t{params['model_mean']}\n" +
                "default set-up:\n" + 
                f"lead_time_mask\t{params['lead_time_mask']}\n" +
                f"hidden_dims\t{params['hidden_dims']}\n" +
                f"time_features_patterns\t{params['time_features_patterns']}\n" +
                f"append_mode_patterns\t{params['append_mode_patterns']}\n" +
                f"ensemble_list\t{params['ensemble_list']}\n" + ## PG: Ensemble list
                f"batch_normalization\t{params['batch_normalization']}\n" +
                f"dropout_rate\t{params['dropout_rate']}\n" +
                f"lr_default\t{params['lr']}\n" +
                f"L2_reg_mean\t{params['L2_reg']}\n" +
                f"time_features_mean\t{params['time_features_mean']}\n" +
                f"extra_predictors_mean\t{params['extra_predictors_mean']}\n" +
                f"append_mode_mean\t{params['append_mode_mean']}\n\n\n" +
                ' ----------------------------------------------------------------------------------\n'
            )


        if sub_model == 'both':
            Path(out_dir+f'/Mean').mkdir(parents=True, exist_ok=True)
            Path(out_dir+f'/Patterns').mkdir(parents=True, exist_ok=True)


            default_params = params.copy()
            for ind, dic in enumerate(hyperparameterspace_patterns):
                print(f'Patterns model: Training for {dic}')
                losses[0][ind], val_losses[0][ind_, ind, :], val_losses_mean[0][ind_, ind, :],  train_losses[0][ind_, ind, :] = training_hp( ds_tuple =  ds_tuple, obs_tuple = obs_tuple , hyperparamater_grid= dic, params = default_params ,test_year=test_year, n_runs=n_runs, results_dir=out_dir, numpy_seed=numpy_seed, torch_seed=torch_seed, sub_model = 'Patterns')
                
            with open(Path(out_dir+f'/Patterns', "Hyperparameter_training.txt"), 'a') as f:
                f.write(
                    f"Best MSE: {min(losses[0])} --> {hyperparameterspace_patterns[np.argmin(losses[0])]} \n" +  ## PG: The scale to be passed to Signloss regularization
                    f"--------------------------------------------------------------------------------------------------------\n")
            
            default_params = params.copy()
            for ind, dic in enumerate(hyperparameterspace_mean):
                print(f'Mean model: Training for {dic}')   
                losses[1][ind], val_losses[1][ind_, ind, :], val_losses_mean[0][ind_, ind, :],  train_losses[1][ind_, ind, :] = training_hp( ds_tuple =  ds_tuple, obs_tuple = obs_tuple , hyperparamater_grid= dic, params = default_params ,test_year=test_year, n_runs=n_runs, results_dir=out_dir, numpy_seed=numpy_seed, torch_seed=torch_seed, sub_model = 'Mean')
 
            with open(Path(out_dir+f'/Mean', "Hyperparameter_training.txt"), 'a') as f:
                f.write(
                    f"Best MSE: {min(losses[1])} --> {hyperparameterspace_mean[np.argmin(losses[1])]} \n" +  ## PG: The scale to be passed to Signloss regularization
                    f"--------------------------------------------------------------------------------------------------------\n" )
                
            with open(Path(out_dir, "Hyperparameter_training.txt"), 'a') as f:
                f.write(
                    f"Best MSE Patterns: {min(losses[0])} --> {hyperparameterspace_patterns[np.argmin(losses[0])]} \n" +  ## PG: The scale to be passed to Signloss regularization
                    f"--------------------------------------------------------------------------------------------------------\n" +
                    f"Best MSE Mean: {min(losses[1])} --> {hyperparameterspace_mean[np.argmin(losses[1])]} \n" +  ## PG: The scale to be passed to Signloss regularization
                    f"--------------------------------------------------------------------------------------------------------\n" )
        
            print(f"Best MSE Patterns: {min(losses[0])} --> {hyperparameterspace_patterns[np.argmin(losses[0])]}")
            print(f"Best MSE Mean: {min(losses[1])} --> {hyperparameterspace_mean[np.argmin(losses[1])]}")
            print(f'Output dir: {out_dir}')
            print('Training done.')


        
        else:

            Path(out_dir+f'/{sub_model}').mkdir(parents=True, exist_ok=True)

        
            for ind, dic in enumerate(hyperparameterspace):
                print(f'Training for {dic}')
                losses[ind], val_losses[ind_, ind, :], val_losses_mean[ind_, ind, :], train_losses[ind_, ind, :] = training_hp( ds_tuple =  ds_tuple, obs_tuple = obs_tuple , hyperparamater_grid= dic, params = params ,test_year=test_year, n_runs=n_runs, results_dir=out_dir, numpy_seed=numpy_seed, torch_seed=torch_seed, sub_model = sub_model)
            
            with open(Path(out_dir+f'/{sub_model}', "Hyperparameter_training.txt"), 'a') as f:
                f.write(
        
                    f"Best MSE: {min(losses)} --> {hyperparameterspace[np.argmin(losses)]} \n" +  ## PG: The scale to be passed to Signloss regularization
                    f"--------------------------------------------------------------------------------------------------------\n" 
                )
            with open(Path(out_dir, "Hyperparameter_training.txt"), 'a') as f:
                f.write(
                    f"Best MSE {sub_model}: {min(losses)} --> {hyperparameterspace[np.argmin(losses)]} \n" +  ## PG: The scale to be passed to Signloss regularization
                    f"--------------------------------------------------------------------------------------------------------\n" )

            print(f"Best MSE: {min(losses)} --> {hyperparameterspace[np.argmin(losses)]}")
            print(f'Output dir: {out_dir}')
            print('Training done.')

    if sub_model == 'both':
            coords = []
            for item in hyperparameterspace_patterns:
                coords.append(str(tuple(item.values())))

            ds_val = xr.DataArray(val_losses[0], dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
            ds_val.attrs['hyperparameters'] = list(hyperparameterspace_patterns[0].keys())
            ds_val.to_netcdf(out_dir_x + '/validation_losses_Patterns.nc')

            ds_val_mean = xr.DataArray(val_losses_mean[0], dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
            ds_val_mean.attrs['hyperparameters'] = list(hyperparameterspace_patterns[0].keys())
            ds_val_mean.to_netcdf(out_dir_x + '/validation_MSE.nc')

            ds_train = xr.DataArray(train_losses[0], dims = ['test_years', 'hyperparameters','epochs'], name = 'Train_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1), 'hyperparameters': coords})
            ds_train.attrs['hyperparameters'] = list(hyperparameterspace_patterns[0].keys())
            ds_train.to_netcdf(out_dir_x + '/train_losses_Patterns.nc')

            coords = []
            for item in hyperparameterspace_mean:
                coords.append(str(tuple(item.values())))

            ds_val = xr.DataArray(val_losses[1], dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
            ds_val.attrs['hyperparameters'] = list(hyperparameterspace_mean[0].keys())
            ds_val.to_netcdf(out_dir_x + '/validation_losses_Mean.nc')

            ds_val_mean = xr.DataArray(val_losses_mean[1], dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
            ds_val_mean.attrs['hyperparameters'] = list(hyperparameterspace_mean[0].keys())
            ds_val_mean.to_netcdf(out_dir_x + '/validation_MSE.nc')

            ds_train = xr.DataArray(train_losses[1], dims = ['test_years', 'hyperparameters','epochs'], name = 'Train_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1), 'hyperparameters': coords})
            ds_train.attrs['hyperparameters'] = list(hyperparameterspace_mean[0].keys())
            ds_train.to_netcdf(out_dir_x + '/train_losses_Mean.nc')
    else:
            coords = []
            for item in hyperparameterspace:
                coords.append(str(tuple(item.values())))

            ds_val = xr.DataArray(val_losses, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
            ds_val.attrs['hyperparameters'] = list(hyperparameterspace[0].keys())
            ds_val.to_netcdf(out_dir_x + '/validation_losses.nc')

            ds_val_mean = xr.DataArray(val_losses_mean, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
            ds_val_mean.attrs['hyperparameters'] = list(hyperparameterspace[0].keys())
            ds_val_mean.to_netcdf(out_dir_x + '/validation_MSE.nc')

            ds_train = xr.DataArray(train_losses, dims = ['test_years', 'hyperparameters','epochs'], name = 'Train_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1), 'hyperparameters': coords})
            ds_train.attrs['hyperparameters'] = list(hyperparameterspace[0].keys())
            ds_train.to_netcdf(out_dir_x + '/train_losses.nc')
            

if __name__ == "__main__":

    
    # test_year =  2019 # last n years to test consecutively
    lead_years = 5
    n_runs = 1  # number of training runs

    params = {
        'ensemble_list' : None, ## PG
        'ensemble_mode' : 'Mean',
        "optimizer": torch.optim.Adam,
        "lr": 0.001,
        "batch_shuffle" : True,
        "loss_region": None,
        "subset_dimensions": None,
        "lead_time_mask" : None,
        "epochs": 200,
        "batch_size": 8,
        "L2_reg" : 0, 
        "model_patterns": Autoencoder,
        "hidden_dims":  [[3600, 1800, 900], [1800, 3600]],# [3600, 1800, 900, 1800, 3600],
        "time_features_patterns":  ['month_sin', 'month_cos'],
        "append_mode_patterns": 1,
        "dropout_rate": 0,
        "batch_normalization": False,
        "hyperparam"  :1, 
        "reg_scale" : None,
        "model_mean": Autoencoder_mean,
        "time_features_mean":  ['month_sin', 'month_cos'],
        "extra_predictors_mean" : [], 
        "append_mode_mean": 1,
    }


    if params["model_mean"] == CNN_mean:
        params["append_mode_mean"] = None
        
    if params["model_patterns"] != Autoencoder:
        params["append_mode_patterns"] = None


    params['ensemble_list'] = None#[f'r{e}i1p2f1' for e in range(1,21,1)] ## PG

    biomes = xr.open_dataset('/home/rpg002/fgco2_decadal_forecast_adjustment/Time_Varying_Biomes.nc').MeanBiomes.transpose()
    params['Biome'] = None


    #######################################  Don't touch the following #######################################################

    ds_tuple, obs_tuple, params = HP_congif(params, data_dir_obs, lead_years)
    

    ################################################ specifics #################################################################

    y_start = 2006
  
    params['num_val_years'] = 5

    config_dict_patterns = {'lr' : [0.001,0.01], 'batch_size' : [16,64,128], 'L2_reg':[0, 0.0001,0.001], "dropout_rate" : [0,0.1,0.5 ], 'reg_scale' : [None, 1,3]}
    hyperparameterspace_patterns = config_grid(config_dict_patterns).draw_random(10)

    config_dict_mean = {'lr' : [0.001,0.01],'batch_size' : [16,64,128], 'L2_reg':[0, 0.001, 0.01], "dropout_rate" : [0,0.1,0.5 ]}
    hyperparameterspace_mean = config_grid(config_dict_mean).draw_random(12)

    params['arch'] = 2
    params['version_patterns'] = 3  ## 0,1,2,3
    params['version_mean'] = 3  ## 1,2,3

    sub_model = 'Mean'
    # hyperparameterspace = (hyperparameterspace_patterns,hyperparameterspace_mean )
    # config_dict = (config_dict_patterns, config_dict_mean)
    hyperparameterspace = hyperparameterspace_mean
    config_dict = config_dict_mean
    ##############################################################################################################################

    if params["arch"] == 2:
        params['hidden_dims'] = [[360, 180, 90], [180, 360]]

    ##############################################################################################################################

    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/fgco2_ems/SOM-FFN/results/{params["model_patterns"].__name__}/run_set_6/Model_tunning/arch{params["arch"]}/decoupled_batch_reg_tunning_adj_lr_v{params["version_patterns"]}{params["version_mean"]}_{sub_model}_e200_2'

    if params["model_mean"] == CNN_mean:
        out_dir_x = out_dir_x + '_CNN_mean'
    
    run_hp_tunning(ds_tuple = ds_tuple ,obs_tuple = obs_tuple,  hyperparameterspace = hyperparameterspace, params = params, y_start = y_start, out_dir_x = out_dir_x, n_runs=1, numpy_seed=1, torch_seed=1, sub_model = sub_model )


