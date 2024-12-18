import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import dask
import xarray as xr
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models.autoencoder import Autoencoder, Autoencoder_mean
from models.unet import UNet
from models.cnn import CNN, SCNN, CNN_mean
from losses import WeightedMSE, WeightedMSESignLoss
from data_utils.datahandling import combine_observations
from preprocessing import align_data_and_targets, get_coordinate_indices, create_train_mask, reshape_obs_to_data
from preprocessing import AnomaliesScaler_v1, AnomaliesScaler_v2, Standardizer, PreprocessingPipeline, Spatialnanremove, calculate_climatology
from torch_datasets import XArrayDataset
from subregions import subregions
from data_locations import LOC_FORECASTS_fgco2, LOC_OBSERVATIONS_fgco2

# specify data directories
data_dir_forecast = LOC_FORECASTS_fgco2
data_dir_obs = LOC_OBSERVATIONS_fgco2
unit_change = 60*60*24*365 * 1000 /12 * -1 ## Change units for ESM data to mol m-2 yr-1


def run_training(params, n_years, lead_years, n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None, save = False):
    if params["model_mean"] != None :
        assert params['version_mean'] in [1,2,3]
    assert params['version_patterns'] in [0,1,2,3]

    if params["model_patterns"] != Autoencoder:
        params["append_mode_patterns"] = None


    if  params["model_mean"] is  None:
        params["extra_predictors_mean"] = []
        params["append_mode_mean"] = None
        params["time_features_mean"] = []
        params['epochs_mean'] = params['epochs_patterns']

    if params["model_mean"] == CNN_mean:
        params["append_mode_mean"] = None

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



    print("Start training")
    print("Load observations")
    obs_in = combine_observations(data_dir_obs, two_dim=True) # 1961.01 - 2021.12

    ##### PG: Ensemble members to load 
    ensemble_list = params['ensemble_list']
    ###### PG: Add ensemble features to training features
    ensemble_mode = params['ensemble_mode'] ##

    Biome = params['Biome']
    
    if params["arch"] == 2:
        params["hidden_dims"] = [[360, 180, 90], [180, 360]]

    if ensemble_list is not None: ## PG: calculate the mean if ensemble mean is none
        print("Load forecasts")
        ds_in = xr.open_mfdataset(str(Path(data_dir_forecast, "*.nc")), combine='nested', concat_dim='year').sel(ensembles = ensemble_list).load()['fgco2']
        if ensemble_mode == 'Mean': ##
            ds_in = ds_in.mean('ensembles') ##
        else:
            print(f'Warning: ensemble_mode is {ensemble_mode}. Training for large ensemble ...')

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

    ##### PG: The ocean carbon flux has NaN values over land in both forecast and obs data and these are not necessarily in the excat same grid points. ###
    ##### We need to extract the common grid points where both obs and model data exist. That said, we need to flatten both the training and target data
    ##### I defined a Nanremover class. See preprocessing.py.
    
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



    hyperparam = params["hyperparam"]
    reg_scale = params["reg_scale"]
    model_patterns = params["model_patterns"]
    hidden_dims = params["hidden_dims"]
    time_features_patterns = params["time_features_patterns"]
    epochs_patterns = params["epochs_patterns"]
    epochs_mean = params["epochs_mean"]
    batch_size_patterns = params["batch_size_patterns"]
    batch_size_mean = params["batch_size_mean"]
    batch_normalization = params["batch_normalization"]
    dropout_rate = params["dropout_rate"]
    optimizer = params["optimizer"]
    lr_patterns = params["lr_patterns"]
    lr_mean = params["lr_mean"]
    l2_reg_patterns = params["L2_reg_patterns"]
    l2_reg_mean = params["L2_reg_mean"]
    loss_region = params["loss_region"]
    subset_dimensions = params["subset_dimensions"]

    if params['model_mean'] is None:

        forecast_preprocessing_steps_mean = []
        observations_preprocessing_steps_mean = []
        

    time_features_mean = params['time_features_mean']


    test_years = ds_raw_ensemble_mean.year[-n_years:].to_numpy()
    test_years = [*test_years,test_years[-1] + 1]

    if n_runs > 1:
        numpy_seed = None
        torch_seed = None

    with open(Path(results_dir, "training_parameters.txt"), 'w') as f:
        f.write(
            f"ensemble_list\t{ensemble_list}\n" + ## PG: Ensemble list
            f"optimizer\t{optimizer.__name__}\n" +
            f"loss_region\t{loss_region}\n" +
            f"subset_dimensions\t{subset_dimensions}\n" +
            f"lead_time_mask\t{params['lead_time_mask']}\n" +
            f"model_patterns\t{model_patterns.__name__}\n" +
            f"reg_scale\t{reg_scale}\n" +  ## PG: The scale to be passed to Signloss regularization
            f"hidden_dims\t{hidden_dims}\n" +
            f"time_features_patterns\t{time_features_patterns}\n" +
            f"append_mode_patterns\t{params['append_mode_patterns']}\n" +
            f"hyperparam\t{hyperparam}\n" + ## PG: Ensemble features
            f"epochs_patterns\t{epochs_patterns}\n" +
            f"batch_size_patterns\t{batch_size_patterns}\n" +
            f"batch_normalization\t{batch_normalization}\n" +
            f"dropout_rate\t{dropout_rate}\n" +
            f"lr_patterns\t{lr_patterns}, halved after each 10 epochs\n" +
            f"L2_reg_patterns\t{l2_reg_patterns}\n" +
            f"forecast_preprocessing_steps_patterns\t{[s[0] if forecast_preprocessing_steps_patterns is not None else None for s in forecast_preprocessing_steps_patterns]}\n" +
            f"observations_preprocessing_steps_patterns\t{[s[0] if observations_preprocessing_steps_patterns is not None else None for s in observations_preprocessing_steps_patterns]}\n" +
            f"model_mean\t{params['model_mean'].__name__}\n" +
            f"epochs_mean\t{epochs_mean}\n" +
            f"time_features_mean\t{time_features_mean}\n" +
            f"append_mode_mean\t{params['append_mode_mean']}\n" +
            f"lr_mean\t{lr_mean}, halved after each 10 epochs\n" +
            f"batch_size_mean\t{batch_size_mean}\n" +
            f"L2_reg_mean\t{l2_reg_mean}\n" +
            f"forecast_preprocessing_steps_mean\t{[s[0] if forecast_preprocessing_steps_mean is not None else None for s in forecast_preprocessing_steps_mean]}\n" +
            f"observations_preprocessing_steps_mean\t{[s[0] if observations_preprocessing_steps_mean is not None else None for s in observations_preprocessing_steps_mean]}\n" +
            f"extra_predictors_mean\t{params['extra_predictors_mean']}\n" 
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for run in range(n_runs):
        print(f"Start run {run + 1} of {n_runs}...")
        for y_idx, test_year in enumerate(test_years):
            print(f"Start run for test year {test_year}...")
        

            train_years = ds_raw_ensemble_mean.year[ds_raw_ensemble_mean.year < test_year].to_numpy()
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

            if numpy_seed is not None:
                np.random.seed(numpy_seed)
            if torch_seed is not None:
                torch.manual_seed(torch_seed)


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
                if test_year < test_years[-1]:
                    ds_test_mean = ds_mean[n_train:n_train + 1,...]
                    obs_test_mean = obs_mean[n_train:n_train + 1,...]

            ds_train_patterns = ds_patterns[:n_train,...]
            obs_train_patterns = obs_patterns[:n_train,...]

            if test_year < test_years[-1]:
                ds_test_patterns = ds_patterns[n_train:n_train + 1,...]
                obs_test_patterns = obs_patterns[n_train:n_train + 1,...]   


            weights = np.cos(np.ones_like(ds_train_patterns.lon) * (np.deg2rad(ds_train_patterns.lat.to_numpy()))[..., None])  # Moved this up
            weights = xr.DataArray(weights, dims = ds_train_patterns.dims[-2:], name = 'weights').assign_coords({'lat': ds_train_patterns.lat, 'lon' : ds_train_patterns.lon}) # Create an DataArray to pass to Spatialnanremove() 
            weights_ = weights.copy()
            
            
            ### Increase the weight over some biome :
            if Biome is not None:
                if type(Biome) == dict:
                    for ind, scale in Biome.items():
                          weights = weights + (scale-1) * weights.where(biomes == ind).fillna(0)       
                    else:
                          weights = weights + weights.where(biomes == Biome).fillna(0)
            #########################
            if params["model_mean"] is not None:

                if params["model_mean"] in [Autoencoder_mean]:
                    ds_train_mean = nanremover.sample(ds_train_mean) ## PG: flatten and sample training data at those locations
                else: 
                    ds_train_mean = ds_train_mean.fillna(0.0)

            ########################################################################

            if model_patterns in [UNet , CNN]: ## PG: If the model starts with a nn.Conv2d write back the flattened data to maps.

                ds_train_patterns = ds_train_patterns.fillna(0.0) ## PG: fill NaN values with 0.0 for training
                obs_train_patterns = obs_train_patterns.fillna(0.0) ## PG: fill NaN values with 0.0 for training

                img_dim_patterns = ds_train_patterns.shape[-2] * ds_train_patterns.shape[-1] 
                if loss_region is not None:
                    loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean_patterns, loss_region)
                
                else:
                    loss_region_indices = None
            
            else: ## PG: If you have a dense first layer keep the data flattened.
                
                ds_train_patterns = nanremover.sample(ds_train_patterns) ## PG: flatten and sample training data at those locations
                obs_train_patterns = nanremover.sample(obs_train_patterns) ## PG: flatten and sample obs data at those locations   
                weights = nanremover.sample(weights) ## PG: flatten and sample weighs at those locations
                weights_ = nanremover.sample(weights_)

                img_dim_patterns = ds_train_patterns.shape[-1] ## PG: The input dim is now the length of the flattened dimention.
                if loss_region is not None:
            
                    loss_region_indices = nanremover.extract_indices(subregions[loss_region]) ## PG: We need 1D index list rather than 2D lat:lon.

                else:
                    loss_region_indices = None
  

            weights = weights.values
            weights_ = weights_.values

            if params["model_mean"] in [Autoencoder_mean]:
                img_dim_mean = ds_train_mean.shape[-1]

            if time_features_patterns is None:

                    add_feature_dim_patterns = 0
            else:

                    add_feature_dim_patterns = len(time_features_patterns)

            if params["model_mean"] is not None:
                if time_features_mean is None:

                        add_feature_dim_mean = 0
                else:

                        add_feature_dim_mean = len(time_features_mean) 
                
                if extra_predictors is not None:
                    add_feature_dim_mean = add_feature_dim_mean + len(params['extra_predictors_mean'])


            if model_patterns == Autoencoder:
                net_patterns = model_patterns(img_dim_patterns, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim_patterns, append_mode=params['append_mode_patterns'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, device = device)
            elif model_patterns == UNet:
                net_patterns = model_patterns(n_channels_x= len(ds_train_patterns.channels)+ add_feature_dim_patterns , bilinear = params['bilinear'])
            elif model_patterns == CNN: ## PG: Combining CNN encoder with dense decoder
                net_patterns = model_patterns(len(ds_train_patterns.channels) + add_feature_dim_patterns ,hidden_dims, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size )
            
            if params["model_mean"] is not None:
                if params["model_mean"] in [Autoencoder_mean]:
                    net_mean = params["model_mean"](img_dim_mean, [360, 180, 90, 45], added_features_dim=add_feature_dim_mean, append_mode=params['append_mode_mean'], batch_normalization=batch_normalization, dropout_rate=dropout_rate)
                else:
                    net_mean = params["model_mean"]( n_channels_x=len(ds_train_mean.channels) + add_feature_dim_mean  , dense_dims= [512,128,32] )


            ################################################## pattern training ##################################################################
            net_patterns.to(device)
            optimizer = torch.optim.Adam(net_patterns.parameters(), lr=lr_patterns, weight_decay = l2_reg_patterns)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            train_set_patterns = XArrayDataset(ds_train_patterns, obs_train_patterns, mask=train_mask,lead_time_mask = params['lead_time_mask'], in_memory=True, lead_time=lead_time, time_features=time_features_patterns, aligned = True, year_max = year_max, model = model_patterns.__name__) 
            dataloader_patterns = DataLoader(train_set_patterns, batch_size=batch_size_patterns, shuffle=True)

            epoch_loss_patterns = []
            net_patterns.train()
            num_batches = len(dataloader_patterns)
            
            if reg_scale is None: ## PG: if no penalizing for negative anomalies

                    criterion_patterns = WeightedMSE(weights=weights, device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices)
            else:

                if type(reg_scale) == dict:

                    criterion_patterns = WeightedMSESignLoss(weights=weights, device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices, scale=reg_scale[test_year], min_val=0, max_val=None)

                else:
                    criterion_patterns = WeightedMSESignLoss(weights=weights, device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, min_val=0, max_val=None)

            

            for epoch in tqdm.tqdm(range(epochs_patterns)):
                batch_loss = 0  ### seed is set to a number so batches are reproducable
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

                epoch_loss_patterns.append(batch_loss / num_batches)

                if epoch<=30: ###
                    scheduler.step() ####

            
            ################################################## mean training ##################################################################
            if params["model_mean"] is not None:
                net_mean.to(device)
                optimizer = torch.optim.Adam(net_mean.parameters(), lr=lr_mean, weight_decay = l2_reg_mean)
                scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

                train_set_mean = XArrayDataset(ds_train_mean, obs_train_mean, mask=train_mask,lead_time_mask = params['lead_time_mask'],extra_predictors = extra_predictors, in_memory=True, lead_time=lead_time, time_features=time_features_mean , aligned = True, year_max = year_max, model = params['model_mean'].__name__ ) 
                dataloader_mean = DataLoader(train_set_mean, batch_size=batch_size_mean, shuffle=True)
                criterion_mean = WeightedMSE(weights=np.ones([1]), device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices)

                epoch_loss_mean = []
                net_mean.train()
                num_batches = len(dataloader_patterns)

                for epoch in tqdm.tqdm(range(epochs_mean)):
                    batch_loss = 0  ### seed is set to a number so batches are reproducable

                    for batch, (x, y) in enumerate(dataloader_mean):

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
                        adjusted_forecast_mean = net_mean(x)
                        loss_mean = criterion_mean(adjusted_forecast_mean, y, mask = m) #/ (torch.max(y) - torch.min(y))
                        batch_loss += loss_mean.item()
                        loss_mean.backward()
                        optimizer.step()

                    epoch_loss_mean.append(batch_loss / num_batches)

                    if epoch<=30: ###
                        scheduler.step() ####

            
            if params["model_mean"] is not None:
                if epochs_patterns == epochs_mean:
                    epoch_loss = np.array(epoch_loss_mean) + np.array(epoch_loss_patterns)
            else:
                epoch_loss = np.array(epoch_loss_patterns)
            # EVALUATE MODEL
            ##################################################################################################################################
            if test_year < test_years[-1]:
                ds_test_patterns = nanremover.sample(ds_test_patterns, mode = 'Eval')  ## PG: Sample the test data at the common locations
                obs_test_patterns = nanremover.sample(obs_test_patterns)

                if params["model_mean"] is not None:

                    ds_test_mean = nanremover.sample(ds_test_mean, mode = 'Eval')  ## PG: Sample the test data at the common locations
                    if params["model_mean"] in [CNN_mean]:
                        ds_test_mean = nanremover.to_map(ds_test_mean).fillna(0.0)

                if model_patterns in [UNet, CNN]:
                    ds_test_patterns = nanremover.to_map(ds_test_patterns).fillna(0.0)  ## PG: Write back to map if the model starts with a nn.Conv2D
                    obs_test_patterns = nanremover.to_map(obs_test_patterns).fillna(0.0)
            ##################################################################################################################################

                test_lead_time_list = np.arange(1, ds_test_patterns.shape[1] + 1)
                test_years_list = np.arange(1, ds_test_patterns.shape[0] + 1)  ## PG: Extract the number of years as well

                test_set_patterns = XArrayDataset(ds_test_patterns, obs_test_patterns, lead_time=None,lead_time_mask = params['lead_time_mask'], time_features=time_features_patterns,  in_memory=False, aligned = True, year_max = year_max, model = model_patterns.__name__)
                criterion_test_patterns =  WeightedMSE(weights=weights_, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
                test_results_patterns = np.zeros_like(obs_test_patterns)


                if 'ensembles' in ds_test_patterns.dims:
                    test_loss_patterns = np.zeros(shape=(ds_test_patterns.shape[0], ds_test_patterns.shape[1], ds_test_patterns.shape[2]))
                else:
                    test_loss_patterns = np.zeros(shape=(ds_test_patterns.shape[0], ds_test_patterns.shape[1]))

                if params["model_mean"] is not None:
                    test_set_mean = XArrayDataset(ds_test_mean, obs_test_mean, extra_predictors = extra_predictors, lead_time=None,lead_time_mask = params['lead_time_mask'], time_features=time_features_mean ,  in_memory=False, aligned = True, year_max = year_max, model = params['model_mean'].__name__)
                    criterion_test_mean =  WeightedMSE(weights=np.ones([1]), device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
                    test_results_mean = np.zeros_like(obs_test_mean)
                    test_loss_mean = test_loss_patterns.copy()

                for i, (x, target) in enumerate(test_set_patterns): 
                    if 'ensembles' in ds_test_patterns.dims:  ## PG: If we have large ensembles:
    
                        ensemble_idx, j = np.divmod(i, len(test_lead_time_list) * len(test_years_list))  ## PG: find out ensemble index
                        year_idx, lead_time_list_idx = np.divmod(j, len(test_lead_time_list)) 
                        lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
                        net_patterns.eval()
                        with torch.no_grad():
                            if (type(x) == list) or (type(x) == tuple):
                                test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                            else:
                                test_raw = x.unsqueeze(0).to(device)
                            if (type(target) == list) or (type(target) == tuple):
                                test_obs, m = (target[0].unsqueeze(0).to(device), target[1].unsqueeze(0).to(device))
                            else:
                                test_obs = target.unsqueeze(0).to(device)
                                m = None
                            test_adjusted_patterns = net_patterns(test_raw)
                            loss_patterns = criterion_test_patterns(test_adjusted_patterns, test_obs)
                            test_results_patterns[year_idx, lead_time_idx,ensemble_idx,] = test_adjusted_patterns.to(torch.device('cpu')).numpy()  ## PG: write back to test_results
                            test_loss_patterns[year_idx, lead_time_idx,ensemble_idx] = loss_patterns.item()

                    else:

                        year_idx, lead_time_list_idx = np.divmod(i, len(test_lead_time_list))
                        lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
                        net_patterns.eval()
                        with torch.no_grad():
                            if (type(x) == list) or (type(x) == tuple):
                                test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                            else:
                                test_raw = x.unsqueeze(0).to(device)
                            if (type(target) == list) or (type(target) == tuple):
                                test_obs, m = (target[0].unsqueeze(0).to(device), target[1].unsqueeze(0).to(device))
                            else:
                                test_obs = target.unsqueeze(0).to(device)
                                m = None
                            test_adjusted_patterns = net_patterns(test_raw)
                            loss_patterns = criterion_test_patterns(test_adjusted_patterns, test_obs) 
                            test_results_patterns[year_idx, lead_time_idx,] = test_adjusted_patterns.to(torch.device('cpu')).numpy()
                            test_loss_patterns[year_idx, lead_time_idx] = loss_patterns.item()
                
                if params["model_mean"] is not None:             

                    for i, (x, target) in enumerate(test_set_mean): 
                        if 'ensembles' in ds_test_mean.dims:  ## PG: If we have large ensembles:
        
                            ensemble_idx, j = np.divmod(i, len(test_lead_time_list) * len(test_years_list))  ## PG: find out ensemble index
                            year_idx, lead_time_list_idx = np.divmod(j, len(test_lead_time_list)) 
                            lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
                            net_mean.eval()
                            with torch.no_grad():
                                if (type(x) == list) or (type(x) == tuple):
                                    test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                                else:
                                    test_raw = x.unsqueeze(0).to(device)
                                if (type(target) == list) or (type(target) == tuple):
                                    test_obs, m = (target[0].unsqueeze(0).to(device), target[1].unsqueeze(0).to(device))
                                else:
                                    test_obs = target.unsqueeze(0).to(device)
                                    m = None
                                test_adjusted_mean = net_mean(test_raw)
                                loss_mean = criterion_test_mean(test_adjusted_mean, test_obs)
                                test_results_mean[year_idx, lead_time_idx,ensemble_idx,] = test_adjusted_mean.to(torch.device('cpu')).numpy()  ## PG: write back to test_results
                                test_loss_mean[year_idx, lead_time_idx,ensemble_idx] = loss_mean.item()

                        else:

                            year_idx, lead_time_list_idx = np.divmod(i, len(test_lead_time_list))
                            lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
                            net_mean.eval()
                            with torch.no_grad():
                                if (type(x) == list) or (type(x) == tuple):
                                    test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                                else:
                                    test_raw = x.unsqueeze(0).to(device)
                                if (type(target) == list) or (type(target) == tuple):
                                    test_obs, m = (target[0].unsqueeze(0).to(device), target[1].unsqueeze(0).to(device))
                                else:
                                    test_obs = target.unsqueeze(0).to(device)
                                    m = None
                                test_adjusted_mean = net_mean(test_raw)
                                loss_mean = criterion_test_mean(test_adjusted_mean, test_obs)
                                test_results_mean[year_idx, lead_time_idx,] = test_adjusted_mean.to(torch.device('cpu')).numpy()
                                test_loss_mean[year_idx, lead_time_idx] = loss_mean.item()
                    
                    if epochs_patterns == epochs_mean:
                        test_loss = np.array(test_loss_mean) / (np.max(test_loss_mean) - np.min(test_loss_mean))  + np.array(test_loss_patterns) / (np.max(test_loss_patterns) - np.min(test_loss_patterns))
                    else:
                        test_loss_mean = np.array(test_loss_mean) / (np.max(test_loss_mean) - np.min(test_loss_mean))
                        test_loss_patterns = np.array(test_loss_patterns) / (np.max(test_loss_patterns) - np.min(test_loss_patterns))
                else:
                    test_loss = np.array(test_loss_patterns) / (np.max(test_loss_patterns) - np.min(test_loss_patterns))
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
                    results_mean = xr.DataArray(test_results_untransformed_mean, obs_test_mean.coords, obs_test_mean.dims, name='nn_adjusted').rename({'month':'lead_time'})
                    result = result  + results_mean[...,0]

                # print(result)
                ##################################################################################################################################
                # Store results as NetCDF            
                result.to_netcdf(path=Path(results_dir, f'nn_adjusted_{test_year}_{run+1}.nc', mode='w'))
                if epochs_patterns == epochs_mean:
                    fig, ax = plt.subplots(1,1, figsize=(8,5))
                    ax.plot(np.arange(1,epochs_patterns+1), epoch_loss)
                    ax.set_title(f'Train Loss \n test loss: {np.mean(test_loss)}') ###
                    
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss (nMSE)')
                    plt.show()
                    plt.savefig(results_dir+f'/Figures/train_loss_1982-{test_year-1}.png')
                    plt.close()
                else:
                    Path(results_dir+f'/Figures/Mean').mkdir(parents=True, exist_ok=True)
                    
                    fig, ax = plt.subplots(1,1, figsize=(8,5))
                    ax.plot(np.arange(1,epochs_mean+1), np.array(epoch_loss_mean))
                    ax.set_title(f'Train Loss \n test loss: {np.mean(test_loss_mean)}') ###
                    
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss (nMSE)')
                    plt.show()
                    plt.savefig(results_dir+f'/Figures/Mean/train_loss_1982-{test_year-1}.png')
                    plt.close()

                    Path(results_dir+f'/Figures/Patterns').mkdir(parents=True, exist_ok=True)

                    fig, ax = plt.subplots(1,1, figsize=(8,5))
                    ax.plot(np.arange(1,epochs_patterns+1), np.array(epoch_loss_patterns))
                    ax.set_title(f'Train Loss \n test loss: {np.mean(test_loss_patterns)}') ###
                    
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss (nMSE)')
                    plt.show()
                    plt.savefig(results_dir+f'/Figures/Patterns/train_loss_1982-{test_year-1}.png')
                    plt.close()

                if save:

                        if params["model_mean"] is not None:
                            nameSave = f'MODEL_mean_V{params["version_patterns"]}{params["version_mean"]}_1982-{test_year-1}.pth'
                            torch.save( net_mean.state_dict(),results_dir + '/' + nameSave)

                        nameSave = f'MODEL_patterns_V{params["version_patterns"]}{params["version_mean"]}_1982-{test_year-1}.pth'
                        torch.save( net_patterns.state_dict(),results_dir + '/' + nameSave)
                    
            else:
                if params["model_mean"] is not None:
                    nameSave = f'MODEL_final_mean_V{params["version_patterns"]}{params["version_mean"]}_1982-{test_years[-2]}.pth'
                    torch.save( net_mean.state_dict(),results_dir + '/' + nameSave)

                nameSave = f'MODEL_final_patterns_V{params["version_patterns"]}{params["version_mean"]}_1982-{test_years[-2]}.pth'
                torch.save( net_patterns.state_dict(),results_dir + '/' + nameSave)

if __name__ == "__main__":
   
    n_years =  35 # last n years to test consecutively
    lead_years = 10
    n_runs = 1  # number of training runs

    params = {
        'ensemble_list' : None, ## PG
        'ensemble_mode' : 'Mean',
        "optimizer": torch.optim.Adam,
        "loss_region": None,
        "subset_dimensions": None,
        "lead_time_mask" : 5,
        "model_patterns": Autoencoder,
        "hidden_dims": [[3600, 1800, 900], [1800, 3600]],
        "time_features_patterns":  ['month_sin', 'month_cos'],
        "epochs_patterns": 25,
        "batch_size_patterns": 64,   
        "lr_patterns": 0.001 ,
        "batch_normalization": False,
        "dropout_rate": 0,
        "append_mode_patterns": 1,
        "hyperparam"  :1, 
        "reg_scale" : None,
        "L2_reg_patterns" : 0.001,
        "model_mean": Autoencoder_mean,
        "time_features_mean":  ['month_sin', 'month_cos','year'],
        "extra_predictors_mean" : [],
        "append_mode_mean": 1,
        "epochs_mean": 120,
        "batch_size_mean": 64,
        "lr_mean": 0.005,
        "L2_reg_mean" : 0 

    }



    params['ensemble_list'] = None #[f'r{e}i1p2f1' for e in range(1,21,1)] ## PG
    params['ensemble_mode'] = None
 
    
    params['reg_scale'] = None
    params['arch'] = 2
    params['version_patterns'] = 3 ### 0,1 , 2 ,3
    params['version_mean'] = 3  ### 1 , 2 ,3

    

    biomes = xr.open_dataset('/home/rpg002/fgco2_decadal_forecast_adjustment/Time_Varying_Biomes.nc').MeanBiomes.transpose()
    params['Biome'] = None
    if params["model_mean"] == None :  params['version_mean'] = ""

    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/fgco2_ems/SOM-FFN/results/{params["model_patterns"].__name__}/run_set_5'

    
    out_dir    = f'{out_dir_x}/N{n_years}_v{params["version_patterns"]}{params["version_mean"]}_decoupled_L{params["reg_scale"]}_arch{params["arch"]}_batch{params["batch_size_patterns"]}-{params["batch_size_mean"]}_e{params["epochs_patterns"]}-{params["epochs_mean"]}_lead_time_mask_5' 
    if len(params["extra_predictors_mean"]) > 0:
        out_dir = out_dir + '_extra_predictors'
    if params["model_mean"] == CNN_mean:
        out_dir = out_dir + '_CNN_mean'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir + '/Figures').mkdir(parents=True, exist_ok=True)

    run_training(params, n_years=n_years, lead_years=lead_years, n_runs=n_runs, results_dir=out_dir, numpy_seed=1, torch_seed=1)
    print(f'Output dir: {out_dir}')
    print('Training done.')

    # for ens in range(1):

    #     print( f'Training {ens}')
    
    #     out_dir    = f'{out_dir_x}/N{n_years}_v{params["version_patterns"]}{params["version_mean"]}_decoupled_L{params["reg_scale"]}_arch{params["arch"]}_batch{params["batch_size_patterns"]}-{params["batch_size_mean"]}_e{params["epochs_patterns"]}-{params["epochs_mean"]}_LE' 
    #     if len(params["extra_predictors_mean"]) > 0:
    #         out_dir = out_dir + f'_extra_predictors'
    #     if params["model_mean"] == CNN_mean:
    #         out_dir = out_dir + f'_CNN_mean'
        
    
    #     Path(out_dir+ f'/E{ens}').mkdir(parents=True, exist_ok=True)
    #     Path(out_dir+ f'/E{ens}' + '/Figures').mkdir(parents=True, exist_ok=True)

    #     run_training(params, n_years=n_years, lead_years=lead_years, n_runs=n_runs, results_dir=out_dir+ f'/E{ens}', numpy_seed=ens, torch_seed=ens)
    #     print(f'Output dir: {out_dir}'+ f'/E{ens}')
    #     print('Training done.')