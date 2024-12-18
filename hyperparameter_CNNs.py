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
from models.autoencoder import Autoencoder
from models.unet2 import UNet2, UNet2_decoupled
from models.cnn import CNN
from losses import WeightedMSE, WeightedMSESignLoss, GlobalLoss, CorrLoss
from preprocessing import align_data_and_targets, create_train_mask, Spatialnanremove, reshape_obs_to_data, get_coordinate_indices
from data_utils.datahandling import combine_observations
from preprocessing import AnomaliesScaler_v1,AnomaliesScaler_v2, Standardizer, PreprocessingPipeline, calculate_climatology, config_grid
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

    if  params["obs_clim"]:
            
            ls = []
            for yr in ds_raw_ensemble_mean.year.values[3:]:
                    
                    ref  = obs_raw.where(obs_raw.year < yr, drop = True)
                    mask = create_train_mask(ref[:-1])
                    mask = np.broadcast_to(mask[...,None,None,None], ref[:-1].shape)
                    ls.append(calculate_climatology(ref[:-1],mask ).expand_dims('year', axis = 0).assign_coords(year = ref[-1:].year))
            clim = xr.concat(ls, dim = 'year')
            if 'ensembles' in ds_raw_ensemble_mean.dims: 
                clim = clim.expand_dims(ensembles = ds_raw_ensemble_mean['ensembles'], axis = 2) ########
            obs_raw = obs_raw.sel(year = clim.year)
            ds_raw_ensemble_mean = ds_raw_ensemble_mean.sel(year = clim.year)
            ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, clim], dim = 'channels')
    
    return  ds_raw_ensemble_mean, obs_raw, params

def smooth_curve(list, factor = 0.9):
    smoothed_list = []
    for point in list:
        if smoothed_list:
            previous = smoothed_list[-1]
            smoothed_list.append(previous* factor + point * (1- factor))
        else:
            smoothed_list.append(point)
    return smoothed_list



def training_hp(hyperparamater_grid: dict, params:dict, ds_raw_ensemble_mean: XArrayDataset ,obs_raw: XArrayDataset , test_year, n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None):
    assert params['version'] in [1,2,3]
    if params['version'] == 1:
            params['forecast_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v1(axis=0)), 
            ('standardize', Standardizer())]

            params['observations_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v1(axis=0))  ]
    elif params['version'] == 2:
            params['forecast_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v2(axis=0)), 
            ('standardize', Standardizer())]

            params['observations_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v2(axis=0))  ]

    else:
            params['forecast_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v1(axis=0)), 
            ('standardize', Standardizer()) ]

            params['observations_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v2(axis=0))  ]

    for key, value in hyperparamater_grid.items():
            params[key] = value 



    ##### PG: Ensemble members to load 
    ensemble_list = params['ensemble_list']
    ###### PG: Add ensemble features to training features
    ensemble_mode = params['ensemble_mode'] ##
    ensemble_features = params['ensemble_features']
    
    if 'atm_co2' in params['extra_predictors']:
        atm_co2 = xr.open_dataset('/home/rpg002/CMIP6_ssp245_xCO2atm_1982_2029.nc').ssp245
        atm_co2 = reshape_obs_to_data(atm_co2, ds_raw_ensemble_mean, return_xarray=True).rename({'month' : 'lead_time'})
        try:
            extra_predictors = atm_co2.sel(year = ds_raw_ensemble_mean.year).expand_dims('channels', axis=2)
        except:
            raise ValueError("Extra predictors not available at the same years as the predictors.")
    else:
         extra_predictors = None     

    if extra_predictors is not None:
        if params["model"] in [Autoencoder]:
            weights = np.cos(np.ones_like(extra_predictors.lon) * (np.deg2rad(extra_predictors.lat.to_numpy()))[..., None])  # Moved this up
            weights = xr.DataArray(weights, dims = extra_predictors.dims[-2:], name = 'weights').assign_coords({'lat': extra_predictors.lat, 'lon' : extra_predictors.lon}) 
            extra_predictors = (extra_predictors * weights).sum(['lat','lon'])/weights.sum(['lat','lon'])
        else:  
            if not all(['ensembles' not in extra_predictors.dims, 'ensembles' in ds_raw_ensemble_mean.dims]): 
                
                  ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, extra_predictors], dim = 'channels')
            else:
                ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, extra_predictors.expand_dims(ensembles = ds_raw_ensemble_mean['ensembles'], axis = 2) ], dim = 'channels')
            extra_predictors = None
 

    reg_scale = params["reg_scale"]
    model = params["model"]
    hidden_dims = params["hidden_dims"]
    time_features = params["time_features"].copy() if params["time_features"] is not None else None
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    kernel_size = params["kernel_size"]
    decoder_kernel_size = params["decoder_kernel_size"]
    optimizer = params["optimizer"]
    lr = params["lr"]

    interp_nan = params['interp_nan']
    if model == UNet2_decoupled:
        interp_nan = False
    land_mask = params['land_mask']
    if not interp_nan:
        land_mask = False

    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]

    loss_region = params["loss_region"]
    subset_dimensions = params["subset_dimensions"]
    Biome = params['Biome']
    num_val_years = params['num_val_years']


    test_years = test_year

    if n_runs > 1:
        numpy_seed = None
        torch_seed = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
    print(f"Start run for test year {test_year}...")

    train_years = ds_raw_ensemble_mean.year[ds_raw_ensemble_mean.year < test_year-num_val_years].to_numpy()
    # validation_years = ds_raw_ensemble_mean.year[(ds_raw_ensemble_mean.year >= test_year-3)&(ds_raw_ensemble_mean.year < test_year)].to_numpy()

    n_train = len(train_years)
    train_mask = create_train_mask(ds_raw_ensemble_mean[:n_train,...])

    ds_baseline = ds_raw_ensemble_mean[:n_train,...]
    obs_baseline = obs_raw[:n_train,...]
    train_mask = create_train_mask(ds_baseline)

    if 'ensembles' in ds_raw_ensemble_mean.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline.shape)
    else:
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline.shape)
    preprocessing_mask_obs = np.broadcast_to(train_mask[...,None,None,None], obs_baseline.shape)

    if subset_dimensions is not None:
        xmin, xmax, ymin, ymax = get_coordinate_indices(ds_raw_ensemble_mean, subset_dimensions)
        ds = ds_raw_ensemble_mean[..., xmin:xmax+1, ymin:ymax+1]
        obs = obs_raw[..., xmin:xmax+1, ymin:ymax+1]

    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    # Data preprocessing
    
    ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_baseline, mask=preprocessing_mask_fct)
    ds = ds_pipeline.transform(ds_raw_ensemble_mean)
    

    obs_pipeline = PreprocessingPipeline(observations_preprocessing_steps).fit(obs_baseline, mask=preprocessing_mask_obs)
    if 'standardize' in ds_pipeline.steps:
        obs_pipeline.add_fitted_preprocessor(ds_pipeline.get_preprocessors('standardize'), 'standardize')
    obs = obs_pipeline.transform(obs_raw)

    year_max = ds[:n_train + num_val_years ].year[-1].values
    # TRAIN MODEL

    lead_time = None
    ds_train = ds[:n_train,...]
    obs_train = obs[:n_train,...]
    ds_validation = ds[n_train:n_train + num_val_years,...]
    obs_validation = obs[n_train:n_train + num_val_years,...]

    ##### PG: The ocean carbon flux has NaN values over land in both forecast and obs data and these are not necessarily in the excat same grid points. ###
    ##### We need to extract the common grid points where both obs and model data exist. That said, we need to flatten both the training and target data
    ##### I defined a Nanremover class. See preprocessing.py.
        
    weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
    weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove() 
    weights_val = weights.copy()

    #### Set weights to 1!!:
    # weights = xr.ones_like(weights)
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
    nanremover = Spatialnanremove() ## PG: Get an instance of the class
    nanremover.fit(ds_train[:,:12,...], obs_train[:,:12,...]) ## PG:extract the commong grid points between training and obs data
    ds_train = nanremover.sample(ds_train) ## PG: flatten and sample training data at those locations
    obs_train = nanremover.sample(obs_train) ## PG: flatten and sample obs data at those locations
    ########################################################################

    if model in [UNet2,UNet2_decoupled , CNN]: ## PG: If the model starts with a nn.Conv2d write back the flattened data to maps.                
        if interp_nan is not None:

                ds_train = nanremover.to_map(ds_train).interpolate_na(dim =interp_nan, method = 'linear' ) ## PG: fill NaN values with 0.0 for training
                obs_train = nanremover.to_map(obs_train).interpolate_na(dim =interp_nan, method = 'linear') ## PG: fill NaN values with 0.0 for training

        else:

                ds_train = nanremover.to_map(ds_train).fillna(0.0) ## PG: fill NaN values with 0.0 for training
                obs_train = nanremover.to_map(obs_train).fillna(0.0) ## PG: fill NaN values with 0.0 for training
                W = nanremover.sample(weights)
                W = nanremover.to_map(W).fillna(0.0)


        img_dim = ds_train.shape[-2] * ds_train.shape[-1] 
        if loss_region is not None:
            loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region)
        
        else:
            loss_region_indices = None
    
    else: ## PG: If you have a dense first layer keep the data flattened.

        weights = nanremover.sample(weights) ## PG: flatten and sample weighs at those locations
        weights_val = nanremover.sample(weights_val)
        img_dim = ds_train.shape[-1] ## PG: The input dim is now the length of the flattened dimention.
        if loss_region is not None:
    
            loss_region_indices = nanremover.extract_indices(subregions[loss_region]) ## PG: We need 1D index list rather than 2D lat:lon.

        else:
            loss_region_indices = None

    weights = weights.values
    weights_val = weights_val.values


    if time_features is None:
        if ensemble_features: ## PG: We can choose to add an ensemble feature.
            add_feature_dim = 1
        else:
            add_feature_dim = 0
    else:
        if ensemble_features:
            add_feature_dim = len(time_features) + 1
        else:
            add_feature_dim = len(time_features)
    if extra_predictors is not None:
            add_feature_dim = add_feature_dim + len(params['extra_predictors'])

    if params['obs_clim']:
        n_channels_x = len(ds_train.channels) + 1
    else:
        n_channels_x = len(ds_train.channels)


    if model == Autoencoder:
        net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, device = device)
    elif model == UNet2:
        net = model(n_channels_x= n_channels_x+ add_feature_dim , bilinear = params['bilinear'])
    elif model == CNN: ## PG: Combining CNN encoder with dense decoder
        net = model(n_channels_x + add_feature_dim ,hidden_dims, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size )
    elif model == UNet2_decoupled:
            net = model(n_channels_x= n_channels_x+ add_feature_dim , bilinear = params['bilinear'])

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    ## PG: XArrayDataset now needs to know if we are adding ensemble features. The outputs are datasets that are maps or flattened in space depending on the model.
    train_set = XArrayDataset(ds_train, obs_train, mask=train_mask, lead_time_mask = params['lead_time_mask'], extra_predictors=extra_predictors, in_memory=True, lead_time=lead_time, time_features=time_features,ensemble_features =ensemble_features, aligned = True, year_max = year_max, model = model.__name__) 
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    if reg_scale is None: ## PG: if no penalizing for negative anomalies
        criterion = WeightedMSE(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
    else:
        criterion = WeightedMSESignLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, min_val=0, max_val=None)
    if model == UNet2_decoupled:
            criterion_global = WeightedMSE(weights=np.ones([1]), device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
            W = torch.from_numpy(W.values)
    # EVALUATE MODEL
    ##################################################################################################################################
    ds_validation = nanremover.sample(ds_validation, mode = 'Eval')  ## PG: Sample the test data at the common locations
    obs_validation = nanremover.sample(obs_validation)
    if model in [UNet2,UNet2_decoupled, CNN]:
        if interp_nan is not None:
            ds_validation = nanremover.to_map(ds_validation).interpolate_na(dim =interp_nan, method = 'linear' )  ## PG: Write back to map if the model starts with a nn.Conv2D
            obs_validation = nanremover.to_map(obs_validation).interpolate_na(dim =interp_nan, method = 'linear' )
        else:
            ds_validation = nanremover.to_map(ds_validation).fillna(0.0)  ## PG: Write back to map if the model starts with a nn.Conv2D
            obs_validation = nanremover.to_map(obs_validation).fillna(0.0)

    val_mask = create_train_mask(ds_validation)
    validation_set = XArrayDataset(ds_validation, obs_validation, mask=val_mask,extra_predictors=extra_predictors, lead_time=None, lead_time_mask = params['lead_time_mask'], time_features=time_features,ensemble_features =ensemble_features,  in_memory=False, aligned = True, year_max=year_max, model = model.__name__) 
    dataloader_val = DataLoader(validation_set, batch_size=batch_size, shuffle=True)   

    
    if reg_scale is None: ## PG: if no penalizing for negative anomalies
        criterion_eval = WeightedMSE(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
    else:
        criterion_eval = WeightedMSESignLoss(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, min_val=0, max_val=None)

    criterion_eval_global = GlobalLoss(device=device, scale=1, weights=weights_val, loss_area=None, map = True)
    criterion_eval_mean = criterion_eval =  WeightedMSE(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
    ##################################################################################################################################
    
    epoch_loss_train = []
    epoch_loss_val = []
    epoch_loss_val_global = []
    epoch_loss_val_mean = []

    for epoch in tqdm.tqdm(range(epochs)):
        net.train()
        batch_loss = 0
        
        for batch, (x, y) in enumerate(dataloader):
            
            if (type(x) == list) or (type(x) == tuple):
                x = (x[0].to(device), x[1].to(device))
            else:
                x = x.to(device)
            if (type(y) == list) or (type(y) == tuple):
                y, m = (y[0], y[1].to(device))
            else:
                y = y
                m  = None   
            if model == UNet2_decoupled:
                y_1 = (y * W).sum(dim = (-1,-2))/ W.sum(dim = (-1,-2))
                try:
                    m_1 = m.mean(dim = (-1,-2)).unsqueeze(-1)
                except:
                    m_1 = None
                y_0 = y - y_1.unsqueeze(-1).unsqueeze(-1).expand_as(y)
                y_0[y == 0] = 0
                y = (y_0.to(device), y_1.unsqueeze(-1).to(device))

                optimizer.zero_grad()
                adjusted_forecast_pattern, adjusted_forecast_mean = net(x)
                loss = criterion(adjusted_forecast_pattern, y[0],mask = m)/ (torch.max(y[0]) - torch.min(y[0])) + criterion_global(adjusted_forecast_mean, y[1], mask = m_1)/ (torch.max(y[1]) - torch.min(y[1]))
            else:
                y = y.to(device)
                optimizer.zero_grad()
                adjusted_forecast = net(x)
                loss = criterion(adjusted_forecast, y, mask = m)
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        epoch_loss_train.append(batch_loss / len(dataloader))

        if epoch<=30: ###
            scheduler.step() ####
        net.eval()
        val_loss = 0
        val_loss_global = 0
        val_loss_mean = 0
        
        for batch, (x, target) in enumerate(dataloader_val):         
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
                    
                if model == UNet2_decoupled:
                    test_adjusted_pattern, test_adjusted_mean = net(test_raw)
                    test_adjusted = test_adjusted_pattern + test_adjusted_mean.unsqueeze(-1).expand_as(test_adjusted_pattern)
                else:
                    test_adjusted = net(test_raw)
                loss_ = criterion_eval(test_adjusted, test_obs, mask = m)
                val_loss += loss_.item()
                loss_global = criterion_eval_global(test_adjusted, test_obs, mask = m)
                val_loss_global += loss_global.item()
                loss_mean = criterion_eval_mean(test_adjusted, test_obs, mask = m)
                val_loss_mean += loss_mean.item()


        epoch_loss_val.append(val_loss / len(dataloader_val))
        epoch_loss_val_global.append(val_loss_global / len(validation_set))
        epoch_loss_val_mean.append(val_loss_mean / len(dataloader_val))
        # Store results as NetCDF            

    epoch_loss_val = smooth_curve(epoch_loss_val)
    epoch_loss_val_global = smooth_curve(epoch_loss_val_global)
    epoch_loss_val_mean = smooth_curve(epoch_loss_val_mean)

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(2,epochs+1), epoch_loss_train[1:], label = 'Train')
    plt.plot(np.arange(2,epochs+1), epoch_loss_val[1:], label = 'Validation')
    plt.title(f'{hyperparamater_grid}')
    plt.legend()
    plt.ylabel('MSE')
    plt.twinx()
    plt.plot(np.arange(2,epochs+1), epoch_loss_val_mean[1:], label = 'Validation MSE',  color = 'k', alpha = 0.5)
    plt.ylabel('MSE')
    plt.legend()
    plt.xlabel('Epoch')
    
    plt.grid()
    plt.show()
    plt.savefig(results_dir+f'/val-train_loss_1982-{test_year}-{hyperparamater_grid}.png')
    plt.close()


    with open(Path(results_dir, "Hyperparameter_training.txt"), 'a') as f:
        f.write(
  
            f"{hyperparamater_grid} ---> val_loss at best epochs: {min(epoch_loss_val)} at {np.argmin(epoch_loss_val)+1}  (MSE : {epoch_loss_val_mean[np.argmin(epoch_loss_val)]})\n" +  ## PG: The scale to be passed to Signloss regularization
            f"-------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n" 
        )
    return epoch_loss_val_mean[np.argmin(epoch_loss_val)], epoch_loss_val, epoch_loss_val_global, epoch_loss_val_mean, epoch_loss_train

                                 #########         ##########

    
def run_hp_tunning( ds_raw_ensemble_mean: XArrayDataset ,obs_raw: XArrayDataset,  hyperparameterspace: list, params:dict, y_start: int, out_dir_x , n_runs=1, numpy_seed=None, torch_seed=None ):

    y_end = ds_raw_ensemble_mean.year[-1].values +1 

    val_losses = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
    val_losses_global = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
    val_losses_mean = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
    train_losses = np.zeros([y_end - y_start + 1, len(hyperparameterspace), params['epochs']]) ####

    for ind_, test_year in enumerate(range(y_start,y_end+1)):
    
        if len(params["extra_predictors"]) > 0:
            out_dir_x = out_dir_x + '_extra_predictors'

        out_dir    = f'{out_dir_x}/_{test_year}'    
        
        
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        with open(Path(out_dir, "Hyperparameter_training.txt"), 'w') as f:
            f.write(
                f"model\t{params['model']}\n" +
                "default set-up:\n" + 
                f"hidden_dims\t{params['hidden_dims']}\n" +
                f"time_features\t{params['time_features']}\n" +
                f"extra_predictors\t{params['extra_predictors']}\n" +
                f"obs_clim\t{params['obs_clim']}\n" +
                f"ensemble_list\t{params['ensemble_list']}\n" + ## PG: Ensemble list
                f"ensemble_features\t{params['ensemble_features']}\n" + ## PG: Ensemble features
                f"lr\t{params['lr']}\n" +
                f"kernel size\t{params['kernel_size']}\n" +
                f"decoder kernel size\t{params['decoder_kernel_size']}\n" +
                f"Lead time mask\t{params['lead_time_mask']}\n\n\n" +
                ' ----------------------------------------------------------------------------------\n'
            )
        

        
        losses = np.zeros(len(hyperparameterspace))
 

        
        for ind, dic in enumerate(hyperparameterspace):
            print(f'Training for {dic}')
            losses[ind], val_losses[ind_, ind, :], val_losses_global[ind_, ind, :], val_losses_mean[ind_, ind, :],  train_losses[ind_, ind, :] = training_hp( ds_raw_ensemble_mean =  ds_raw_ensemble_mean,obs_raw = obs_raw , hyperparamater_grid= dic, params = params ,test_year=test_year, n_runs=n_runs, results_dir=out_dir, numpy_seed=numpy_seed, torch_seed=torch_seed)

        
        with open(Path(out_dir, "Hyperparameter_training.txt"), 'a') as f:
            f.write(
    
                f"Best MSE: {min(losses)} --> {hyperparameterspace[np.argmin(losses)]} \n" +  ## PG: The scale to be passed to Signloss regularization
                f"--------------------------------------------------------------------------------------------------------\n" 
            )

        print(f"Best loss: {min(losses)} --> {hyperparameterspace[np.argmin(losses)]}")
        print(f'Output dir: {out_dir}')
        print('Training done.')

    coords = []
    for item in hyperparameterspace:
        coords.append(str(tuple(item.values())))

    ds_val = xr.DataArray(val_losses, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
    ds_val.attrs['hyperparameters'] = list(config_dict.keys())
    ds_val.to_netcdf(out_dir_x + '/validation_losses.nc')

    ds_val_global = xr.DataArray(val_losses_global, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
    ds_val_global.attrs['hyperparameters'] = list(config_dict.keys())
    ds_val_global.to_netcdf(out_dir_x + '/validation_losses_global.nc')

    ds_val_mean = xr.DataArray(val_losses_mean, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
    ds_val_mean.attrs['hyperparameters'] = list(config_dict.keys())
    ds_val_mean.to_netcdf(out_dir_x + '/validation_losses_mean.nc')

    ds_train = xr.DataArray(train_losses, dims = ['test_years', 'hyperparameters','epochs'], name = 'Train_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1), 'hyperparameters': coords})
    ds_train.attrs['hyperparameters'] = list(config_dict.keys())
    ds_train.to_netcdf(out_dir_x + '/train_losses.nc')

    

    

if __name__ == "__main__":

 
    lead_years = 5
    n_runs = 1  # number of training runs

    params = {
        "model": UNet2,
        "hidden_dims": [16, 64, 128, 64, 32],
        "time_features": ['month_sin', 'month_cos'],
        "extra_predictors" : [],
        "obs_clim" : False,
        "ensemble_features": False, ## PG
        'ensemble_list' : None, ## PG
        'ensemble_mode' : 'Mean',
        "epochs": 75,
        "batch_size": 100,
        "reg_scale" : None,
        "optimizer": torch.optim.Adam,
        "lr": 0.001,
        "loss_region": None,
        "subset_dimensions": None,
        "kernel_size" : 5,
        "decoder_kernel_size" : 5,
        "bilinear" : False, ## only for UNet,
        "lead_time_mask" : None,
    }


    if params["model"] != Autoencoder:
        params["append_mode"] = None

    if params['model'] in [UNet2, UNet2_decoupled]:
        params['kernel_size'] = None
        params['decoder_kernel_size'] = None
        params['hidden_dims'] = None



    params['ensemble_list'] = None #[f'r{e}i1p2f1' for e in range(1,21,1)] ## PG
    params['ensemble_features'] = False ## PG

    biomes = xr.open_dataset('/home/rpg002/fgco2_decadal_forecast_adjustment/Time_Varying_Biomes.nc').MeanBiomes.transpose()
    params['Biome'] = None

    #######################################  Don't touch the following #######################################################

    ds_raw_ensemble_mean, obs_raw, params = HP_congif(params, data_dir_obs, lead_years)

    ################################################ specifics #################################################################

    y_start = 2006
  
    params['num_val_years'] = 5

    config_dict = {'batch_size' : [8,16, 64], 'reg_scale':[None,1,3]}
    hyperparameterspace = config_grid(config_dict).full_grid()

    params['interp_nan'] = None #'lat' or 'lon'


    params['version'] = 3  ## 1,2,3

    ##############################################################################################################################

    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/fgco2_ems/SOM-FFN/results/{params["model"].__name__}/run_set_1/Model_tunning/batch_reg_tunning_adj_lr_v{params["version"]}'

    run_hp_tunning(ds_raw_ensemble_mean = ds_raw_ensemble_mean ,obs_raw = obs_raw,  hyperparameterspace = hyperparameterspace, params = params, y_start = y_start, out_dir_x = out_dir_x, n_runs=1, numpy_seed=1, torch_seed=1 )