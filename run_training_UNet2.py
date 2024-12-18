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
from models.autoencoder import Autoencoder
from models.unet2 import UNet2, UNet2_decoupled
from models.cnn import CNN, SCNN
from losses import WeightedMSE, WeightedMSESignLoss
from preprocessing import align_data_and_targets, create_train_mask, Spatialnanremove, reshape_obs_to_data, get_coordinate_indices
from data_utils.datahandling import combine_observations
from preprocessing import AnomaliesScaler_v1,AnomaliesScaler_v2, Standardizer, PreprocessingPipeline, calculate_climatology
from torch_datasets import XArrayDataset
from subregions import subregions
from data_locations import LOC_FORECASTS_fgco2, LOC_OBSERVATIONS_fgco2

# specify data directories
data_dir_forecast = LOC_FORECASTS_fgco2
data_dir_obs = LOC_OBSERVATIONS_fgco2
unit_change = 60*60*24*365 * 1000 /12 * -1 ## Change units for ESM data to mol m-2 yr-1


def run_training(params, n_years, lead_years, n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None, save = False):
    
    assert params['version'] in [1,2,3]
    
    if params["model"] != Autoencoder:
        params["append_mode"] = None

    if params['model'] in [UNet2, UNet2_decoupled]:
        params['kernel_size'] = None
        params['decoder_kernel_size'] = None
        params['hidden_dims'] = None

    
    if params['version'] == 1:
        params['forecast_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v1(axis=0)), 
        ('standardize', Standardizer()) ]

        params['observations_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v1(axis=0))  ]
    elif params['version'] == 2:
        params['forecast_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v2(axis=0)), 
        ('standardize', Standardizer()) ]

        params['observations_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v2(axis=0))  ]
    else:
        params['forecast_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v1(axis=0)), 
        ('standardize', Standardizer()) ]

        params['observations_preprocessing_steps'] = [
        ('anomalies', AnomaliesScaler_v2(axis=0))  ]


    print("Start training")
    print("Load observations")
    obs_in = combine_observations(data_dir_obs, two_dim=True) # 1961.01 - 2021.12

    ##### PG: Ensemble members to load 
    ensemble_list = params['ensemble_list']
    ###### PG: Add ensemble features to training features
    ensemble_mode = params['ensemble_mode'] ##
    ensemble_features = params['ensemble_features']

    Biome = params['Biome']
    

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
        if params["model"] in [ Autoencoder]:
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
    model = params["model"]
    hidden_dims = params["hidden_dims"]
    time_features = params["time_features"]
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
    obs_clim = params["obs_clim"]

    if obs_clim:
            
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

    
    test_years = ds_raw_ensemble_mean.year[-n_years:].to_numpy()
    test_years = [*test_years,test_years[-1] + 1]

    if n_runs > 1:
        numpy_seed = None
        torch_seed = None

    with open(Path(results_dir, "training_parameters.txt"), 'w') as f:
        f.write(
            f"model\t{model.__name__}\n" +
            f"reg_scale\t{reg_scale}\n" +  ## PG: The scale to be passed to Signloss regularization
            f"hidden_dims\t{hidden_dims}\n" +
            f"time_features\t{time_features}\n" +
            f"obs_clim\t{obs_clim}\n" +
            f"extra_predictors\t{params['extra_predictors']}\n" +
            f"ensemble_list\t{ensemble_list}\n" + ## PG: Ensemble list
            f"ensemble_features\t{ensemble_features}\n" + ## PG: Ensemble features
            f"epochs\t{epochs}\n" +
            f"batch_size\t{batch_size}\n" +
            f"optimizer\t{optimizer.__name__}\n" +
            f"lr\t{0.001}, halved after each 10 epochs\n" +
            f"kernel size\t{kernel_size}\n" +
            f"decoder kernel size\t{decoder_kernel_size}\n" +
            f"forecast_preprocessing_steps\t{[s[0] if forecast_preprocessing_steps is not None else None for s in forecast_preprocessing_steps]}\n" +
            f"observations_preprocessing_steps\t{[s[0] if observations_preprocessing_steps is not None else None for s in observations_preprocessing_steps]}\n" +
            f"loss_region\t{loss_region}\n" +
            f"subset_dimensions\t{subset_dimensions}\n"+
            f"lead_time_mask\t{params['lead_time_mask']}"
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for run in range(n_runs):
        print(f"Start run {run + 1} of {n_runs}...")
        for y_idx, test_year in enumerate(test_years):
            print(f"Start run for test year {test_year}...")


            train_years = ds_raw_ensemble_mean.year[ds_raw_ensemble_mean.year < test_year].to_numpy()
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

            year_max = ds[:n_train + 1].year[-1].values 

            # TRAIN MODEL

            lead_time = None
            ds_train = ds[:n_train,...]
            obs_train = obs[:n_train,...]
            if test_year < test_years[-1]:
                ds_test = ds[n_train:n_train + 1,...]
                obs_test = obs[n_train:n_train + 1,...]


             
            weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
            weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove() 
            weights_ = weights.copy()
            
            ### Increase the weight over some biome :
            if Biome is not None:
                if type(Biome) == dict:
                    for ind, scale in Biome.items():
                          weights = weights + (scale-1) * weights.where(biomes == ind).fillna(0)       
                    else:
                          weights = weights + weights.where(biomes == Biome).fillna(0)
    
            ########################################################################

            if model in [UNet2, UNet2_decoupled , CNN]: ## PG: If the model starts with a nn.Conv2d write back the flattened data to maps.
                if interp_nan:

                    temp = ds_train.copy()
                    ds_train = temp.interpolate_na(dim ='lon', method = 'linear' ).interpolate_na(dim ='lat', method = 'linear' , fill_value="extrapolate") ## PG: fill NaN values with 0.0 for training
                    if land_mask:
                        mask_land = temp.where(~np.isnan(temp),1).fillna(0.0)
                        ds_train = xr.concat([ds_train, mask_land], dim = 'channels')
                    
                    temp = obs_train.copy()     
                    obs_train = temp.interpolate_na(dim ='lon', method = 'linear' ).interpolate_na(dim ='lat', method = 'linear' , fill_value="extrapolate")## PG: fill NaN values with 0.0 for training
                    if land_mask:
                        mask_land = temp.where(~np.isnan(temp),1).fillna(0.0)
                        obs_train = xr.concat([obs_train, mask_land], dim = 'channels')

                else:

                    ds_train = ds_train.fillna(0.0) ## PG: fill NaN values with 0.0 for training
                    obs_train = obs_train.fillna(0.0) ## PG: fill NaN values with 0.0 for training
                    W = nanremover.sample(weights)
                    W = nanremover.to_map(W).fillna(0.0)

                img_dim = ds_train.shape[-2] * ds_train.shape[-1] 

                if loss_region is not None:
                    loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region)  
                else:
                    loss_region_indices = None
            
            else: ## PG: If you have a dense first layer keep the data flattened.

                ds_train = nanremover.sample(ds_train) ## PG: flatten and sample training data at those locations
                obs_train = nanremover.sample(obs_train) ## PG: flatten and sample obs data at those locations   
                weights = nanremover.sample(weights) ## PG: flatten and sample weighs at those locations
                weights_ = nanremover.sample(weights_)

                img_dim = ds_train.shape[-1] ## PG: The input dim is now the length of the flattened dimention.
                if loss_region is not None:
            
                    loss_region_indices = nanremover.extract_indices(subregions[loss_region]) ## PG: We need 1D index list rather than 2D lat:lon.

                else:
                    loss_region_indices = None

            weights = weights.values
            weights_ = weights_.values

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

            if obs_clim:
                n_channels_x = len(ds_train.channels) + 1 
            else:
                n_channels_x = len(ds_train.channels)
            
            if land_mask:
                n_channels_x = n_channels_x +1 



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
            train_set = XArrayDataset(ds_train, obs_train, mask=train_mask, extra_predictors= extra_predictors, lead_time_mask = params['lead_time_mask'], in_memory=True, lead_time=lead_time, time_features=time_features,ensemble_features =ensemble_features, aligned = True, year_max = year_max, model = model.__name__) 
            dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

            if reg_scale is None: ## PG: if no penalizing for negative anomalies

                    criterion = WeightedMSE(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
            else:

                if type(reg_scale) == dict:

                    criterion = WeightedMSESignLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale[test_year], min_val=0, max_val=None)

                else:
                    criterion = WeightedMSESignLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, min_val=0, max_val=None)
            if model == UNet2_decoupled:
                criterion_global = WeightedMSE(weights=np.ones([1]), device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
                W = torch.from_numpy(W.values)
  

            epoch_loss = []
            net.train()
            num_batches = len(dataloader)
            for epoch in tqdm.tqdm(range(epochs)):
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
                        loss = criterion(adjusted_forecast_pattern, y[0], mask = m)/ (torch.max(y[0]) - torch.min(y[0])) + criterion_global(adjusted_forecast_mean, y[1], mask = m_1)/ (torch.max(y[1]) - torch.min(y[1]))
                    else:
                        y = y.to(device)
                        optimizer.zero_grad()
                        adjusted_forecast = net(x)
                        loss = criterion(adjusted_forecast, y, mask = m)
                    batch_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                epoch_loss.append(batch_loss / num_batches)

                if epoch<=30: ###
                    scheduler.step() ####

            # EVALUATE MODEL
            ##################################################################################################################################
            if test_year < test_years[-1]:
                ds_test = nanremover.sample(ds_test, mode = 'Eval')  ## PG: Sample the test data at the common locations
                obs_test = nanremover.sample(obs_test)
                if model in [UNet2, UNet2_decoupled, CNN]:

                    if interp_nan:
                        
                        temp = nanremover.to_map(ds_test)
                        ds_test = temp.interpolate_na(dim ='lon', method = 'linear' ).interpolate_na(dim ='lat', method = 'linear' , fill_value="extrapolate") ## PG: Write back to map if the model starts with a nn.Conv2D
                        if land_mask:
                            mask_land = temp.where(~np.isnan(temp),1).fillna(0.0)
                            ds_test = xr.concat([ds_test, mask_land], dim = 'channels')

                        temp = nanremover.to_map(obs_test)
                        obs_test = temp.interpolate_na(dim ='lon', method = 'linear' ).interpolate_na(dim ='lat', method = 'linear' , fill_value="extrapolate")
                        if land_mask:
                            mask_land = temp.where(~np.isnan(temp),1).fillna(0.0)
                            obs_test = xr.concat([obs_test, mask_land], dim = 'channels')
                        
                    else:
                        ds_test = nanremover.to_map(ds_test).fillna(0.0)  ## PG: Write back to map if the model starts with a nn.Conv2D
                        obs_test = nanremover.to_map(obs_test).fillna(0.0)
            ##################################################################################################################################

                test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)
                test_years_list = np.arange(1, ds_test.shape[0] + 1)  ## PG: Extract the number of years as well 
                test_set = XArrayDataset(ds_test, obs_test, extra_predictors= extra_predictors, lead_time=None,lead_time_mask = params['lead_time_mask'], time_features=time_features, ensemble_features =ensemble_features,  in_memory=False, aligned = True, year_max = year_max,  model = model.__name__)
                dataloader_test = DataLoader(test_set, batch_size=len(test_lead_time_list), shuffle=False)
                criterion_test =  WeightedMSE(weights=weights_, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)

                test_results = np.zeros_like(ds_test)

                if 'ensembles' in ds_test.dims:
                    test_loss = np.zeros(shape=(ds_test.shape[0], ds_test.shape[1], ds_test.shape[2]))
                else:
                    test_loss = np.zeros(shape=(ds_test.shape[0], ds_test.shape[1]))

                for i, (x, target) in enumerate(dataloader_test): 
                    if 'ensembles' in ds_test.dims:  ## PG: If we have large ensembles:
    
                        ensemble_idx, year_idx = np.divmod(i,  len(test_years_list))  ## PG: find out ensemble index
                        # year_idx, lead_time_list_idx = np.divmod(j, len(test_lead_time_list)) 
                        # lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
                        net.eval()
                        with torch.no_grad():
                            if (type(x) == list) or (type(x) == tuple):
                                test_raw = (x[0].to(device), x[1].to(device))
                            else:
                                test_raw = x.to(device)
                            if (type(target) == list) or (type(target) == tuple):
                                test_obs, m = (target[0].to(device), target[1].to(device))
                            else:
                                test_obs = target.to(device)
                                m = None
                            
                            if model == UNet2_decoupled:
                                test_adjusted_pattern, test_adjusted_mean = net(test_raw)
                                test_adjusted = test_adjusted_pattern + test_adjusted_mean.unsqueze(-1).expand_as(test_adjusted_pattern)
                            else:
                                test_adjusted = net(test_raw)
                            loss = criterion_test(test_adjusted, test_obs)
                            test_results[year_idx, :,ensemble_idx,] = test_adjusted.to(torch.device('cpu')).numpy()  ## PG: write back to test_results
                            test_loss[year_idx, :,ensemble_idx] = loss.item()

                    else:
                        year_idx = i
                        # year_idx, lead_time_list_idx = np.divmod(i, len(test_lead_time_list))
                        # lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
                        net.eval()
                        with torch.no_grad():
                            if (type(x) == list) or (type(x) == tuple):
                                test_raw = (x[0].to(device), x[1].to(device))
                            else:
                                test_raw = x.to(device)
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
                            loss = criterion_test(test_adjusted, test_obs)
                            test_results[year_idx, :,] = test_adjusted.to(torch.device('cpu')).numpy()
                            test_loss[year_idx, :] = loss.item()

                ##################################################################################################################################
                if model in [UNet2, UNet2_decoupled , CNN]:   ## PG: if the output is already a map
                    test_results_untransformed = obs_pipeline.inverse_transform(test_results)
                    result = xr.DataArray(test_results_untransformed, ds_test.coords, ds_test.dims, name='nn_adjusted')
                else:  
                    test_results = nanremover.to_map(test_results)  ## PG: If the output is spatially flat, write back to maps
                    test_results_untransformed = obs_pipeline.inverse_transform(test_results.values) ## PG: Check preprocessing.AnomaliesScaler for changes
                    result = xr.DataArray(test_results_untransformed, test_results.coords, test_results.dims, name='nn_adjusted')

                # print(test_results)
                ##################################################################################################################################
                # Store results as NetCDF            
                result.to_netcdf(path=Path(results_dir, f'nn_adjusted_{test_year}_{run+1}.nc', mode='w'))

                fig, ax = plt.subplots(1,1, figsize=(8,5))
                ax.plot(np.arange(1,epochs+1), epoch_loss)
                ax.set_title(f'Train Loss \n test loss: {np.mean(test_loss)}') ###
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                plt.show()
                plt.savefig(results_dir+f'/Figures/train_loss_1982-{test_year-1}.png')
                plt.close()

                if save:
                    nameSave = f"MODEL_V{params['version']}_1982-{test_year-1}.pth"
                    torch.save( net.state_dict(),results_dir + '/' + nameSave)

            else:

                nameSave = f"MODEL_final_V{params['version']}_1982-{test_years[-2]}.pth"
                # Save locally
                torch.save( net.state_dict(),results_dir + '/' + nameSave)

if __name__ == "__main__":

    
    n_years =  35 # last n years to test consecutively
    lead_years = 5
    n_runs = 1  # number of training runs

    params = {
        "model": UNet2_decoupled,
        "hidden_dims": [16, 64, 128, 64, 32],
        "time_features": ['month_sin', 'month_cos'],
        "extra_predictors" : ['atm_co2'],
        "obs_clim" : False,
        "ensemble_features": False, ## PG
        'ensemble_list' : None, ## PG
        'ensemble_mode' : 'Mean',
        "epochs": 40,
        "batch_size": 64,
        "reg_scale" : None,
        "optimizer": torch.optim.Adam,
        "lr": 0.001 ,
        "loss_region": None,
        "subset_dimensions": None,
        "kernel_size" : 5,
        "decoder_kernel_size" : 5,
        "bilinear" : False, ## only for UNet
        "lead_time_mask":None,
    }


    params['ensemble_list'] = None #[f'r{e}i1p2f1' for e in range(1,21,1)] ## PG
    params['ensemble_mode'] = None
    params['ensemble_features'] = False ## PG
 
    params['interp_nan'] = False 
    params['land_mask'] = False
    params['reg_scale'] = None
    params['version'] = 3    ### 1 , 2 ,3

    
    biomes = xr.open_dataset('/home/rpg002/fgco2_decadal_forecast_adjustment/Time_Varying_Biomes.nc').MeanBiomes.transpose()
    params['Biome'] = None

    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/fgco2_ems/SOM-FFN/results/{params["model"].__name__}/run_set_1'
    # out_dir_xx = f'{out_dir_x}/git_data_20230426'
    # out_dir    = f'{out_dir_xx}/SPNA' 
    if params['model'] == CNN:
        out_dir    = f'{out_dir_x}/N{n_years}_v{params["version"]}_{params["kernel_size"]}{params["decoder_kernel_size"]}_epochs{params["epochs"]}_batch{params["batch_size"]}_L{params["reg_scale"]}_e{params["epochs"]}' 
    if params['model'] in [UNet2, UNet2_decoupled]:
        out_dir    = f'{out_dir_x}/N{n_years}_v{params["version"]}_epochs{params["epochs"]}_batch{params["batch_size"]}_L{params["reg_scale"]}_e{params["epochs"]}'
    
    if len(params["extra_predictors"]) > 0:
        out_dir = out_dir + '_extra_predictors'

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir + '/Figures').mkdir(parents=True, exist_ok=True)

    run_training(params, n_years=n_years, lead_years=lead_years, n_runs=n_runs, results_dir=out_dir, numpy_seed=1, torch_seed=1)
    print(f'Output dir: {out_dir}')
    print('Training done.')

