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
from models.autoencoder import Autoencoder, Autoencoder_decoupled
from models.unet2 import UNet2, UNet2_decoupled
from models.cnn import CNN, SCNN
from losses import WeightedMSE,WeightedMSESignLoss
from data_utils.datahandling import combine_observations
from preprocessing import align_data_and_targets, get_coordinate_indices, create_train_mask, reshape_obs_to_data, calculate_climatology
from preprocessing import AnomaliesScaler_v1, AnomaliesScaler_v2, Detrender, Standardizer, Normalizer, PreprocessingPipeline, Spatialnanremove
from torch_datasets import XArrayDataset
from subregions import subregions
from data_locations import LOC_FORECASTS_fgco2, LOC_OBSERVATIONS_fgco2_v2022, LOC_OBSERVATIONS_fgco2_v2023

# specify data directories
data_dir_forecast = LOC_FORECASTS_fgco2
data_dir_obs = LOC_OBSERVATIONS_fgco2_v2023
unit_change = 60*60*24*365 * 1000 /12 * -1 ## Change units for ESM data to mol m-2 yr-1


def predict(fct:xr.DataArray , observation:xr.DataArray , params, lead_years, model_dir,  test_years, model_year = None, ensemble_list = None, ensemble_mode = 'Mean', btstrp_it = 200, save=True):
    if 'extra_predictors' not in params.keys():
        params['extra_predictors'] = []

    if model_year is None:
        model_year_ = np.min(test_years) - 1
    else:
        model_year_ = model_year

    if params['version'] == 1:
        forecast_preprocessing_steps = [
        ('anomalies', AnomaliesScaler_v1(axis=0)), 
        ('standardize', Standardizer()) ]

        observations_preprocessing_steps = [
        ('anomalies', AnomaliesScaler_v1(axis=0))  ]

    elif params['version'] == 2:
        forecast_preprocessing_steps = [
        ('anomalies', AnomaliesScaler_v2(axis=0)), 
        ('standardize', Standardizer()) ]

        observations_preprocessing_steps = [
        ('anomalies', AnomaliesScaler_v2(axis=0))  ]
        
    else:
        forecast_preprocessing_steps = [
        ('anomalies', AnomaliesScaler_v1(axis=0)), 
        ('standardize', Standardizer()) ]

        observations_preprocessing_steps = [
        ('anomalies', AnomaliesScaler_v2(axis=0))  ]



    print(f"Start run for test year {test_years}...")

    ############################################## load data ##################################

    ensemble_features = params['ensemble_features']
    time_features = params["time_features"]
    model = params['model']
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

    ds_raw, obs_raw = align_data_and_targets(ds_in, obs_in, lead_years)  # extract valid lead times and usable years ## used to be np.min(test_years)

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
    #######################################################################################################################################
    nanremover = Spatialnanremove()## PG: Get an instance of the class
    nanremover.fit(ds_raw_ensemble_mean[:,:12,...], obs_raw[:,:12,...]) ## PG:extract the commong grid points between training and obs data
    ds_raw_ensemble_mean = nanremover.to_map(nanremover.sample(ds_raw_ensemble_mean)) ## PG: flatten and sample training data at those locations
    obs_raw = nanremover.to_map(nanremover.sample(obs_raw)) ## PG: flatten and sample obs data at those locations    
    #######################################################################################################################################
    ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean.where(ds_in.year <= model_year_, drop = True), ds_in_ ], dim = 'year')
    
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
        if params["model"] in [Autoencoder_decoupled, Autoencoder]:
            weights = np.cos(np.ones_like(extra_predictors.lon) * (np.deg2rad(extra_predictors.lat.to_numpy()))[..., None])  # Moved this up
            weights = xr.DataArray(weights, dims = extra_predictors.dims[-2:], name = 'weights').assign_coords({'lat': extra_predictors.lat, 'lon' : extra_predictors.lon}) 
            extra_predictors = (extra_predictors * weights).sum(['lat','lon'])/weights.sum(['lat','lon'])
        else:  
            if not all(['ensembles' not in extra_predictors.dims, 'ensembles' in ds_raw_ensemble_mean.dims]): 
                
                ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, extra_predictors], dim = 'channels')
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


    # Data preprocessing
    
    ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_baseline, mask=preprocessing_mask_fct)
    ds = ds_pipeline.transform(ds_raw_ensemble_mean)


    obs_pipeline = PreprocessingPipeline(observations_preprocessing_steps).fit(obs_baseline, mask=preprocessing_mask_obs)
    if 'standardize' in ds_pipeline.steps:
        obs_pipeline.add_fitted_preprocessor(ds_pipeline.get_preprocessors('standardize'), 'standardize')
    obs = obs_pipeline.transform(obs_raw)
  
    # TRAIN MODEL
    year_max = ds[:n_train].year[-1].values 
    lead_time = None
    ds_train = ds[:n_train,...]
    obs_train = obs[:n_train,...]
    
    ds_test = ds[n_train:,...]

        
    weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
    weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove() 
    ########################################################################

    if model in [UNet2, UNet2_decoupled , CNN]: ## PG: If the model starts with a nn.Conv2d write back the flattened data to maps.
        if interp_nan is not None:
            ds_train = ds_train.interpolate_na(dim =interp_nan, method = 'linear' ) ## PG: fill NaN values with 0.0 for training
        else:
            ds_train = ds_train.fillna(0.0) ## PG: fill NaN values with 0.0 for training

        img_dim = ds_train.shape[-2] * ds_train.shape[-1] 
 
    else: ## PG: If you have a dense first layer keep the data flattened.
        ds_train = nanremover.sample(ds_train)
        weights = nanremover.sample(weights) 
        img_dim = ds_train.shape[-1] ## PG: The input dim is now the length of the flattened dimention.
    weights = weights.values
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
    ########################################### load the model ######################################
    try:
        if params['obs_clim']:
            n_channels_x = len(ds_train.channels) + 1
        else:
            n_channels_x = len(ds_train.channels)
    except:
        pass

    if model == Autoencoder:
        net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, device = device)
    elif model == UNet2:
        net = model(n_channels_x= n_channels_x+ add_feature_dim , bilinear = params['bilinear'])
    elif model == UNet2_decoupled:
        net = model(n_channels_x= n_channels_x+ add_feature_dim , bilinear = params['bilinear'])
    elif model == CNN: ## PG: Combining CNN encoder with dense decoder
        net = model(n_channels_x + add_feature_dim ,hidden_dims, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size )
    elif model == Autoencoder_decoupled:
        net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate)

    print('Loading model ....')
    net.load_state_dict(torch.load(glob.glob(model_dir + f'/*-{model_year_}*.pth')[0], map_location=torch.device('cpu'))) 
    net.to(device)
    net.eval()
    ##################################################################################################################################
    
    ds_test = nanremover.sample(ds_test, mode = 'Eval')  ## PG: Sample the test data at the common locations
    if model in [UNet2, UNet2_decoupled, CNN]:
        if interp_nan is not None:
                ds_test = nanremover.to_map(ds_test).interpolate_na(dim =interp_nan, method = 'linear' )  ## PG: Write back to map if the model starts with a nn.Conv2D
        else:
                ds_test = nanremover.to_map(ds_test).fillna(0.0)  ## PG: Write back to map if the model starts with a nn.Conv2D
    ##################################################################################################################################

    test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)
    test_years_list = np.arange(1, ds_test.shape[0] + 1)  ## PG: Extract the number of years as well 
    test_set = XArrayDataset(ds_test, xr.ones_like(ds_test).rename({'lead_time':'month'}), extra_predictors= extra_predictors, lead_time=None, time_features=time_features,ensemble_features =ensemble_features,  in_memory=False, aligned = True, year_max = year_max,  model = model.__name__)

    test_results = np.zeros_like(ds_test)

    if 'ensembles' in ds_test.dims:
        test_loss = np.zeros(shape=(ds_test.shape[0], ds_test.shape[1], ds_test.shape[2]))
    else:
        test_loss = np.zeros(shape=(ds_test.shape[0], ds_test.shape[1]))

    for i, (x, target) in enumerate(test_set): 
        if 'ensembles' in ds_test.dims:  ## PG: If we have large ensembles:

            ensemble_idx, j = np.divmod(i, len(test_lead_time_list) * len(test_years_list))  ## PG: find out ensemble index
            year_idx, lead_time_list_idx = np.divmod(j, len(test_lead_time_list)) 
            lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
            
            with torch.no_grad():
                if (type(x) == list) or (type(x) == tuple):
                    test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                else:
                    test_raw = x.unsqueeze(0).to(device)
                if model == Autoencoder_decoupled:
                    test_adjusted_pattern, test_adjusted_mean = net(test_raw)
                    test_adjusted = test_adjusted_pattern + test_adjusted_mean.expand_as(test_adjusted_pattern)
                elif model == UNet2_decoupled:
                    test_adjusted_pattern, test_adjusted_mean = net(test_raw)
                    test_adjusted = test_adjusted_pattern + test_adjusted_mean.unsqueeze(-1).expand_as(test_adjusted_pattern)
                else:
                    test_adjusted = net(test_raw)
                test_results[year_idx, lead_time_idx,ensemble_idx,] = test_adjusted.to(torch.device('cpu')).numpy()  ## PG: write back to test_results

        else:

            year_idx, lead_time_list_idx = np.divmod(i, len(test_lead_time_list))
            lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
            with torch.no_grad():
                if (type(x) == list) or (type(x) == tuple):
                    test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                else:
                    test_raw = x.unsqueeze(0).to(device)
                if model == Autoencoder_decoupled:
                    test_adjusted_pattern, test_adjusted_mean = net(test_raw)
                    test_adjusted = test_adjusted_pattern + test_adjusted_mean.expand_as(test_adjusted_pattern)
                elif model == UNet2_decoupled:
                    test_adjusted_pattern, test_adjusted_mean = net(test_raw)
                    test_adjusted = test_adjusted_pattern + test_adjusted_mean.unsqueeze(-1).expand_as(test_adjusted_pattern)
                else:
                    test_adjusted = net(test_raw)
                test_results[year_idx, lead_time_idx,] = test_adjusted.to(torch.device('cpu')).numpy()

    ##################################################################################################################################
    if model in [UNet2, UNet2_decoupled , CNN]:   ## PG: if the output is already a map
        test_results_untransformed = obs_pipeline.inverse_transform(test_results)
        result = xr.DataArray(test_results_untransformed, ds_test.coords, ds_test.dims, name='nn_adjusted')
    else:  
        test_results = nanremover.to_map(test_results)  ## PG: If the output is spatially flat, write back to maps
        test_results_untransformed = obs_pipeline.inverse_transform(test_results.values) ## PG: Check preprocessing.AnomaliesScaler for changes
        result = xr.DataArray(test_results_untransformed, test_results.coords, test_results.dims, name='nn_adjusted')
    
    result = result.isel(lead_time = np.arange(0,np.min([params['lead_months_to_predict'], lead_years * 12]))) 

    if save:
        if model_year is not None:
            Path(model_dir + f'/{model_year}_model_predictions').mkdir(parents=True, exist_ok=True)
            model_dir = model_dir + f'/{model_year}_model_predictions'

        if ensemble_mode != 'Mean':
            if np.min(test_years) != np.max(test_years):
                save_name = f'saved_model_nn_adjusted_{np.min(test_years)}-{np.max(test_years)}_LE.nc'
            else:
                save_name = f'saved_model_nn_adjusted_{np.min(test_years)}_LE.nc'

        else:
            if np.min(test_years) != np.max(test_years):
                save_name = f'saved_model_nn_adjusted_{np.min(test_years)}-{np.max(test_years)}.nc'
            else:
                save_name = f'saved_model_nn_adjusted_{np.min(test_years)}.nc'

        if params['real_emissions']:
            save_name = save_name.split('.nc')[0] + '_realistic_emissions.nc'

        result.to_netcdf(path=Path(model_dir, save_name, mode='w'))
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
            params[key] = value
    return params

if __name__ == "__main__":

    ############################################## Set_up ############################################

    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/fgco2_ems/SOM-FFN/results/Autoencoder/run_set_7'
    out_dir    = f'{out_dir_x}/N36_v3_LNone_arch3_batch64_e60_lr_scheduler_obs_V2023_LE' 

    lead_years = int(len(xr.open_mfdataset(str(Path(out_dir + '/E1'   , "*.nc")), combine='nested', concat_dim='year').lead_time)/12)
    bootstrap = False
    test_years = [2023]
    real_emissions = False

    observation = combine_observations(data_dir_obs, two_dim=True) # 1961.01 - 2021.12
    fct = xr.open_mfdataset(str(Path(data_dir_forecast, "*.nc")), combine='nested', concat_dim='year').load()
    ##################################################################################################
    
    # params = extract_params(out_dir)
    # print(f'loaded configuration: \n')
    # for key, values in params.items():
    #     print(f'{key} : {values} \n')
    
    # try:
    #     version = int(out_dir.split('/')[-1].split('_')[1][1])
    # except:
    #     version = (out_dir.split('/')[-1].split('_')[1][1:])
      
    # params["version"] = version
    # print( f'Version: {version}')
    # print(f'lead_years: {lead_years}')

    # if real_emissions:
    #     params['real_emissions'] = True
    #     fct_real = xr.open_mfdataset(str(Path('/space/hall5/sitestore/eccc/crd/ccrn/users/rsa001/big_store/AI/FGCO2/data/forecasts/canesm5-ems-gcb-v2023.1', "*.nc")), combine='nested', concat_dim='year').load() 
    #     fct_real = fct_real.assign_coords(lead_time = np.arange(1,13))
    #     params['lead_months_to_predict'] = len(fct_real.lead_time)
    #     for yr in fct_real.year.values:
    #         if yr > 2015:
    #             fct_1 = fct.where(fct.year < yr, drop = True)
    #             fct_2 = fct.where(fct.year > yr, drop = True)
    #             fct_ = xr.concat([fct_1, fct_real.sel(year= yr), fct_2], dim = 'year')
    #             test_years = [yr] 
    #             predict(fct_, observation, params, lead_years, out_dir,  test_years,  ensemble_mode='Mean',  save=True)
    #     # fct_1 = fct.where(fct.year < 2024, drop = True)
    #     # fct_2 = fct.where(fct.year > 2024, drop = True)
    #     # fct_ = xr.concat([fct_1, fct_real.sel(year= 2024), fct_2], dim = 'year')
    #     # test_years = [2024] 
    #     # predict(fct_, observation, params, lead_years, out_dir, test_years,model_year= 2022, ensemble_mode='Mean',  save=True)
        
    # else:
    #     params['real_emissions'] = False
    #     if bootstrap:

    #         result_list = []
    #         ensembles = np.arange(1,11)#[f'r{i}i1p2f1' for i in range(1,11)]

    #         for it in tqdm(range(200)):

    #             ensemble_list = [random.choice(ensembles) for _ in range(len(ensembles))]
    #             result_list.append(predict(fct, observation, params, lead_years, out_dir,  test_years, ensemble_list = ensemble_list,  save=False))
            
    #         output = xr.concat(result_list, dim = 'iteration')

    #         if np.min(test_years) != np.max(test_years):
    #             output.to_netcdf(path=Path(out_dir, f'saved_model_nn_adjusted_{np.min(test_years)}-{np.max(test_years)}_bootstrap.nc', mode='w'))
    #         else:
    #             output.to_netcdf(path=Path(out_dir, f'saved_model_nn_adjusted_{np.min(test_years)}_bootstrap.nc', mode='w'))
        
    #     else:

    #         params['lead_months_to_predict'] = lead_years * 12
    #         predict(fct, observation, params, lead_years, out_dir,  test_years,  ensemble_mode='Mean',  save=True)

    # print(f'Output dir: {out_dir}')
    # print('Saved!')

    ##################################################################################################
    for ens in range(25):
        print( f'\n running for ensemble member {ens}\n' + 
              '-------------------------------------------------------------------------------------------------------------------\n')

        params = extract_params(out_dir + f'/E{ens}')
        print(f'loaded configuration: \n')
        for key, values in params.items():
            print(f'{key} : {values} \n')
            
        try:
            version = int(out_dir.split('/')[-1].split('_')[1][1])
        except:
            version = (out_dir.split('/')[-1].split('_')[1][1:])
        
        params["version"] = version
        print( f'Version: {version}')
        print(f'lead_years: {lead_years}')
        params['real_emissions'] = False
        params['lead_months_to_predict'] = lead_years * 12 
        predict(fct, observation, params, lead_years, out_dir + f'/E{ens}',  test_years, ensemble_mode='Mean',  save=True)

        print(f'Output dir: {out_dir}')
        print('Saved!\n' + 
              '-------------------------------------------------------------------------------------------------------------------\n')
