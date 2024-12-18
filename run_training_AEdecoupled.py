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
from models.autoencoder import Autoencoder_decoupled, Autoencoder
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
    
    assert params['version'] in [1,2,3]
    
    if params["model"] != Autoencoder_decoupled:
        params["append_mode"] = None
        
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
    batch_normalization = params["batch_normalization"]
    dropout_rate = params["dropout_rate"]

    optimizer = params["optimizer"]
    lr = params["lr"]

    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]

    loss_region = params["loss_region"]
    subset_dimensions = params["subset_dimensions"]

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
            f"extra_predictors\t{params['extra_predictors']}\n" +
            f"append_mode\t{params['append_mode']}\n" +
            f"ensemble_list\t{ensemble_list}\n" + ## PG: Ensemble list
            f"hyperparam\t{hyperparam}\n" + ## PG: Ensemble features
            f"epochs\t{epochs}\n" +
            f"batch_size\t{batch_size}\n" +
            f"batch_normalization\t{batch_normalization}\n" +
            f"dropout_rate\t{dropout_rate}\n" +
            f"optimizer\t{optimizer.__name__}\n" +
            f"lr\t{lr}, halved after each 10 epochs\n" +
            f"forecast_preprocessing_steps\t{[s[0] if forecast_preprocessing_steps is not None else None for s in forecast_preprocessing_steps]}\n" +
            f"observations_preprocessing_steps\t{[s[0] if observations_preprocessing_steps is not None else None for s in observations_preprocessing_steps]}\n" +
            f"loss_region\t{loss_region}\n" +
            f"subset_dimensions\t{subset_dimensions}"
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

            ds_train = nanremover.sample(ds_train) ## PG: flatten and sample training data at those locations
            obs_train = nanremover.sample(obs_train) 
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

                    add_feature_dim = 0
            else:

                    add_feature_dim = len(time_features)
            
            if extra_predictors is not None:
                add_feature_dim = add_feature_dim + len(params['extra_predictors'])



            if model == Autoencoder_decoupled:
                net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate)


            net.to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            ## PG: XArrayDataset now needs to know if we are adding ensemble features. The outputs are datasets that are maps or flattened in space depending on the model.
            train_set = XArrayDataset(ds_train, obs_train, mask=train_mask, extra_predictors= extra_predictors, in_memory=True, lead_time=lead_time, time_features=time_features, aligned = True, year_max = year_max) 
            dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

            if reg_scale is None: ## PG: if no penalizing for negative anomalies

                    criterion = WeightedMSE(weights=weights, device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices)
            else:

                if type(reg_scale) == dict:

                    criterion = WeightedMSESignLoss(weights=weights, device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices, scale=reg_scale[test_year], min_val=0, max_val=None)

                else:
                    criterion = WeightedMSESignLoss(weights=weights, device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, min_val=0, max_val=None)
            
            criterion_global = WeightedMSE(weights=np.ones([1]), device=device, hyperparam=hyperparam, reduction='mean', loss_area=loss_region_indices)
            weights = torch.from_numpy(weights)

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
                    
                    y_1 = (y * weights).sum(dim = -1)/ weights.sum(dim = -1)
                    y_0 = y - y_1.unsqueeze(-1).expand_as(y)
                    y = (y_0.to(device), y_1.unsqueeze(-1).to(device))

                    optimizer.zero_grad()
                    adjusted_forecast_pattern, adjusted_forecast_mean = net(x)
                    loss = criterion(adjusted_forecast_pattern, y[0])/ (torch.max(y[0]) - torch.min(y[0])) + criterion_global(adjusted_forecast_mean, y[1])/ (torch.max(y[1]) - torch.min(y[1]))

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
            ##################################################################################################################################

                test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)
                test_years_list = np.arange(1, ds_test.shape[0] + 1)  ## PG: Extract the number of years as well 
                test_set = XArrayDataset(ds_test, obs_test, extra_predictors= extra_predictors, lead_time=None, time_features=time_features,  in_memory=False, aligned = True, year_max = year_max)
                criterion_test =  WeightedMSE(weights=weights_, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
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
                        net.eval()
                        with torch.no_grad():
                            if (type(x) == list) or (type(x) == tuple):
                                test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                            else:
                                test_raw = x.unsqueeze(0).to(device)

                            test_obs = target.unsqueeze(0).to(device)
                            test_adjusted_pattern, test_adjusted_mean = net(test_raw)
                            test_adjusted = test_adjusted_pattern + test_adjusted_mean.expand_as(test_adjusted_pattern)
                            loss = criterion_test(test_adjusted, test_obs)
                            test_results[year_idx, lead_time_idx,ensemble_idx,] = test_adjusted.to(torch.device('cpu')).numpy()  ## PG: write back to test_results
                            test_loss[year_idx, lead_time_idx,ensemble_idx] = loss.item()

                    else:

                        year_idx, lead_time_list_idx = np.divmod(i, len(test_lead_time_list))
                        lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
                        net.eval()
                        with torch.no_grad():
                            if (type(x) == list) or (type(x) == tuple):
                                test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                            else:
                                test_raw = x.unsqueeze(0).to(device)

                            test_obs = target.unsqueeze(0).to(device)
                            test_adjusted_pattern, test_adjusted_mean = net(test_raw)
                            test_adjusted = test_adjusted_pattern + test_adjusted_mean.expand_as(test_adjusted_pattern)
                            loss = criterion_test(test_adjusted, test_obs)
                            test_results[year_idx, lead_time_idx,] = test_adjusted.to(torch.device('cpu')).numpy()
                            test_loss[year_idx, lead_time_idx] = loss.item()

                ##################################################################################################################################

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
        "model": Autoencoder_decoupled,
        "hidden_dims": [[3600, 1800, 900], [1800, 3600]],
        "time_features":  ['month_sin', 'month_cos'],
        "extra_predictors" : [],
        'ensemble_list' : None, ## PG
        'ensemble_mode' : 'Mean',
        "epochs": 50,
        "batch_size": 64,
        "batch_normalization": False,
        "dropout_rate": 0,
        "append_mode": 3,
        "hyperparam"  :1, 
        "reg_scale" : None,
        "optimizer": torch.optim.Adam,
        "lr": 0.001 ,

        "loss_region": None,
        "subset_dimensions": None
    }



    params['ensemble_list'] = None #[f'r{e}i1p2f1' for e in range(1,21,1)] ## PG
 
    params["arch"] = 2
    params['version'] = 3  ### 1 , 2 ,3
    params['reg_scale'] = 3

    
    biomes = xr.open_dataset('/home/rpg002/fgco2_decadal_forecast_adjustment/Time_Varying_Biomes.nc').MeanBiomes.transpose()
    params['Biome'] = None

    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/fgco2_ems/SOM-FFN/results/{params["model"].__name__}/run_set_1'
    # out_dir_xx = f'{out_dir_x}/git_data_20230426'
    # out_dir    = f'{out_dir_xx}/SPNA' 
    out_dir    = f'{out_dir_x}/N{n_years}_v{params["version"]}_L{params["reg_scale"]}_arch{params["arch"]}_batch{params["batch_size"]}_e{params["epochs"]}'
    if len(params["extra_predictors"]) > 0:
        out_dir = out_dir + '_extra_predictors'
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir + '/Figures').mkdir(parents=True, exist_ok=True)

    run_training(params, n_years=n_years, lead_years=lead_years, n_runs=n_runs, results_dir=out_dir, numpy_seed=1, torch_seed=1)
    print(f'Output dir: {out_dir}')
    print('Training done.')

