import numpy as np
import xarray as xr


def annual_global_mean(data, weights=None, region=None):
    if weights is None:
        weights = np.ones(data.shape[-3:])
    n_years = int(data.shape[1] / 12)
    if n_years > 1:
        x = data * weights
        if isinstance(x, xr.DataArray):
            x = x.coarsen(lead_time=12).sum().sum(axis=(2,3,4))
            annual_global_mean = x / (weights.sum()*12)
        else:
            x = np.concatenate([x[:, np.arange(i*12, i*12+12), ...].sum(axis=(1, 2, 3, 4))[..., None] for i in range(n_years)], axis=1)
            annual_global_mean = x / (weights.sum()*12)
    else:
        annual_global_mean = (data * weights).sum(axis=(1, 2, 3, 4)) / (weights.sum()*12)
    return annual_global_mean


def global_rmse(y_pred, y_true, weights=None, region=None):
    if weights is None:
        weights = np.ones(y_pred.shape[-3:])
    rmse = np.sqrt(((y_pred - y_true)**2 * weights).sum(axis=(-3, -2, -1)) / weights.sum())
    return rmse



# class Spatialnanremove:


#     def __init__(self):
#         pass

#     def fit(self, data):

#         self.data = data
#         self.reference_shape = xr.full_like(data, fill_value = np.nan )
#         return self

#     def transform(self):

#         nanremoved = self.data.stack(ref = ['lat','lon']).dropna(dim = 'ref')
#         self.coords = nanremoved.coords 
#         self.dims = nanremoved.dims
#         self.ref = nanremoved.ref
#         return nanremoved

#     def inverse_transform(self, npar : np.ndarray):
#         if not isinstance(npar, np.ndarray):
#             raise TypeError("Input must be a numpy.ndarray")
        
#         output =  xr.DataArray(npar, self.coords, self.dims, name='nn_adjusted').unstack()
#         return output.combine_first(self.reference_shape)

#     def sample(self, target, name = None, dims = None, coords = None):

#         if not isinstance(target, DataArray):
#             if dims is not None:
#                 if coords in None:
#                     raise ValueError(f"Please provide coordinates for {dims}")
#                 out = xr.DataArray(target, dims = dims, coords = coords, name=name)
#             else:
#                 target = xr.DataArray(target, dims = self.data.dims , coords = self.data.coords, name=name)
#         else:
#             conditions = ['lat' in target.dims, 'lon' in target.dims]
#             if all(conditions):
#                 return target.stack(ref = ['lat','lon']).sel(ref = self.ref)
#             else:
#                 return target.sel(ref = self.ref)
