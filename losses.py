import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

class WeightedMSE:

    def __init__(self, weights, device, hyperparam=2.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, exclude_dim = None):
        self.reduction = reduction
        self.hyperparam = hyperparam
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.loss_area = loss_area
        self.device = device
        self.exclude_dim = exclude_dim
        if self.loss_area is not None:

            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            self.weights = torch.from_numpy(weights).to(device)

    def __call__(self, data, target, mask = None):

        if self.loss_area is not None:


            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data[..., indices]
                y = target[..., indices]
        else:
            y_hat = data
            y = target

        m = torch.ones_like(y)
        m[(y < self.min_threshold) & (y_hat >= 0)] *= self.hyperparam
        m[(y > self.max_threshold) & (y_hat <= 0)] *= self.hyperparam

        if mask is not None:
            weight = self.weights * mask
        else:
            weight = self.weights 
        
        dims_to_sum = tuple(d for d in range(y.dim()) if d != self.exclude_dim) if self.exclude_dim is not None else tuple(d for d in range(y.dim()))

        if self.reduction == 'mean':
            loss = ((y_hat - y)**2 * m * weight).sum(dims_to_sum) / (torch.ones_like(y) * weight).sum(dims_to_sum)
        elif self.reduction == 'sum':
            # loss = torch.mean((y_hat - y)**2 * m, dim=0)
            # loss = torch.sum(loss * weight)
            # loss = torch.sum((y_hat - y)**2 * m* weight, dim = dims_to_sum)
            loss = torch.sum((y_hat - y)**2 * m, dim = -1).mean()
        elif self.reduction == 'none':
            loss = (y_hat - y)**2 * m * (weight /weight.sum())
        
        return loss



class WeightedMSESignLoss:  ## PG: penalizing negative anomalies
    def __init__(self, weights, device, hyperparam=2.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, exclude_zeros=True, scale=1, min_val=0, max_val=None, exclude_dim = None):
        self.mse = WeightedMSE(weights=weights, device=device, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area, exclude_dim = exclude_dim)
        self.sign_loss = SignLoss( device=device, scale=scale, min_val=min_val, max_val=max_val, weights=weights, loss_area=loss_area, exclude_zeros=exclude_zeros, exclude_dim = exclude_dim)

    def __call__(self, data, target, mask = None):
        loss = 0
        loss += self.mse(data, target, mask = mask)
        loss += self.sign_loss(data, target, mask = mask)
        return loss
    

class WeightedMSEKLD:  ## PG: penalizing negative anomalies
    def __init__(self, weights, device, hyperparam=2.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, Beta = 1):
        self.mse = WeightedMSE(weights=weights, device=device, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area)
        self.reduction = reduction
        if Beta is None:
            self.beta = 1
        else:
            self.beta = Beta

    def __call__(self, data, target, mu, log_var, mask = None, return_ind_loss = False, print_loss = True):
        loss = 0
        MSE = self.mse(data, target, mask)
        if print_loss:
            print(f'MSE : {MSE}')
        loss += MSE #MSE.mean()/(MSE.max() - MSE.min())
        
        var = (torch.exp(log_var) + 1e-4)
        KL = kl_divergence(
                        Normal(mu,  torch.sqrt(var)),
                        Normal(torch.zeros_like(mu), torch.ones_like(log_var)))
        if self.reduction == 'mean':
            KL = KL.mean()
        if self.reduction == 'sum':
            KL = KL.sum(dim=-1).mean()
        # KL =  ( 0.5 * (var + mu**2 - torch.log(var) - 1)).sum(dim=1).mean()
        loss += KL*self.beta #/(KL.max() - KL.min())
        if print_loss: 
            print(f'KLD : {KL}')
        if return_ind_loss:
            return loss, MSE, KL
        else:
            return loss

class WeightedMSESignLossKLD:  ## PG: penalizing negative anomalies
    def __init__(self, weights, device, hyperparam=2.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, exclude_zeros=True, scale=1, min_val=0, max_val=None, Beta = 1):
        self.mse = WeightedMSESignLoss(weights=weights, device=device, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area, scale=scale, min_val=min_val, max_val=max_val)
        if Beta is None:
            self.beta = 1
        else:
            self.beta = Beta
        self.reduction = reduction
    def __call__(self, data, target, mu, log_var, mask = None, return_ind_loss = False, print_loss = True):
        loss = 0
        MSE = self.mse(data, target, mask)
        loss += MSE#.mean()/(MSE.max() - MSE.min())
        if print_loss:
            print(f'MSE : {loss}')
        var = (torch.exp(log_var) + 1e-4)
        KL = kl_divergence(
                        Normal(mu, torch.sqrt(var)),
                        Normal(torch.zeros_like(mu), torch.ones_like(log_var)))
        # KL =  ( 0.5 * (var + mu**2 - torch.log(var) - 1)).sum(dim=1).mean()
        if self.reduction == 'mean':
            KL = KL.mean()
        if self.reduction == 'sum':
            KL = KL.sum(dim=-1).mean()
        loss += KL * self.beta #/(KL.max() - KL.min()) 
        if print_loss: 
            print(f'KLD : {KL}')
        if return_ind_loss:
            return loss, MSE, KL
        else:
            return loss
    

class SignLoss:  ## PG: Loss function based on negative anomalies

    def __init__(self,  device, scale=1, min_val=0, max_val=None, weights=None, loss_area=None, exclude_zeros=True, exclude_dim = None):
        self.scale=scale
        self.min_val = min_val
        self.max_val=max_val
        self.weights = weights
        self.exclude_zeros = exclude_zeros
        self.loss_area = loss_area
        self.device = device
        self.exclude_dim = exclude_dim
        if loss_area is not None:
            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            self.weights = torch.from_numpy(weights).to(device)

    def __call__(self, data, target, mask = None):
        if self.loss_area is not None:

            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data[..., indices]
                y = target[..., indices]
        else:
            y_hat = data
            y = target

        dims_to_sum = tuple(d for d in range(y.dim()) if d != self.exclude_dim) if self.exclude_dim is not None else tuple(d for d in range(y.dim()))

        l = torch.clamp((y * y_hat) * (-1) * self.scale, self.min_val, self.max_val) ## Check
        if self.weights is None:
            if self.exclude_zeros:
                loss = l.sum(dim = dims_to_sum ) / self.loss_mask.sum(dim = dims_to_sum )
            else:
                loss = torch.mean(l, dim = dims_to_sum )
        else:
            if mask is not None:
                weight = self.weights * mask
            else:
                weight = self.weights

            loss = (l * weight).sum(dim = dims_to_sum ) / ( torch.ones_like(l) * weight).sum(dim = dims_to_sum ) ## Check
            
        return loss
    


class CorrLoss:

    def __init__(self,  device, scale=1,  map = False):
        self.scale=scale
        self.map = map
        self.device = device


    def __call__(self, data, target, mask = None):

        y_hat = data
        y = target

        if self.map:
                if mask is not None:
                    mask = mask.mean(dim = (2,3))
                
                mean1 = y.mean(dim=(2,3)).unsqueeze(2).unsqueeze(3).expand_as(y)
                mean2 = y_hat.mean(dim=(2,3)).unsqueeze(2).unsqueeze(3).expand_as(y_hat)
                std1 = y.std(dim=(2,3))
                std2 = y_hat.std(dim=(2,3))

                covariance = ((y - mean1) * (y_hat - mean2)).mean(dim = (2,3))
                correlation = covariance / (std1 * std2)

        else:
                if mask is not None:
                    mask = mask.mean(dim = (2))

                mean1 = y.mean(dim=(2)).unsqueeze(2).expand_as(y)
                mean2 = y_hat.mean(dim=(2)).unsqueeze(2).expand_as(y_hat)
                std1 = y.std(dim=(2))
                std2 = y_hat.std(dim=(2))


                covariance = ((y - mean1) * (y_hat - mean2)).mean(dim = (2))
                correlation = covariance / (std1 * std2)
        if mask is not None:
            if len(correlation[mask  != 0 ]) == 0:
                loss = correlation[mask  != 0 ].sum()
            else:
                loss = correlation[mask  != 0 ].mean()
        else:
            loss = correlation.mean()
        return loss



class GlobalLoss:  ## PG: Loss function based on negative anomalies

    def __init__(self,  device, scale=1, weights=None, loss_area=None, map = False):
        self.scale=scale
        self.weights = weights
        self.loss_area = loss_area
        self.device = device
        self.map = map
        if loss_area is not None:
            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            self.weights = torch.from_numpy(weights).to(device)

    def __call__(self, data, target, mask = None):
        
        if self.loss_area is not None:

            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data[..., indices]
                y = target[..., indices]
        else:
            y_hat = data
            y = target
        if self.map:
                if mask is not None:
                    mask = mask.mean(dim=(2,3))
                l1 = (y *self.weights).sum(dim=(2,3)) 
                l2 = (y_hat *self.weights).sum(dim=(2,3)) 

        else:
                if mask is not None:
                    mask = mask.mean(dim=(2))
                l1 = (y *self.weights).sum(dim=(2)) 
                l2 = (y_hat *self.weights).sum(dim=(2)) 

        if mask is None:
            loss =  ((l1-l2)**2 * self.scale).mean()
        else:
            loss =  ((l1-l2)**2 * self.scale * mask).sum()/mask.sum()

        return loss
    

class VAEloss:  ## PG: penalizing negative anomalies
    def __init__(self, weights, device, reduction='sum'):
        if weights is not None:
            self.mse = WeightedMSE(weights=weights, device=device, hyperparam=1, reduction=reduction)
        else:
            self.mse = None
    def __call__(self, data, target, mask = None, Beta = 1):
        loss = 0
        if self.mse is not None:
            MSE = self.mse(torch.mean(data,0).squeeze(), target[...,0,:], mask)
            print(f'MSE : {MSE}')
            loss += MSE #MSE.mean()/(MSE.max() - MSE.min())
            KL1 = kl_divergence(
                            Normal(0, torch.std(data, 0).squeeze()),
                            Normal(0, target[...,1,:]))
            loss += KL1.sum(dim=-1).mean()
            print(f'KLD1 : {KL1.sum(dim=-1).mean()}')
        
        else:
            KL1 = kl_divergence(
                            Normal(torch.mean(data,0).squeeze(), torch.std(data, 0).squeeze()),
                            Normal(target[...,0,:], target[...,1,:]))
            loss += KL1.sum(dim=-1).mean()
            print(f'KLD1 : {KL1.sum(dim=-1).mean()}') 

        return loss