import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SCNN(nn.Module):
    def __init__(
        self,
        n_channels_x,
        hidden_dimensions:list,
        kernel_size = 3,
        DSC = False,
        kernels_per_layer=2,
        reduction_ratio=16,
        decoder_kernel_size = 1):

        super().__init__()
        self.n_channels_x = n_channels_x
        kernels_per_layer = kernels_per_layer
        reduction_ratio = reduction_ratio

        assert np.mod(kernel_size,2) == 1 
        if decoder_kernel_size is not None:
            assert np.mod(decoder_kernel_size,2) == 1
            self.decoder_kernel = decoder_kernel_size
        else:
            self.decoder_kernel = None


        layers = []
        layers.append(SphericalConv(n_channels_x , hidden_dimensions[0] ,DSC = DSC, kernel_size = kernel_size,  kernels_per_layer=kernels_per_layer))
        for layer in range(len(hidden_dimensions) - 1):

                layers.append(SphericalConv(hidden_dimensions[layer], hidden_dimensions[layer + 1] ,DSC = DSC, kernel_size=kernel_size, kernels_per_layer=kernels_per_layer))

        self.encoder = nn.Sequential(*layers)

        if self.decoder_kernel is None:
            self.decoder =  nn.Sequential(DepthwiseSeparableConv(hidden_dimensions[-1],  1, kernel_size=3,  kernels_per_layer=kernels_per_layer))
        else:
            self.decoder = nn.Sequential(nn.Conv2d(hidden_dimensions[-1], 1, kernel_size=decoder_kernel_size))



    def pad(self, x,    size): # NxCxHxW

        if type(size) in [list, tuple]:
            size_v = size[0]
            size_h = size[1]
        else:
            size_h = size_v = size

        north_pad = torch.flip(x[...,-1*size_v:,:], dims=[-2])
        south_pad = north_pad = torch.flip(x[...,:size_v,:], dims=[-2])
        north_pad = torch.roll(north_pad, shifts = 180, dims = [-1])
        south_pad = torch.roll(south_pad, shifts = 180, dims = [-1])
        x_padded = torch.cat([south_pad, x, north_pad], dim = -2 )
        east_pad = x_padded[...,:size_h]
        west_pad = x_padded[...,-1*size_h:]
        x_padded = torch.cat([west_pad, x_padded, east_pad], dim = -1 )
        
        return x_padded
    
    def forward(self, x):

        if (type(x) == list) or (type(x) == tuple):
                
                x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x

        x_encoded = self.encoder(x_in)

        if self.decoder_kernel is not None:
            if self.decoder_kernel>1:
                x_encoded = self.pad(x_encoded, int(self.decoder_kernel /2) )
        else:
             x_encoded = self.pad(x_encoded, 1)

        x_out = self.decoder(x_encoded)

        return x_out
        
class CNN(nn.Module):
    def __init__(
        self,
        n_channels_x,
        hidden_dimensions:list,
        kernel_size = 3, 
        decoder_kernel_size = 1):

        super().__init__()
        self.n_channels_x = n_channels_x

        assert np.mod(decoder_kernel_size,2) == 1
        self.decoder_kernel = decoder_kernel_size

        layers = []
        layers.append(Convblock(n_channels_x , hidden_dimensions[0] ,  kernel_size = kernel_size))
        for layer in range(len(hidden_dimensions) - 1):
                layers.append(Convblock(hidden_dimensions[layer], hidden_dimensions[layer + 1] , kernel_size = kernel_size))

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Conv2d(hidden_dimensions[-1], 1, kernel_size=decoder_kernel_size, padding= int(self.decoder_kernel /2))

    
    def forward(self, x):

        if (type(x) == list) or (type(x) == tuple):
                
                x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x

        x_encoded = self.encoder(x_in)
        x_out = self.decoder(x_encoded)

        return x_out
    



class Convblock(nn.Module):
        def __init__( self, in_channels, out_channels, DSC = False,  kernel_size = 3,  kernels_per_layer=1 ):
                
                super().__init__()
                assert np.mod(kernel_size,2) == 1
                if DSC:
                        self.conv = nn.Sequential( 
                                    DepthwiseSeparableConv(
                                    in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    padding= int(kernel_size/2),
                                    kernels_per_layer=kernels_per_layer),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),)
                else:
                        self.conv = nn.Sequential( 
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding = int(kernel_size /2) ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),)
                

        def forward(self, x):

            out = self.conv(x)
            return out
        



class SphericalConv(nn.Module):
        def __init__( self, in_channels, out_channels, DSC = False, kernel_size = 3,  kernels_per_layer=1 ):
                
                super().__init__()
                self.kernel_size = kernel_size
                if DSC:
                        self.conv = nn.Sequential( 
                                    DepthwiseSeparableConv(
                                    in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    kernels_per_layer=kernels_per_layer),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),)
                else:
                        self.conv = nn.Sequential( 
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),)
                

        def forward(self, x):

            # print(f'before padding  : {x.shape}')

            x = self.pad(x, int(self.kernel_size/2))

            # print(f'after padding : {x.shape}')

            out = self.conv(x)

            # print(f'after conv block : {out.shape}')

            return out
        
        def pad(self, x,    size): # NxCxHxW

            if type(size) in [list, tuple]:
                size_v = size[0]
                size_h = size[1]
            else:
                size_h = size_v = size

            north_pad = torch.flip(x[...,-1*size_v:,:], dims=[-2])
            south_pad = north_pad = torch.flip(x[...,:size_v,:], dims=[-2])
            north_pad = torch.roll(north_pad, shifts = 180, dims = [-1])
            south_pad = torch.roll(south_pad, shifts = 180, dims = [-1])
            x_padded = torch.cat([south_pad, x, north_pad], dim = -2 )
            east_pad = x_padded[...,:size_h]
            west_pad = x_padded[...,-1*size_h:]
            x_padded = torch.cat([west_pad, x_padded, east_pad], dim = -1 )
            
            return x_padded
        


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super().__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels * kernels_per_layer, output_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



    
class DoubleConv(nn.Module):
		"""(convolution => [BN] => ReLU) * 2"""
	
		def __init__(self, in_channels, out_channels, mid_channels=None):
				super().__init__()
				if not mid_channels:
						mid_channels = out_channels
				self.double_conv = nn.Sequential(
						nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
						nn.BatchNorm2d(mid_channels),
						nn.ReLU(inplace=True),
						nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
						nn.BatchNorm2d(out_channels),
						nn.ReLU(inplace=True)
				)
			
		def forward(self, x):
				return self.double_conv(x)
              
class Down(nn.Module):
		"""Downscaling with double conv then maxpool"""
	
		def __init__(self, in_channels, out_channels, pool_padding = 0):
				super().__init__()
				self.maxpool = nn.MaxPool2d(3,stride = 3, padding =  pool_padding)
				self.doubleconv = DoubleConv(in_channels, out_channels)
			
		def forward(self, x):
				x1 = self.doubleconv(x)
				x2 = self.maxpool(x1)
				return x2, x1

class InitialConv(nn.Module):
		def __init__(self, in_channels, out_channels):
				super().__init__()

				self.firstconv = nn.Sequential(
						nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1), 
						nn.BatchNorm2d(out_channels),
						nn.ReLU(inplace=True)
				)
		def forward(self, x):
				x1 = self.firstconv(x)
				return x1


class CNN_mean(nn.Module):
	
    
    def __init__( self,  n_channels_x=1 , dense_dims = [360,190,45],dropout_rate = None, dense_batch_normalization = False,  bilinear=False ):
        
        super().__init__()
        self.n_channels_x = n_channels_x
        self.bilinear = bilinear
    
        # input  (batch, n_channels_x, 180, 360)
        
        self.initial_conv = InitialConv(n_channels_x, 16)
        # downsampling:
        self.d1 = Down(16, 32)
        self.d2 = Down(32, 64)
        self.d3 = Down(64, 128)
        self.d4 = Down(128, 256)


        dense_dims = [256*2*4, *dense_dims]
        layers = []
        for i in range(len(dense_dims) - 1):
            layers.append(nn.Linear(dense_dims[i], dense_dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            if dense_batch_normalization:
                layers.append(nn.BatchNorm1d(encoder_dims[i + 1]))
        
        self.encoder = nn.Sequential(*layers)
        self.out = nn.Linear(dense_dims[-1] ,1)
            

    def forward(self, x):
    # input  (batch, n_channels_x, 40, 360)
        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x

        x1 = self.initial_conv(x_in)  # (batch, 16, 180, 360)

    # Downsampling
        x2, x2_bm = self.d1(x1)  # (batch, 32, 60, 120)
        x3, x3_bm = self.d2(x2)  # (batch, 64, 20, 40)
        x4, x4_bm = self.d3(x3)  # (batch, 128, 6, 13)
        x5, x5_bm = self.d4(x4)  # (batch, 256, 2, 4)
        
        x_out = self.encoder(x5.view(x5.size(0), -1))
        x_out = self.out(x_out)

        return x_out.unsqueeze(-1)
