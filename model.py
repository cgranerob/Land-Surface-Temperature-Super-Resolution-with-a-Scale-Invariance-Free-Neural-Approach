#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Python file to import. Contains:
    - The definition of the models and their building blocks.
    - The definition of some loss function used to train these models.

Example
-------
>>> import model as md 
-- or --
>>> from model import ModelB_2

Misc
----
@author: Romuald Ait Bachir
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Serf(nn.Module):
    """
    Description
    -----------
    Class implementing the Serf non linearity / activation function. It is
    supposed to:
        - Have better convergence properties than ReLU (Continuously Diff)
        - Allow more information to pass (No dying neuron)
        - Reach better performances than most layers
        - At some higher computational costs
    
    It was defined in:
    Serf: Towards better training of deep neural networks using log-Softplus ERror
    activation Function
    
    # It is not used but it is easily added using the activation dictionnary.
    
    Methods
    -------
    forward(x)
    Method executing the forward pass of a tensor x into the non-linearity.
        
    """
    def __init__(self):
        """
        Description
        -----------
        Constructor of the class Serf.
        
        """
        super(Serf, self).__init__()

    def forward(self, x):
        """
        Description
        -----------
        Forward process of the activation.
        
        f(x) = xerf(ln(1+exp(x)))

        Parameters
        ----------
        x : tensor
            Input tensor.
        
        Returns
        -------
        f_x: tensor
            Output tensor.
        """
        return x*torch.erf(torch.log(1+torch.exp(x)))

# Activation function dictionnary.
activation_functions = {
    "ReLU": torch.nn.ReLU(),
    "Serf": Serf()
    }


class DoubleConvolution(nn.Module):
    """
    Description
    -----------
    DoubleConvolution block used many times in the UNet and MRUNet architectures.
    
    Attributes
    ----------
    bloc: torch.nn.Sequential
        List of layers representing the DoubleConvolution.
    
    Methods
    -------
    forward(x)
    Method executing the forward pass of a tensor x of shape (Batch_size, in_channels, N, N) into the bloc.
        
    """
    def __init__(self, in_channels, out_channels, mid_channels = None, padding_mode = 'zeros', activation = 'ReLU'):
        """
        Description
        -----------
        Constructor of the class DoubleConvolution. 
        DoubleConv = (Conv2D -> BatchNorm2D -> ReLU)²
        
        Parameters
        ----------
        in_channels : int
            Second dimension of the input tensor. Number of channels (e.g. RGB = 3, NDVI = 1)
        out_channels : int 
            Second dimension of the output tensor. Shape: (Batch_size, out_channels, N, N)
        mid_channels : int, optional
            Intermediate optional value. Second dimension of the tensor present before the 
            second convolution. By default, this value equates to out_channels.
        padding_mode : str, optional
            Choice of the type of padding used. Choose between: 'zeros', 'reflect', 
            'replicate', 'circular'. Default value: 'zeros'.
        activation : str
            Choice of the activation function. The activation function should be inside
            activation_functions. Default value: "ReLU"
        """
        super(DoubleConvolution, self).__init__()
        
        # In the original paper, the convolutions are not padded, which results
        # in a loss of resolution inside the segmented output.
        if not mid_channels: 
            mid_channels = out_channels
        
        # ReLU is used as non-linearity, BatchNorm2D as regularization. The 
        # convolutions are unbiased since BatchNorm2D already takes care of it.
        self.bloc = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = mid_channels, bias = False, kernel_size = 3, stride = 1, padding = 1, padding_mode = padding_mode),
            nn.BatchNorm2d(mid_channels),
            activation_functions[activation],
            nn.Conv2d(in_channels = mid_channels, out_channels = out_channels, bias = False, kernel_size = 3, stride = 1, padding = 1, padding_mode = padding_mode),
            nn.BatchNorm2d(out_channels),
            activation_functions[activation]
            )

    def forward(self, x):
        """
        Description
        -----------
        Forward process of the bloc. y = ReLU(BN2(Conv2D2(ReLU(BN1(Conv2D1(x)))))).

        Parameters
        ----------
        x : tensor
            Input tensor. Shape: (Batch_size, in_channels, height, width)
            
        Returns
        -------
        tensor
            Output tensor. Shape: (Batch_size, out_channels, height, width).
        """
        return self.bloc(x)
    
class UpBlock(nn.Module):
    """
    Description
    -----------
    Implementation of the Upsampling block present in the right side 
    of UNet.
    
    Attributes
    ----------
    up : torch.nn.Module
        UpSampling Layer. Can be either an Upsampling by convolution or by bilinear upsampling.
    convbloc : torch.nn.Module
        DoubleConv layer.
        
    Methods
    -------
    forward(x)
    Method executing the forward pass of a tensor into the block.
    """
    
    def __init__(self, in_channels, out_channels, bilinear = False, padding_mode = 'zeros', activation = 'ReLU'):
        """
        Description
        -----------
        Constructor of the class ResidualConnection.
        
        Parameters
        ----------
        in_channels : int
            Second dimension of the input tensor's shape.
        out_channels : int
            Second dimension of the output tensor's shape.
        bilinear : bool, optional
            UpSampling method chosen. Default is False (ConvTranspose2D)
        padding_mode : str, optional
            Choice of the type of padding. Choose between 1, "reflect" and "replicate". 
            The default is 'zeros'.
        activation : str
            Choice of the activation function. The activation function should be inside
            activation_functions. Default value: "ReLU"
        """
        
        super(UpBlock, self).__init__()
        
        if bilinear:
            # Use a bilinear interpolation to do a x2 in resolution (x4 since bidimensionnal)
            self.up = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
            self.convbloc = DoubleConvolution(in_channels, out_channels, in_channels//2, padding_mode, activation = activation)
            
        else:
            # Use a convolution filter to do the x2.
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size = 2, stride = 2)
            self.convbloc = DoubleConvolution(in_channels, out_channels, padding_mode = padding_mode, activation = activation)
            
    def forward(self, x, x_os):
        """
        Description
        -----------
        Forward process of the bloc. x <-DoubleConv(Cat(UpSample(x), x_connection)).
        
        Note: The shape of the input vector is different /2 when using Upsample due to 
        ConvTranspose2D already having an halfed output dimension.
        
        Parameters
        ----------
        x : tensor
            Input tensor. 

        Returns
        -------
        tensor
            Output tensor.
        """
        # Passing through the upsampling layer
        x = self.up(x)
        
        # Measuring the error between the main connection and the skip connection.
        diffX = x_os.shape[-1] - x.shape[-1]
        diffY = x_os.shape[-2] - x.shape[-2]
        
        ## Padding or cropping if the shape is different. Only happens if there is
        ## no padding in the convolutions like in the 1st article of UNet.
        # x_os = x_os[:,:,diffY//2 : x_os.shape[-2] - diffY//2, diffX//2 : x_os.shape[-1] - diffX//2]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x, x_os], dim = 1)
        return self.convbloc(x)


class ResidualConnection(nn.Module):
    """
    Description
    -----------
    Implementation of a ResidualConnection introduced in ResNet.
    
    x -- |--DoubleConv --| -- out
         |---------------|
    
    Attributes
    ----------
    doubleconv : torch.nn.Module
        DoubleConv bloc.
        
    Methods
    -------
    forward(x)
        Forward pass of the Residual Connection.
    """
    
    def __init__(self, in_channels, out_channels, padding_mode = 'zeros', activation = 'ReLU'):
        """
        Description
        -----------
        Constructor of the class ResidualConnection.
        
        Parameters
        ----------
        in_channels : int
            Second dimension of the input tensor's shape.
        out_channels : int
            Second dimension of the output tensor's shape.
        padding_mode : str, optional
            Choice of the type of padding. Choose between 1, "reflect" and "replicate".
            The default is 'zeros'.
        activation : str
            Choice of the activation function. The activation function should be inside
            activation_functions. Default value: "ReLU"
        """
        
        super(ResidualConnection, self).__init__()
        self.doubleconv = DoubleConvolution(in_channels, out_channels, padding_mode = padding_mode, activation = activation)
        
    def forward(self, x):
        """
        Description
        -----------
        Forward process of the bloc. x <-x + DoubleConv(x).

        Parameters
        ----------
        x : tensor
            Input tensor. 

        Returns
        -------
        tensor
            Output tensor.
        """
        
        x_out = self.doubleconv(x)
        return torch.add(x, x_out)


class ResBridgeBlock(nn.Module):
    """
    Description
    -----------
    Implementation of the bridge block used in MultiResidualUNet.
    
    It mainly consists in:
    x --> Residual --|-- Conv2D - BN - ReLU - Conv2D ---+ -----> x_out
                     |----------------------------------|
    
    Attributes
    ----------
    block : torch.nn.Module
        Block described in the description. Doesn't incorporate the skip connection.
    
    Methods
    -------
    forward(x)
        Forward pass of the ResBridgeBlock.
    """
    
    def __init__(self, in_channels, padding_mode = 'zeros', activation = 'ReLU'):
        """
        Description
        -----------
        Constructor of the class ResBridgeBlock.
        
        Parameters
        ----------
        in_channels : int
            Second dimension of the input tensor's shape.
        padding_mode : str, optional
            Choice of the type of padding. Choose between 1, "reflect" and "replicate". 
            The default is 'zeros'.
        activation : str
            Choice of the activation function. The activation function should be inside
            activation_functions. Default value: "ReLU"
        """
        
        super(ResBridgeBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, bias = False, padding = 1, padding_mode = padding_mode),
            nn.BatchNorm2d(in_channels),
            activation_functions[activation],
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, bias = False, padding = 1, padding_mode = padding_mode),
            nn.BatchNorm2d(in_channels)) 

    def forward(self, x):
        """
        Description
        -----------
        Forward process of the bloc. x <-x + Bloc(x).

        Parameters
        ----------
        x : tensor
            Input tensor. 

        Returns
        -------
        tensor
            Output tensor.
        """
        
        return x + self.block(x)


class DownBlock(nn.Module):
    """
    Description
    -----------
    Implementation of a DownBlock, representing all the layers at a precise floor
    in the left side of the MultiResidualUNet architecture.
    
    It mainly consists in:
        DownConv(k = 2x2, s = 2) -> Residual-DoubleConv --> MonoConv 
    (MonoConv = 1/2 layers of DoubleConv)
    
    Attributes
    ----------
    downsampling : torch.nn.Module
        Downsampling layer (not a pooling but a convolution with stride = kernel_size = 2)
    resblock : torch.nn.Module
        Block containing the residual connection + the doubleconv
    lastconv : torch.nn.Module
        MonoConv: Conv2D + BN + ReLU
    
    Methods
    -------
    forward(x)
        Forward pass of the block.
    """
    
    def __init__(self, in_channels, out_channels, padding_mode = 'zeros', activation = 'ReLU'):
        """
        Description
        -----------
        Constructor of the DownBlock class.

        Parameters
        ----------
        in_channels : int
            Second dimension of the input tensor's shape.
        out_channels : int 
            Second dimension of the output tensor's shape.
        padding_mode : str, optional
            Type of padding to do inside the Conv2D layers. The default is 'zeros'.
        activation : str
            Choice of the activation function. The activation function should be inside
            activation_functions. Default value: "ReLU"
        """
        
        super(DownBlock, self).__init__()
        self.downsampling = nn.Conv2d(in_channels, in_channels, kernel_size = 2, stride = 2)
        self.resblock = ResidualConnection(in_channels, in_channels, padding_mode = padding_mode, activation = activation)
        self.lastconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, bias = False, padding = 1, padding_mode = padding_mode),
            nn.BatchNorm2d(out_channels),
            activation_functions[activation])
        
    def forward(self, x):
        """
        Description
        -----------
        Forward process of the bloc. x <- Bloc(x).

        Parameters
        ----------
        x : tensor
            Input tensor. 

        Returns
        -------
        x : tensor
            Output tensor.
        """
        
        x = self.downsampling(x)
        x = self.resblock(x)
        x = self.lastconv(x)
        return  x
    

class DownBlock_pool(nn.Module):
    """
    Description
    -----------
    Implementation of a DownBlock, representing all the layers at a precise floor
    in the left side of the MultiResidualUNet architecture.
    
    It mainly consists in:
        DownConv(k = 2x2, s = 2) -> Residual-DoubleConv --> MonoConv 
    (MonoConv = 1/2 layers of DoubleConv)
    
    Attributes
    ----------
    downsampling : torch.nn.Module
        Downsampling layer (not a pooling but a convolution with stride = kernel_size = 2)
    resblock : torch.nn.Module
        Block containing the residual connection + the doubleconv
    lastconv : torch.nn.Module
        MonoConv: Conv2D + BN + ReLU
    
    Methods
    -------
    forward(x)
        Forward pass of the block.
    """
    
    def __init__(self, in_channels, out_channels, padding_mode = 'zeros', activation = 'ReLU'):
        """
        Description
        -----------
        Constructor of the DownBlock class.

        Parameters
        ----------
        in_channels : int
            Second dimension of the input tensor's shape.
        out_channels : int 
            Second dimension of the output tensor's shape.
        padding_mode : str, optional
            Type of padding to do inside the Conv2D layers. The default is 'zeros'.
        activation : str
            Choice of the activation function. The activation function should be inside
            activation_functions. Default value: "ReLU"
        """
        
        super(DownBlock_pool, self).__init__()
        self.downsampling = torch.nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.resblock = ResidualConnection(in_channels, in_channels, padding_mode = padding_mode, activation = activation)
        self.lastconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, bias = False, padding = 1, padding_mode = padding_mode),
            nn.BatchNorm2d(out_channels),
            activation_functions[activation])
        
    def forward(self, x):
        """
        Description
        -----------
        Forward process of the bloc. x <- Bloc(x).

        Parameters
        ----------
        x : tensor
            Input tensor. 

        Returns
        -------
        x : tensor
            Output tensor.
        """
        
        x = self.downsampling(x)
        x = self.resblock(x)
        x = self.lastconv(x)
        return  x

class ModelB_2(nn.Module):
    """
    Implementation of a smaller ModelB.
    
    This model B takes The NDVI and the LST_bicubic concatenated as input.
    
    Attributes
    ----------
    in_channels : int
        Second dimension of the input tensor's shape. Number of channels (e.g. RGB = 3, NDVI = 1)
    downchannels : list of int
        "Evolution" of the number of channels at each floor of the UNet.
    padding: str
        Type of padding used inside every Conv2D layer in the model. The default is replicate.
    upfactor: int
        Necessary factor to process used when changing from ConvTranspose2d to Upsample as Upsampling layer.
    bridge : int
        Number of ResBridgeBlocks making the bridge.
    blocs: torch.nn.Module
        Many attributes under this name representing every bloc used inside the model.
        (inbloc, dbx, bridge, ubx, outlay).
        Note: The weights of the left side are shared which could maybe lead to 
        a loss of performance...

    Methods
    -------
    forward(x_lst, x_ndvi)
    Method executing the forward pass of a tensor x_lst and x_ndvi of shape (Batch_size, in_channels, height, width) into the network.
    """
    
    def __init__(self, in_channels, downchannels = [16,32,64,128], padding_mode = 'replicate', activation = 'ReLU', bilinear = True, n_bridge_blocks = 1):
        """
        Description
        -----------
        Constructor of the class ModelB.

        Parameters
        ----------
        in_channels : int
            Second dimension of the input tensor's shape. Should maybe be modified into a tuple?
        downchannels : list of int, optional
            "Evolution" of the number of channels at each floor of the UNet. The default is [16,32,64,128].
        padding_mode : str, optional
            Type of padding used inside every Conv2D layer in the model. The default is "replicate".
        bilinear : bool, optional
            Choice of the kind of UpSampling layer. True = Bilinear, False = ConvTranspose2D. The default is True.
            Note: In general, the results when putting this parameter to False is less satisfying.
        n_bridge_blocks : int, optional
            Number of ResBridgeBlocks making the bridge. The default is 1.
        weight_factor : float, optional
            Weight to give to all the NDVI connections. The default is 0.8.
        """
        
        super(ModelB_2, self).__init__()
        self.in_channels = in_channels
        self.downchannels = downchannels
        self.padding = padding_mode
        self.activation = activation
        self.upfactor = 2 if bilinear else 1
        self.bridge = n_bridge_blocks
        
        # LST
        # Left side
        self.inbloc = DoubleConvolution(self.in_channels, self.downchannels[0],padding_mode = self.padding, activation = self.activation)
        self.db1 = DownBlock_pool(self.downchannels[0], self.downchannels[1], self.padding, activation = self.activation)
        self.db2 = DownBlock_pool(self.downchannels[1], self.downchannels[2], self.padding, activation = self.activation)
        self.db3 = DownBlock_pool(self.downchannels[2], self.downchannels[3]//self.upfactor, self.padding, activation = self.activation)

        # Right side
        self.ub1 = UpBlock(self.downchannels[3], self.downchannels[2]//self.upfactor, bilinear, self.padding, activation = self.activation)
        self.ub2 = UpBlock(self.downchannels[2], self.downchannels[1]//self.upfactor, bilinear, self.padding, activation = self.activation)
        self.ub3 = UpBlock(self.downchannels[1], self.downchannels[0], bilinear, self.padding, activation = self.activation)
        self.outlay = nn.Conv2d(self.downchannels[0], 1, kernel_size=3, stride = 1, padding = 1, padding_mode = self.padding)

        
    def forward(self, x_lst_ndvi):
        """
        Description
        -----------
        Forward process of the neural net. y_pred = Net(x_lst_ndvi).

        Parameters
        ----------
        x_lst : tensor
            Input Bicubic_LST tensor. Shape: (Batch_size, 2, 256, 256)

        Returns
        -------
        x : tensor
            y_pred. Reconstructed Super Resolved LST. Shape: (Batch_size, 1, 256, 256).
        """
        x_os = []
        
        ## Input Layer + Encoder Block + Bridge for the LST.
        x_lst_ndvi = self.inbloc(x_lst_ndvi)
        x_os.append(x_lst_ndvi)
        x_lst_ndvi = self.db1(x_lst_ndvi)
        x_os.append(x_lst_ndvi)
        x_lst_ndvi = self.db2(x_lst_ndvi)
        x_os.append(x_lst_ndvi)
        x_lst_ndvi = self.db3(x_lst_ndvi)
        x_os.append(x_lst_ndvi)
        

        ## DECODER BLOCK
        x_lst_ndvi = self.ub1(x_lst_ndvi, x_os[-2])
        x_lst_ndvi = self.ub2(x_lst_ndvi, x_os[-3])
        x_lst_ndvi = self.ub3(x_lst_ndvi, x_os[-4])
        
        # Output Layer
        x_lst_ndvi = self.outlay(x_lst_ndvi)
        
        return x_lst_ndvi

# # Just looking at the summary (number of parameters, size of the forward pass, ...)
# from torchinfo import summary

# batch_size = 1

# t_a = torch.randn((batch_size, 1, 256, 256))
# t_b = torch.randn((batch_size, 1, 256, 256))
# modelB = ModelB_2(in_channels = 2, 
#                 downchannels = [16,32,64,128],
#                 padding_mode = 'replicate',
#                 activation = 'ReLU',
#                 bilinear = 1,
#                 n_bridge_blocks = 0) 

# t = torch.cat((t_a, t_b), dim = 1)

# summary(modelB, input_size=(batch_size, 2, 256, 256))
