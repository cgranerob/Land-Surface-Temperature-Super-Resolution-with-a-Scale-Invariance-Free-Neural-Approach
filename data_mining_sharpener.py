# -*- coding: utf-8 -*-

#%% Here is the part of the code from pyDMS made by Guzinski and Nieto that we use, 
# if you want the full functionality, go to https://github.com/radosuav/pyDMS

import math
import os
from multiprocessing import Pool

import numpy as np
from osgeo import gdal
from sklearn import tree, linear_model, ensemble, preprocessing
import sklearn.neural_network as ann_sklearn

import scipy.ndimage as ndi
from numba import njit, stencil

from pyproj import Proj, Transformer


def openRaster(raster):
    closeOnExit = False
    try:
        raster.GetProjection()
        openRaster = raster
    except AttributeError:
        openRaster = gdal.Open(str(raster))
        closeOnExit = True
        print("Case2")
    return openRaster, closeOnExit


def getRasterInfo(raster):
    r, closeOnExit = openRaster(raster)
    proj = r.GetProjection()
    gt = r.GetGeoTransform()
    sizeX = r.RasterXSize
    sizeY = r.RasterYSize
    extent = [gt[0], gt[3]+gt[5]*sizeY, gt[0]+gt[1]*sizeX, gt[3]]
    bands = r.RasterCount
    if closeOnExit:
        r = None
    return proj, gt, sizeX, sizeY, extent, bands


def resampleWithGdalWarp(srcFile, templateFile, outFile="", outFormat="MEM",
                         resampleAlg="average"):
    # Get template projection, extent and resolution
    proj, gt, sizeX, sizeY, extent, _ = getRasterInfo(templateFile)

    # Resample with GDAL warp
    outDs = gdal.Warp(str(outFile),
                      openRaster(srcFile)[0],
                      format=outFormat,
                      dstSRS=proj,
                      xRes=gt[1],
                      yRes=gt[5],
                      outputBounds=extent,
                      resampleAlg=resampleAlg)

    return outDs


def point2pix(point, gt, upperBound=False):
    mx = point[0]
    my = point[1]
    if not upperBound:
        px = math.floor((mx - gt[0]) / gt[1])  # x pixel
        py = math.floor((my - gt[3]) / gt[5])  # y pixel
    else:
        px = math.ceil((mx - gt[0]) / gt[1])  # x pixel
        py = math.ceil((my - gt[3]) / gt[5])  # y pixel
    return [int(px), int(py)]


def pix2point(pix, gt):
    px = pix[0]
    py = pix[1]
    mx = px*gt[1] + gt[0]  # x coordinate
    my = py*gt[5] + gt[3]  # y coordinate
    return [mx, my]


# save the data to geotiff or memory
def saveImg(data, geotransform, proj, outPath, noDataValue=None, fieldNames=[]):
    outPath = str(outPath)

    # Save to memory first
    is_netCDF = False
    memDriver = gdal.GetDriverByName("MEM")
    shape = data.shape
    if len(shape) > 2:
        ds = memDriver.Create("MEM", shape[1], shape[0], shape[2], gdal.GDT_Float32)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        for i in range(shape[2]):
            ds.GetRasterBand(i+1).WriteArray(data[:, :, i])
    else:
        ds = memDriver.Create("MEM", shape[1], shape[0], 1, gdal.GDT_Float32)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        ds.GetRasterBand(1).WriteArray(data)

    # Save to file if required
    if outPath == "MEM":
        if noDataValue is None:
            noDataValue = np.nan
        ds.GetRasterBand(1).SetNoDataValue(noDataValue)
    else:
        # If the output file has .nc extension then save it as netCDF,
        # otherwise assume that the output should be a GeoTIFF (COG)
        ext = os.path.splitext(outPath)[1]
        if ext.lower() == ".nc":
            fileFormat = "netCDF"
            driverOpt = ["FORMAT=NC2"]
            is_netCDF = True
        else:
            fileFormat = "COG"
            driverOpt = ['COMPRESS=DEFLATE', 'PREDICTOR=YES', 'BIGTIFF=IF_SAFER']
        out_ds = gdal.Translate(outPath, ds, format=fileFormat, creationOptions=driverOpt,
                                noData=noDataValue, stats=True)
        # If GDAL driers for other formats do not exist then default to GeoTiff
        if out_ds is None:
            print("Warning: Selected GDAL driver is not supported! Saving as GeoTiff!")
            fileFormat = "GTiff"
            driverOpt = ['COMPRESS=DEFLATE', 'PREDICTOR=1', 'BIGTIFF=IF_SAFER']
            is_netCDF = False
            ds = gdal.Translate(outPath, ds, format=fileFormat, creationOptions=driverOpt,
                                noData=noDataValue, stats=True)
        else:
            ds = out_ds

        # In case of netCDF format use netCDF4 module to assign proper names
        # to variables (GDAL can't do this). Also it seems that GDAL has
        # problems assigning projection to all the bands so fix that.
        if is_netCDF and fieldNames:
            from netCDF4 import Dataset
            ds = None
            ds = Dataset(outPath, 'a')
            grid_mapping = ds["Band1"].grid_mapping
            for i, field in enumerate(fieldNames):
                ds.renameVariable("Band"+str(i+1), field)
                ds[field].grid_mapping = grid_mapping
            ds.close()
            ds = gdal.Open('NETCDF:"'+outPath+'":'+fieldNames[0])

    print('Saved ' + outPath)

    return ds


def binomialSmoother(data):
    def filterFunction(footprint):
        weight = [1, 2, 1, 2, 4, 2, 1, 2, 1]
        # Don't smooth land and invalid pixels
        if np.isnan(footprint[4]):
            return footprint[4]

        footprintSum = 0
        weightSum = 0
        for i in range(len(weight)):
            # Don't use land and invalid pixels in smoothing of other pixels
            if not np.isnan(footprint[i]):
                footprintSum = footprintSum + weight[i] * footprint[i]
                weightSum = weightSum + weight[i]
        try:
            ans = footprintSum/weightSum
        except ZeroDivisionError:
            ans = footprint[4]
        return ans

    smoothedData = ndi.filters.generic_filter(data, filterFunction, 3)

    return smoothedData


def appendNpArray(array, data, axis=None):
    if array is None or array.size == 0:
        array = data
    else:
        array = np.append(array, data, axis=axis)
    return array


# Reproject and subset the given low resolution datasets to high resolution
# scene projection and extent
def reprojectSubsetLowResScene(highResScene, lowResScene, resampleAlg=gdal.GRA_Bilinear):

    # Read the required metadata
    proj_HR, gt_HR, xsize_HR, ysize_HR, extent = getRasterInfo(highResScene)[0:5]
    proj_LR, gt_LR, xsize_LR, ysize_LR = getRasterInfo(lowResScene)[0:4]

    # Transform "middle pixel" and "middle pixel + 1" between LR and HR projections
    # to obtain LR resolution in HR projection. This method can handle different
    # x and y resolution
    midPix = [int(xsize_LR/2), int(ysize_LR/2)]
    midPix_2 = [midPix[0]+1, midPix[1]+1]
    midPoint = pix2point(midPix, gt_LR)
    midPoint_2 = pix2point(midPix_2, gt_LR)
    transformer = Transformer.from_proj(Proj(proj_LR), Proj(proj_HR), always_xy=True)
    x1, y1 = transformer.transform(midPoint[0], midPoint[1])
    x2, y2 = transformer.transform(midPoint_2[0], midPoint_2[1])
    xRes_LR_proj = x2 - x1
    yRes_LR_proj = y2 - y1
    # Now do the same with UL pixel to obtain low resolution geotransform in
    # the new projection
    UL_x, UL_y = transformer.transform(gt_LR[0], gt_LR[3])
    gt_LR = [UL_x, xRes_LR_proj, 0, UL_y, 0, yRes_LR_proj]

    # Now subset to high resolution scene extent while not shifting pixels
    UL = pix2point(point2pix([extent[0], extent[3]], gt_LR, upperBound=False), gt_LR)
    BR = pix2point(point2pix([extent[2], extent[1]], gt_LR, upperBound=True), gt_LR)
    out = gdal.Warp("",
                    openRaster(lowResScene)[0],
                    format="MEM",
                    dstSRS=proj_HR,
                    resampleAlg=gdal.GRA_NearestNeighbour,
                    xRes=gt_LR[1],
                    yRes=gt_LR[5],
                    outputBounds=[UL[0], BR[1], BR[0], UL[1]])

    return out


# Resample high res scene to low res pixel while extracting homogeneity
# statistics. It is assumed that both scenes have the same projection and extent.
def resampleHighResToLowRes(highResScene, lowResScene):

    gt_HR = getRasterInfo(highResScene)[1]
    bands_HR = getRasterInfo(highResScene)[5]
    gt_LR, xSize_LR, ySize_LR = getRasterInfo(lowResScene)[1:4]
    xRes_HR = gt_HR[1]
    yRes_HR = abs(gt_HR[5])
    xRes_LR = gt_LR[1]
    yRes_LR = gt_LR[5]

    aggregatedMean = np.zeros((ySize_LR,
                               xSize_LR,
                               bands_HR))
    aggregatedStd = np.zeros(aggregatedMean.shape)

    # Go through all the high res bands and calculate mean and standard
    # deviation when aggregated to the low resolution
    highRes, close = openRaster(highResScene)
    for band in range(bands_HR):
        bandData_HR = highRes.GetRasterBand(band+1).ReadAsArray().astype(np.float32)
        nodataValue = highRes.GetRasterBand(1).GetNoDataValue()
        bandData_HR[bandData_HR == nodataValue] = np.nan
        aggregatedMean[:, :, band], aggregatedStd[:, :, band] =\
            _resampleHighResToLowRes(bandData_HR, ySize_LR, yRes_LR, yRes_HR, xSize_LR, xRes_LR,
                                     xRes_HR, gt_HR, gt_LR)
    if close:
        highRes = None
    return aggregatedMean, aggregatedStd


@njit
def _resampleHighResToLowRes(bandData_HR, ySize_LR, yRes_LR, yRes_HR, xSize_LR, xRes_LR, xRes_HR,
                             gt_HR, gt_LR):
    aggregatedMean = np.zeros((ySize_LR, xSize_LR))
    aggregatedStd = np.zeros((ySize_LR, xSize_LR))
    for yPix_LR in range(ySize_LR):
        yPos_LR_min = gt_LR[3] + yPix_LR*yRes_LR
        yPix_HR_min = int(round(max(0, gt_HR[3] - yPos_LR_min) / yRes_HR))
        yPix_HR_max = int(round(max(0, gt_HR[3] - (yPos_LR_min + yRes_LR)) / yRes_HR))
        for xPix_LR in range(xSize_LR):
            xPos_LR_min = gt_LR[0] + xPix_LR*xRes_LR
            xPix_HR_min = int(round(max(0, xPos_LR_min - gt_HR[0]) / xRes_HR))
            xPix_HR_max = int(round(max(0, xPos_LR_min + xRes_LR - gt_HR[0]) / xRes_HR))
            aggregatedMean[yPix_LR, xPix_LR] =\
                np.nanmean(bandData_HR[yPix_HR_min:yPix_HR_max, xPix_HR_min:xPix_HR_max])
            aggregatedStd[yPix_LR, xPix_LR] =\
                np.nanstd(bandData_HR[yPix_HR_min:yPix_HR_max, xPix_HR_min:xPix_HR_max])

    return aggregatedMean, aggregatedStd


def resampleLowResToHighRes(lowResScene, highResScene, resampleAlg="cubic"):

    lowResScene_resampled = resampleWithGdalWarp(lowResScene, highResScene,
                                                 resampleAlg=resampleAlg)

    data = lowResScene_resampled.GetRasterBand(1).ReadAsArray()
    # Sometimes there can be 1 HR pixel NaN border arond LR invalid pixels due to resampling.
    # Fuction below fixes this. Image border pixels are excluded due to numba stencil
    # limitations.
    data[1:-1, 1:-1] = removeEdgeNaNs(data)[1:-1, 1:-1]

    # The resampled array might be slightly smaller then the high-res because
    # of the subsetting of the low resolution scene. In that case just pad
    # the missing values with neighbours.
    highResShape = (getRasterInfo(highResScene)[3], getRasterInfo(highResScene)[2])
    if highResShape != data.shape:
        data = np.pad(data,
                      ((0, highResShape[0]-data.shape[0]), (0, highResShape[1]-data.shape[1])),
                      "edge")
    return data


@stencil(cval=1.0)
def removeEdgeNaNs(a):
    if np.isnan(a[0, 0]) and (not np.isnan(a[-1, 0]) or not np.isnan(a[1, 0]) or
                              not np.isnan(a[0, -1]) or not np.isnan(a[0, 1])):
        return np.nanmean(np.array([a[-1, 0], a[1, 0], a[0, -1], a[0, 1]]))
    else:
        return a[0, 0]


REG_sknn_ann = 0
REG_sklearn_ann = 1


class DecisionTreeRegressorWithLinearLeafRegression(tree.DecisionTreeRegressor):
    ''' Decision tree regressor with added linear (ridge) regression
    for all the data points falling within each decision tree leaf node.

    Parameters
    ----------
    linearRegressionExtrapolationRatio: float (optional, default: 0.25)
        A limit on extrapolation allowed in the per-leaf linear regressions.
        The ratio is multiplied by the range of values present in each leaves'
        training dataset and added (substracted) to the maxiumum (minimum)
        value.

    decisionTreeRegressorOpt: dictionary (optional, default: {})
        Options to pass to DecisionTreeRegressor constructor. See
        http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        for possibilities.

    Returns
    -------
    None
    '''
    def __init__(self, linearRegressionExtrapolationRatio=0.25, decisionTreeRegressorOpt={}):
        super(DecisionTreeRegressorWithLinearLeafRegression, self).__init__(**decisionTreeRegressorOpt)
        self.decisionTreeRegressorOpt = decisionTreeRegressorOpt
        self.leafParameters = {}
        self.linearRegressionExtrapolationRatio = linearRegressionExtrapolationRatio

    def fit(self, X, y, sample_weight, fitOpt={}):
        ''' Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X: array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            dtype=np.float32 and if a sparse matrix is provided to a sparse
            csc_matrix.

        y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (real numbers). Use dtype=np.float64 and
            order='C' for maximum efficiency.

        sample_weight: array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        fitOpt: dictionary (optional, default: {})
            Options to pass to DecisionTreeRegressor fit function. See
            http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
            for possibilities.

        Returns
        -------
        Self
        '''

        # Fit a normal regression tree
        super(DecisionTreeRegressorWithLinearLeafRegression, self).fit(X, y, sample_weight,
                                                                       **fitOpt)

        # Create a linear regression for all input points which fall into
        # one output leaf
        predictedValues = super(DecisionTreeRegressorWithLinearLeafRegression, self).predict(X)
        leafValues = np.unique(predictedValues)
        for value in leafValues:
            ind = predictedValues == value
            leafLinearRegrsion = linear_model.Ridge()
            leafLinearRegrsion.fit(X[ind, :], y[ind])
            self.leafParameters[value] = {"linearRegression": leafLinearRegrsion,
                                          "max": np.max(y[ind]),
                                          "min": np.min(y[ind])}

        return self

    def predict(self, X, predictOpt={}):
        ''' Predict class or regression value for X.

        Parameters
        ----------
        X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            dtype=np.float32 and if a sparse matrix is provided to a sparse
            csr_matrix.

        predictOpt: dictionary (optional, default: {})
            Options to pass to DecisionTreeRegressor predict function. See
            http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
            for possibilities.

        Returns
        -------
        y: array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        '''

        # Do normal regression tree prediction
        y = super(DecisionTreeRegressorWithLinearLeafRegression, self).predict(X, **predictOpt)

        # And also apply per-leaf linear regression
        for leafValue in self.leafParameters.keys():
            ind = y == leafValue
            if X[ind, :].size > 0:
                y[ind] = self.leafParameters[leafValue]["linearRegression"].predict(X[ind, :])
                # Limit extrapolation
                extrapolationRange = self.linearRegressionExtrapolationRatio * (
                                        self.leafParameters[leafValue]["max"] -
                                        self.leafParameters[leafValue]["min"])
                y[ind] = np.maximum(y[ind],
                                    self.leafParameters[leafValue]["min"] - extrapolationRange)
                y[ind] = np.minimum(y[ind],
                                    self.leafParameters[leafValue]["max"] + extrapolationRange)

        return y


class DecisionTreeSharpener(object):
    ''' Decision tree based sharpening (disaggregation) of low-resolution
    images using high-resolution images. The implementation is mostly based on [Gao2012].

    Decision tree based regressor is trained with high-resolution data resampled to
    low resolution and low-resolution data and then applied
    directly to high-resolution data to obtain high-resolution representation
    of the low-resolution data.

    The implementation includes selecting training data based on homogeneity
    statistics and using the homogeneity as weight factor ([Gao2012], section 2.2),
    performing linear regression with samples located within each regression
    tree leaf node ([Gao2012], section 2.1), using an ensemble of regression trees
    ([Gao2012], section 2.1), performing local (moving window) and global regression and
    combining them based on residuals ([Gao2012] section 2.3) and performing residual
    analysis and bias correction ([Gao2012], section 2.4)


    Parameters
    ----------
    highResFiles: list of strings
        A list of file paths to high-resolution images to be used during the
        training of the sharpener.

    lowResFiles: list of strings
        A list of file paths to low-resolution images to be used during the
        training of the sharpener. There must be one low-resolution image
        for each high-resolution image.

    lowResQualityFiles: list of strings (optional, default: [])
        A list of file paths to low-resolution quality images to be used to
        mask out low-quality low-resolution pixels during training. If provided
        there must be one quality image for each low-resolution image.

    lowResGoodQualityFlags: list of integers (optional, default: [])
        A list of values indicating which pixel values in the low-resolution
        quality images should be considered as good quality.

    cvHomogeneityThreshold: float (optional, default: 0)
        A threshold of coeficient of variation below which high-resolution
        pixels resampled to low-resolution are considered homogeneous. Non-homogenous pixels
        get extra penalized during the training of the disaggregator. If threshold is 0 or
        negative then it is set automatically such that 80% of pixels are below
        it.

    movingWindowSize: integer (optional, default: 0)
        The size of local regression moving window in low-resolution pixels. If
        set to 0 then only global regression is performed.

    minimumSampleNumber: integer (optional, default: 10)
        The number of samples requried to train a regression model. Applicable both to local and
        global regressions.

    disaggregatingTemperature: boolean (optional, default: False)
        Flag indicating whether the parameter to be disaggregated is
        temperature (e.g. land surface temperature). If that is the case then
        at some points it needs to be converted into radiance. This is becasue
        sensors measure energy, not temperature, plus radiance is the physical
        measurements it makes sense to average, while radiometric temperature
        behaviour is not linear.

    perLeafLinearRegression: boolean (optional, default: True)
        Flag indicating if linear regression should be performed on all data
        points falling within each regression tree leaf node.

    linearRegressionExtrapolationRatio: float (optional, default: 0.25)
        A limit on extrapolation allowed in the per-leaf linear regressions.
        The ratio is multiplied by the range of values present in each leaves'
        training dataset and added (substracted) to the maxiumum (minimum)
        value.

    regressorOpt: dictionary (optional, default: {})
        Options to pass to DecisionTreeRegressor constructor See
        http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        for possibilities. Note that max_leaf_nodes and min_samples_leaf
        parameters will beoverwritten in the code.

    baggingRegressorOpt: dictionary (optional, default: {})
        Options to pass to BaggingRegressor constructor. See
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
        for possibilities.

    Returns
    -------
    None


    References
    ----------
    .. [Gao2012] Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data
       Mining Approach for Sharpening Thermal Satellite Imagery over Land.
       Remote Sensing, 4(11), 3287-3319. https://doi.org/10.3390/rs4113287
    '''
    def __init__(self,
                 highResFiles,
                 lowResFiles,
                 lowResQualityFiles=[],
                 lowResGoodQualityFlags=[],
                 cvHomogeneityThreshold=0,
                 movingWindowSize=0,
                 minimumSampleNumber=10,
                 disaggregatingTemperature=False,
                 perLeafLinearRegression=True,
                 linearRegressionExtrapolationRatio=0.25,
                 regressorOpt={},
                 baggingRegressorOpt={}):

        self.highResFiles = highResFiles
        self.lowResFiles = lowResFiles
        self.lowResQualityFiles = lowResQualityFiles
        self.lowResGoodQualityFlags = lowResGoodQualityFlags

        if len(self.highResFiles) != len(self.lowResFiles):
            print("There must be a matching high resolution file for each low resolution file")
            raise IOError

        if len(self.lowResQualityFiles) == 0 or \
           (len(self.lowResQualityFiles) == 1 and self.lowResQualityFiles[0] == ""):
            self.useQuality_LR = False
        else:
            self.useQuality_LR = True

        if self.useQuality_LR and len(self.lowResQualityFiles) != len(self.lowResFiles):
            print("The number of quality files must be 0 or the same as number of low " +
                  "resolution files")
            raise IOError

        self.cvHomogeneityThreshold = cvHomogeneityThreshold
        # If threshold is 0 or negative then it is set automatically such that
        # 80% of pixels are below it.
        if self.cvHomogeneityThreshold <= 0:
            self.autoAdjustCvThreshold = True
            self.precentileThreshold = 80
        else:
            self.autoAdjustCvThreshold = False

        # Moving window size in low resolution pixels
        self.movingWindowSize = float(movingWindowSize)
        # The extension (on each side) by which sampling window size is larger
        # then prediction window size (see section 2.3 of Gao paper)
        self.movingWindowExtension = self.movingWindowSize * 0.25
        self.windowExtents = []

        self.minimumSampleNumber = minimumSampleNumber

        self.disaggregatingTemperature = disaggregatingTemperature

        # Flag to determine whether a multivariate linear regression should be
        # constructed for samples in each leaf of the regression tree
        # (see section 2.1 of Gao paper)
        self.perLeafLinearRegression = perLeafLinearRegression
        self.linearRegressionExtrapolationRatio = linearRegressionExtrapolationRatio

        self.regressorOpt = regressorOpt
        self.baggingRegressorOpt = baggingRegressorOpt

    def trainSharpener(self):
        ''' Train the sharpener using high- and low-resolution input files
        and settings specified in the constructor. Local (moving window) and
        global regression decision trees are trained with high-resolution data
        resampled to low resolution and low-resolution data. The training
        dataset is selected based on homogeneity of resampled high-resolution
        data being below specified threshold and quality mask (if given) of
        low resolution data. The homogeneity statistics are also used as weight
        factors for the training samples (more homogenous - higher weight).

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Select good data (training samples) from low- and high-resolution
        # input images.
        fileNum = 0
        for highResFile, lowResFile in zip(self.highResFiles, self.lowResFiles):

            scene_HR = gdal.Open(highResFile)
            scene_LR = gdal.Open(lowResFile)

            # First subset and reproject low res scene to fit with
            # high res scene
            subsetScene_LR = reprojectSubsetLowResScene(scene_HR, scene_LR)
            data_LR = subsetScene_LR.GetRasterBand(1).ReadAsArray()
            gt_LR = subsetScene_LR.GetGeoTransform()

            # Do the same with low res quality file (if provided) and flag
            # pixels which are considered to be of good quality
            if self.useQuality_LR:
                quality_LR = gdal.Open(self.lowResQualityFiles[fileNum])
                subsetQuality_LR = reprojectSubsetLowResScene(scene_HR, quality_LR)
                subsetQualityMask = subsetQuality_LR.GetRasterBand(1).ReadAsArray()
                qualityPix = np.in1d(subsetQualityMask.ravel(),
                                     self.lowResGoodQualityFlags).reshape(subsetQualityMask.shape)
                quality_LR = None
            else:
                qualityPix = np.ones(data_LR.shape).astype(bool)

            # Low resolution pixels with NaN value are always of bad quality
            qualityPix = np.logical_and(qualityPix, ~np.isnan(data_LR))

            # Then resample high res scene to low res pixel size while
            # extracting sub-low-res-pixel homogeneity statistics
            resMean, resStd = resampleHighResToLowRes(scene_HR, subsetScene_LR)
            resMean[resMean == 0] = 0.000001
            resCV = np.sum(resStd/resMean, 2) / resMean.shape[2]
            resCV[np.isnan(resCV)] = 1000

            # Resampled high resolution pixels where at least one "parameter"
            # is NaN are also of bad quality
            resNaN = np.any(np.isnan(resMean), -1)
            qualityPix = np.logical_and(qualityPix, ~resNaN)

            windows = []
            extents = []
            # If moving window approach is used (section 2.3 of Gao paper)
            # then calculate the extent of each sampling window in low
            # resolution pixels
            if self.movingWindowSize > 0:
                for y in range(int(math.ceil(data_LR.shape[0]/self.movingWindowSize))):
                    for x in range(int(math.ceil(data_LR.shape[1]/self.movingWindowSize))):
                        windows.append([int(max(y*self.movingWindowSize-self.movingWindowExtension, 0)),
                                        int(min((y+1)*self.movingWindowSize+self.movingWindowExtension,
                                                data_LR.shape[0])),
                                        int(max(x*self.movingWindowSize-self.movingWindowExtension, 0)),
                                        int(min((x+1)*self.movingWindowSize+self.movingWindowExtension,
                                                data_LR.shape[1]))])
                        # Save the extents of this window in projection coordinates as
                        # UL and LR point coordinates
                        ul = pix2point([x*self.movingWindowSize, y*self.movingWindowSize],
                                             gt_LR)
                        lr = pix2point([(x+1)*self.movingWindowSize,
                                              (y+1)*self.movingWindowSize],
                                             gt_LR)
                        extents.append([ul, lr])
            
            # And always add the whole extent of low res image to also estimate
            # the regression tree for the whole image
            windows.append([0, data_LR.shape[0], 0, data_LR.shape[1]])

            goodData_LR = [None for _ in range(len(windows))]
            goodData_HR = [None for _ in range(len(windows))]
            weight = [None for _ in range(len(windows))]
            
            # For each window extract the good quality low res and high res pixels
            for i, window in enumerate(windows):
                rows = slice(window[0], window[1])
                cols = slice(window[2], window[3])
                qualityPixWindow = qualityPix[rows, cols]
                resCVWindow = resCV[rows, cols]

                # Good pixels are those where both low and high resolution data exists
                goodPix = np.logical_and.reduce((qualityPixWindow,
                                                 resCVWindow > 0,
                                                 resCVWindow < 1000))
                # If number of good pixels is below threshold then do not train a model
                if np.sum(goodPix) < self.minimumSampleNumber:
                    goodPix = np.zeros(goodPix.shape).astype(bool)

                if self.autoAdjustCvThreshold:
                    if ~np.any(goodPix):
                        self.cvHomogeneityThreshold = 0
                    else:
                        self.cvHomogeneityThreshold = np.percentile(resCVWindow[goodPix],
                                                                    self.precentileThreshold)
                        print('Homogeneity CV threshold: %.2f' % self.cvHomogeneityThreshold)
                homogenousPix = np.logical_and(resCVWindow < self.cvHomogeneityThreshold,
                                               resCVWindow > 0)

                goodData_LR[i] = appendNpArray(goodData_LR[i],
                                                     data_LR[rows, cols][goodPix])
                goodData_HR[i] = appendNpArray(goodData_HR[i],
                                                     resMean[rows, cols, :][goodPix, :], axis=0)

                # Also estimate weight given to each pixel as the inverse of its
                # heterogeneity. The most heterogenous (beyond CV treshold) pixels are extra
                # penalized by having their weight halved.
                w = 1/resCVWindow[goodPix]
                if w.size > 1:
                    w = (w - np.min(w)) / (np.max(w) - np.min(w))
                    w[~homogenousPix[goodPix]] = w[~homogenousPix[goodPix]] / 2
                weight[i] = appendNpArray(weight[i], w)

                # Print some stats
                if goodData_LR[i].size > 0:
                    percentageUsedPixels = int(float(goodData_LR[i].size) /
                                               float(data_LR[rows, cols][qualityPixWindow].size) * 100)
                    print('Number of training elements for is ' +
                          str(goodData_LR[i].size) + ' representing ' +
                          str(percentageUsedPixels)+'% of avaiable low-resolution data.')
            
            # Close all files
            scene_HR = None
            scene_LR = None
            subsetScene_LR = None
            if self.useQuality_LR:
                subsetQuality_LR = None
            fileNum = fileNum + 1

        self.windowExtents = extents
        windowsNum = len(windows)

        # Once all the samples have been picked fit all the local and global
        # regressions
        self.reg = [None for _ in range(windowsNum)]
        for i in range(windowsNum):
            if i < windowsNum-1:
                local = True
            else:
                local = False
            if len(goodData_LR[i]) > 0:
                self.reg[i] = \
                    self._doFit(goodData_LR[i], goodData_HR[i], weight[i], local)
        

    def applySharpener(self, highResFilename, lowResFilename=None):
        ''' Apply the trained sharpener to a given high-resolution image to
        derive corresponding disaggregated low-resolution image. If local
        regressions were used during training then they will only be applied
        where their moving window extent overlaps with the high resolution
        image passed to this function. Global regression will be applied to the
        whole high-resolution image wihtout geographic constraints.

        Parameters
        ----------
        highResFilename: string
            Path to the high-resolution image file do be used during
            disaggregation.

        lowResFilename: string (optional, default: None)
            Path to the low-resolution image file corresponding to the
            high-resolution input file. If local regressions
            were trained and low-resolution filename is given then the local
            and global regressions will be combined based on residual values of
            the different regressions to the low-resolution image (see [Gao2012]
            2.3). If local regressions were trained and low-resolution
            filename is not given then only the local regressions will be used.


        Returns
        -------
        outImage: GDAL memory file object
            The file object contains an in-memory, georeferenced disaggregator
            output.
        '''

        # Open and read the high resolution input file
        highResFile = gdal.Open(highResFilename)
        inData = np.zeros((highResFile.RasterYSize, highResFile.RasterXSize,
                           highResFile.RasterCount))
        for band in range(highResFile.RasterCount):
            data = highResFile.GetRasterBand(band+1).ReadAsArray().astype(float)
            no_data = highResFile.GetRasterBand(band+1).GetNoDataValue()
            data[data == no_data] = np.nan
            inData[:, :, band] = data
        gt = highResFile.GetGeoTransform()

        shape = inData.shape
        ysize = shape[0]
        xsize = shape[1]

        # Temporarly get rid of NaN's
        nanInd = np.isnan(inData)
        inData[nanInd] = 0

        outWindowData = np.empty((ysize, xsize))*np.nan
        outFullData = np.empty((ysize, xsize))*np.nan
        # Do the downscailing on the moving windows if there are any and also process the full
        # scene using the same windows to optimize memory usage
        for i, extent in enumerate(self.windowExtents):
            print(i)
            if self.reg[i] is not None:
                [minX, minY] = point2pix(extent[0], gt)  # UL
                [minX, minY] = [max(minX, 0), max(minY, 0)]
                [maxX, maxY] = point2pix(extent[1], gt)  # LR
                [maxX, maxY] = [min(maxX, xsize), min(maxY, ysize)]
                windowInData = inData[minY:maxY, minX:maxX, :]
                outWindowData[minY:maxY, minX:maxX] = \
                    self._doPredict(windowInData, self.reg[i])
                if self.reg[-1] is not None:
                    outFullData[minY:maxY, minX:maxX] = \
                        self._doPredict(windowInData, self.reg[-1])
        print(self.windowExtents)
        # If there were no moving windows then do the downscailing on the whole input image
        if np.all(np.isnan(outFullData)) and self.reg[-1] is not None:
            outFullData = self._doPredict(inData, self.reg[-1])

        # Combine the windowed and whole image regressions
        # If there is no windowed regression just use the whole image regression
        if np.all(np.isnan(outWindowData)):
            outData = outFullData
        # If corresponding low resolution file is provided then combine the two
        # regressions based on residuals (see section 2.3 of Gao paper)
        elif lowResFilename is not None:
            lowResScene = gdal.Open(lowResFilename)
            outWindowScene = saveImg(outWindowData,
                                           highResFile.GetGeoTransform(),
                                           highResFile.GetProjection(),
                                           "MEM",
                                           noDataValue=np.nan)
            windowedResidual_LR, gt_LR = self._calculateResidual(outWindowScene, lowResScene)
            outWindowScene = None
            outFullScene = saveImg(outFullData,
                                         highResFile.GetGeoTransform(),
                                         highResFile.GetProjection(),
                                         "MEM",
                                         noDataValue=np.nan)
            fullResidual_LR, gt_LR = self._calculateResidual(outFullScene, lowResScene)
            lowResScene = None
            # windowed weight
            ww_LR = (1/windowedResidual_LR)**2/((1/windowedResidual_LR)**2 +
                                                (1/fullResidual_LR)**2)
            wwLowResScene = saveImg(ww_LR,
                                          gt_LR,
                                          highResFile.GetProjection(),
                                          "MEM",
                                          noDataValue=np.nan)
            ww = resampleLowResToHighRes(wwLowResScene, outFullScene)
            ww = np.clip(ww, 0.0, 1.0)
            # full weight
            fw = 1 - ww
            if self.disaggregatingTemperature:
                outData = ((outWindowData**4)*ww + (outFullData**4)*fw)**0.25
            else:
                outData = outWindowData*ww + outFullData*fw
        # Otherwised use just windowed regression
        else:
            outData = outWindowData

        # Fix NaN's
        nanInd = np.any(nanInd, -1)
        outData[nanInd] = np.nan

        outImage = saveImg(outData,
                                 highResFile.GetGeoTransform(),
                                 highResFile.GetProjection(),
                                 "MEM",
                                 noDataValue=np.nan)

        highResFile = None
        inData = None
        return outImage, self.reg


    def residualAnalysis(self, disaggregatedFile, lowResFilename, lowResQualityFilename=None,
                         doCorrection=True):
        ''' Perform residual analysis and (optional) correction on the
        disaggregated file (see [Gao2012] 2.4).

        Parameters
        ----------
        disaggregatedFile: string or GDAL file object
            If string, path to the disaggregated image file; if gdal file
            object, the disaggregated image.

        lowResFilename: string
            Path to the low-resolution image file corresponding to the
            high-resolution disaggregated image.

        lowResQualityFilename: string (optional, default: None)
            Path to low-resolution quality image file. If provided then low
            quality values are masked out during residual analysis. Otherwise
            all values are considered to be of good quality.

        doCorrection: boolean (optional, default: True)
            Flag indication whether residual (bias) correction should be
            performed or not.


        Returns
        -------
        residualImage: GDAL memory file object
            The file object contains an in-memory, georeferenced residual image.

        correctedImage: GDAL memory file object
            The file object contains an in-memory, georeferenced residual
            corrected disaggregated image, or None if doCorrection was set to
            False.
        '''

        if not os.path.isfile(str(disaggregatedFile)):
            scene_HR = disaggregatedFile
        else:
            scene_HR = gdal.Open(disaggregatedFile)
        scene_LR = gdal.Open(lowResFilename)
        if lowResQualityFilename is not None:
            quality_LR = gdal.Open(lowResQualityFilename)
        else:
            quality_LR = None
        
        #!!! The problem happening when correcting a patch comes from the next 3 calls
        # The reason is the resampling making null values appear on the borders 
        # and a correction would pose problem...
        residual_LR, gt_res = self._calculateResidual(scene_HR, scene_LR, quality_LR)
        residualImage = saveImg(residual_LR,
                                      gt_res,
                                      scene_HR.GetProjection(),
                                      "MEM",
                                      noDataValue=np.nan)
        residual_HR = resampleLowResToHighRes(residualImage, scene_HR)

        if self.disaggregatingTemperature:
            if doCorrection:
                corrected = (residual_HR + scene_HR.GetRasterBand(1).ReadAsArray()**4)**0.25
                correctedImage = saveImg(corrected,
                                               scene_HR.GetGeoTransform(),
                                               scene_HR.GetProjection(),
                                               "MEM",
                                               noDataValue=np.nan)
            else:
                correctedImage = None
            # Convert residual back to temperature for easier visualisation
            residual_LR = (residual_LR + 273.15**4)**0.25 - 273.15
            residualImage = saveImg(residual_LR,
                                          gt_res,
                                          scene_HR.GetProjection(),
                                          "MEM",
                                          noDataValue=np.nan)
        else:
            if doCorrection:
                corrected = residual_HR + scene_HR.GetRasterBand(1).ReadAsArray()
                correctedImage = saveImg(corrected,
                                               scene_HR.GetGeoTransform(),
                                               scene_HR.GetProjection(),
                                               "MEM",
                                               noDataValue=np.nan)
            else:
                correctedImage = None

        print("LR residual bias: "+str(np.nanmean(residual_LR)))
        print("LR residual RMSD: "+str(np.nanmean(residual_LR**2)**0.5))

        scene_HR = None
        scene_LR = None
        quality_LR = None

        return residualImage, correctedImage

    def _doFit(self, goodData_LR, goodData_HR, weight, local):
        ''' Private function. Fits the regression tree.
        '''

        # For local regression constrain the number of tree
        # nodes (rules) - section 2.3
        if local:
            self.regressorOpt["max_leaf_nodes"] = 10
        else:
            self.regressorOpt["max_leaf_nodes"] = 30
        self.regressorOpt["min_samples_leaf"] = min(self.minimumSampleNumber, 10)

        # If per leaf linear regression is used then use modified
        # DecisionTreeRegressor. Otherwise use the standard one.
        if self.perLeafLinearRegression:
            baseRegressor = \
                DecisionTreeRegressorWithLinearLeafRegression(self.linearRegressionExtrapolationRatio,
                                                              self.regressorOpt)
        else:
            baseRegressor = \
                tree.DecisionTreeRegressor(**self.regressorOpt)

        reg = ensemble.BaggingRegressor(baseRegressor, **self.baggingRegressorOpt)
        if goodData_HR.shape[0] <= 1:
            reg.max_samples = 1.0
        reg = reg.fit(goodData_HR, goodData_LR, sample_weight=weight)

        return reg

    def _doPredict(self, inData, reg):
        ''' Private function. Calls the regression tree.
        '''

        origShape = inData.shape
        if len(origShape) == 3:
            bands = origShape[2]
        else:
            bands = 1
        # Do the actual decision tree regression
        inData = inData.reshape((-1, bands))
        outData = reg.predict(inData)
        outData = outData.reshape((origShape[0], origShape[1]))

        return outData

    def _calculateResidual(self, downscaledScene, originalScene, originalSceneQuality=None):
        ''' Private function. Calculates residual between overlapping
            high-resolution and low-resolution images.
        '''

        # First subset and reproject original (low res) scene to fit with
        # downscaled (high res) scene
        subsetScene_LR = reprojectSubsetLowResScene(downscaledScene,
                                                          originalScene,
                                                          resampleAlg=gdal.GRA_NearestNeighbour)
        data_LR = subsetScene_LR.GetRasterBand(1).ReadAsArray().astype(float)
        gt_LR = subsetScene_LR.GetGeoTransform()

        # If quality file for the low res scene is provided then mask out all
        # bad quality pixels in the subsetted LR scene. Otherwise assume that all
        # low res pixels are of good quality.
        if originalSceneQuality is not None:
            subsetQuality_LR = reprojectSubsetLowResScene(downscaledScene,
                                                                originalSceneQuality,
                                                                resampleAlg=gdal.GRA_NearestNeighbour)
            goodPixMask_LR = subsetQuality_LR.GetRasterBand(1).ReadAsArray()
            goodPixMask_LR = np.in1d(goodPixMask_LR.ravel(),
                                     self.lowResGoodQualityFlags).reshape(goodPixMask_LR.shape)
            data_LR[~goodPixMask_LR] = np.nan

        # Then resample high res scene to low res pixel size
        if self.disaggregatingTemperature:
            # When working with tempratures they should be converted to
            # radiance values before aggregating to be physically accurate.
            radianceScene = saveImg(downscaledScene.GetRasterBand(1).ReadAsArray()**4,
                                          downscaledScene.GetGeoTransform(),
                                          downscaledScene.GetProjection(),
                                          "MEM",
                                          noDataValue=np.nan)
            resMean, _ = resampleHighResToLowRes(radianceScene,
                                                       subsetScene_LR)
            # Find the residual (difference) between the two)
            residual_LR = data_LR**4 - resMean[:, :, 0]
        else:
            resMean, _ = resampleHighResToLowRes(downscaledScene,
                                                       subsetScene_LR)
            # Find the residual (difference) between the two
            residual_LR = data_LR - resMean[:, :, 0]

        return residual_LR, gt_LR