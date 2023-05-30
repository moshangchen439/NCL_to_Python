# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Feiyang Wang, University of Hohai
# If you have any questions or comments, please contact me at: wangfy@hhu.edu.cn
# ------------------------------------------------------------------------------
# This software may be used, copied, or redistributed as long as it is not
# sold and this notice is reproduced on each copy made. This
# routine is provided as is without any express or implied warranties
# whatsoever.
# ------------------------------------------------------------------------------
# CAUTION: side effects of reading or using this poorly coded,
# uncommented program may include nausea, hives, and uncontrolled
# weeping.  Good luck!
# ------------------------------------------------------------------------------
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from scipy.stats import linregress


def dim_avg_n(data, dim=0):
    '''均值, 元数据丢失'''
    assert isinstance(dim, int)
    inputdata = np.array(data)
    return np.nanmean(inputdata, axis=dim)


def dim_avg_n_Wrap(data, dim=0, keepdims=False):
    '''均值, 保留元数据'''
    assert isinstance(dim, int)
    if isinstance(data, xr.DataArray):
        return data.mean(dim=data.dims[dim], skipna=True)
    else:
        raise ValueError('Not a DataArray, please use dim_avg_n')


def percentile(data, q, dim=0, interpolation='linear'):
    '''分位数'''
    assert isinstance(dim, int)
    if isinstance(data, xr.DataArray):
        return data.quantile(q/100, dim=data.dims[dim], method=interpolation, skipna=True)
    else:
        inputdata = np.array(data)
        return np.nanpercentile(inputdata, q, axis=dim, method=interpolation)


def clmMon(data):
    '''气候态: 月平均'''
    if isinstance(data, xr.DataArray):
        return data.groupby('time.month').mean(dim='time')
    else:
        raise ValueError('Not a DataArray')


def clmDay(data):
    '''气候态: 日平均'''
    if isinstance(data, xr.DataArray):
        return data.groupby(data.time.dt.strftime('%m-%d')).mean(dim='time')
    else:
        raise ValueError('Not a DataArray')


def calcDayAnom(data):
    '''距平: 去季节(日数据)'''
    if isinstance(data, xr.DataArray):
        return data.groupby(data.time.dt.strftime('%m-%d')) - clmDay(data)
    else:
        raise ValueError('Not a DataArray')


def calcMonAnom(data):
    '''距平: 去季节(月数据)'''
    if isinstance(data, xr.DataArray):
        return data.groupby('time.month') - clmMon(data)
    else:
        raise ValueError('Not a DataArray')


def dim_stddev_n(data, dim=0):
    '''标准差, 元数据丢失'''
    assert isinstance(dim, int)
    inputdata = np.array(data)
    return np.nanstd(inputdata, axis=dim)


def dim_stddev_n_Wrap(data, dim=0):
    '''标准差, 元数据保留'''
    assert isinstance(dim, int)
    if isinstance(data, xr.DataArray):
        return data.std(dim=data.dims[dim], skipna=True)
    else:
        raise ValueError('Not a DataArray')


def dim_variance_n(data, dim=0):
    '''无偏样本方差, 元数据丢失'''
    assert isinstance(dim, int)
    inputdata = np.array(data)
    return np.nanvar(inputdata, axis=dim, ddof=1)


def dim_variance_n_Wrap(data, dim=0):
    '''无偏样本方差， 元数据保留'''
    assert isinstance(dim, int)
    if isinstance(data, xr.DataArray):
        return data.var(dim=data.dims[dim], skipna=True, ddof=1)
    else:
        raise ValueError('Not a DataArray')


def copy_VarCoords(data, metadata):
    '''坐标拷贝'''
    if isinstance(metadata, xr.DataArray):
        dims = metadata.dims
        coords = metadata.coords
        return xr.DataArray(data=np.array(data), dims=dims, coords=coords)
    else:
        raise ValueError('Second argument is not a DataArray')


def escorc(data1, data2):
    '''求相关系数及其双尾P值(贝塔检验), 元数据丢失'''
    x = np.array(data1)
    y = np.array(data2)
    ndimx = x.ndim
    ndimy = y.ndim
    shapex = x.shape
    shapey = y.shape
    if ndimx == 1 and ndimy == 1 and (shapex[0] == shapey[0]):
        r, p = pearsonr(x, y)
        return r, p
    elif (ndimx <= ndimy) and (list(shapex) == list(shapey[:ndimx])):
        if ndimx == ndimy:
            xtemp = np.reshape(x, (shapex[0], -1))
        else:
            for i in np.arange(0, ndimy - ndimx):
                x = np.expand_dims(x, -1)
            x = np.broadcast_to(x, shapey)
            xtemp = np.reshape(x, (shapex[0], -1))
        ytemp = np.reshape(y, (shapey[0], -1))
        rtemp = np.full((ytemp.shape[1]), np.nan)
        ptemp = np.full((ytemp.shape[1]), np.nan)
        for i in np.arange(0, ytemp.shape[1]):
            rtemp[i], ptemp[i] = pearsonr(xtemp[:, i], ytemp[:, i])
        r = np.reshape(rtemp, shapey[1:])
        p = np.reshape(ptemp, shapey[1:])
        return r, p
    else:
        raise ValueError('该函数计算两个变量最左边维度的线性相关系数及P值, \
                          要求第一个变量的维数小于等于第二个变量, 且同一维度大小一样')


def linreg(data1, data2):
    '''求线性回归系数、截距、相关系数及双尾P值(t检验), 元数据丢失'''
    x = np.array(data1)
    y = np.array(data2)
    ndimx = x.ndim
    ndimy = y.ndim
    shapex = x.shape
    shapey = y.shape
    if ndimx == 1 and ndimy == 1 and (shapex[0] == shapey[0]):
        result = linregress(x, y)
        return result.slop, result.intercept, result.rvalue, result.pvalue
    elif (ndimx <= ndimy) and (list(shapex) == list(shapey[:ndimx])):
        if ndimx == ndimy:
            xtemp = np.reshape(x, (shapex[0], -1))
        else:
            for i in np.arange(0, ndimy - ndimx):
                x = np.expand_dims(x, -1)
            x = np.broadcast_to(x, shapey)
            xtemp = np.reshape(x, (shapex[0], -1))
        ytemp = np.reshape(y, (shapey[0], -1))
        slopetemp = np.full((ytemp.shape[1]), np.nan)
        intertemp = np.full((ytemp.shape[1]), np.nan)
        rtemp = np.full((ytemp.shape[1]), np.nan)
        ptemp = np.full((ytemp.shape[1]), np.nan)
        for i in np.arange(0, ytemp.shape[1]):
            slopetemp[i], intertemp[i], rtemp[i], ptemp[i], _ = linregress(xtemp[:, i], ytemp[:, i])
        slope = np.reshape(slopetemp, shapey[1:])
        intercept = np.reshape(intertemp, shapey[1:])
        r = np.reshape(rtemp, shapey[1:])
        p = np.reshape(ptemp, shapey[1:])
        return slope, intercept, r, p
    else:
        raise ValueError('该函数计算两个变量最左边维度的线性回归, \
                          要求第一个变量的维数小于等于第二个变量, 且同一维度大小一样')


def dim_standardize_n_Wrap(data, dim=0):
    '''标准化, 元数据保留'''
    assert isinstance(dim, int)
    if isinstance(data, xr.DataArray):
        return xr.apply_ufunc(
            lambda x, m, s: (x - m) / s,
            data,
            dim_avg_n_Wrap(data, dim),
            dim_stddev_n_Wrap(data, dim)
        )
    else:
        raise ValueError('Not a DataArray')


def dim_standardize_n(data, dim=0):
    '''标准化， 元数据丢失'''
    assert isinstance(dim, int)
    inputdata = np.array(data)
    return (inputdata - np.nanmean(inputdata, dim, keepdims=True)) / np.nanstd(inputdata, dim, keepdims=True)


if __name__ == '__main__':
    a = [1, 2, 3]
    f1 = xr.open_dataset('hgt.2015.nc')
    hgt = f1.hgt
    f2 = xr.open_dataset('uwnd.2015.nc')
    u = f2.uwnd
    # b = dim_avg_n(a, 1)
    d = xr.DataArray(a)
    # c = dim_avg_n_Wrap(d)
    # e = percentile(data, 50)
    da = dim_standardize_n(hgt, 1)
    print(da[10, 1, 4, 3])
