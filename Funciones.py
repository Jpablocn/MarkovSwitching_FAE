# -*- coding: utf-8 -*-

"""
Modulo Auxiliar para markov_switching

Este modulo, tiene todos los metodos auxiliares
que se utilizan para compilar los archivos
markov_regression.py y markov_autoregression.py.
Por tanto, ahora podemos llamar a todos los
metodos de este modulo utilizando,

            from modulo_aux import *

Para probar estos metodos, a modo de ejemplo importaremos
la data rgnp de stats models,
    from statsmodels.tsa.regime_switching.tests.test_markov_autoregression import rgnp
Luego, basta llamar a cada funcion de la sgte. forma,
_get_yarr(rgnp)
_get_xarr(rgnp)
_convert_endog_exog(rgnp,rgnp)

"""

import numpy as np
import pandas as pd

from pandas import DataFrame
from statsmodels.tools.validation import int_like, bool_like, string_like
from statsmodels.tools.data import _is_using_pandas, _is_recarray
from statsmodels.tools.validation import array_like

import statsmodels.tools.data as data_util
from statsmodels.tools.data import _is_using_pandas


def prepare_exog(exog):
    """
    Para probar este metodo, definimos una variable
    exog, e.g.,
        prepare_exog(exog=[[1,2,2,0.5],[1,1.2,1.3]])
    """
    k_exog = 0
    if exog is not None:
        exog_is_using_pandas = _is_using_pandas(exog, None)
        if not exog_is_using_pandas:
            exog = np.asarray(exog)

        # Debemos tener una matriz 2-dim.
        if exog.ndim == 1:
            if not exog_is_using_pandas:
                exog = exog[:, None]
            else:
                exog = pd.DataFrame(exog)

        k_exog = exog.shape[1]
    return k_exog, exog


def _get_yarr(endog):
    if data_util._is_structured_ndarray(endog):
        endog = data_util.struct_to_ndarray(endog)
    endog = np.asarray(endog)
    if len(endog) == 1:
        if endog.ndim == 1:
            return endog
        elif endog.ndim > 1:
            return np.asarray([endog.squeeze()])
    return endog.squeeze()


def _get_xarr(exog):
    if data_util._is_structured_ndarray(exog):
        exog = data_util.struct_to_ndarray(exog)
    return np.asarray(exog)


def _convert_endog_exog(endog, exog):
    # para salidas consistentes si endog es (n, 1)
    yarr = _get_yarr(endog)
    xarr = None
    if exog is not None:
        xarr = _get_xarr(exog)
        if xarr.ndim == 1:
            xarr = xarr[:, None]
        if xarr.ndim != 2:
            raise ValueError("exog is not 1d or 2d")

    return yarr, xarr


def lagmat(x, maxlag, trim='forward', original='ex', use_pandas=False):
    """
    Crea un 2d-array de lags.

    Parameters
    ----------
    x : array_like
        Data; if 2d, observation in rows and variables in columns.
    maxlag : int
        All lags from zero to maxlag are included.
    trim : {'forward', 'backward', 'both', 'none', None}
        The trimming method to use.

        * 'forward' : trim invalid observations in front.
        * 'backward' : trim invalid initial observations.
        * 'both' : trim invalid observations on both sides.
        * 'none', None : no trimming of observations.
    original : {'ex','sep','in'}
        How the original is treated.

        * 'ex' : drops the original array returning only the lagged values.
        * 'in' : returns the original array and the lagged values as a single
          array.
        * 'sep' : returns a tuple (original array, lagged values). The original
                  array is truncated to have the same number of rows as
                  the returned lagmat.
    use_pandas : bool
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    lagmat : ndarray
        The array with lagged observations.
    y : ndarray, optional
        Only returned if original == 'sep'.

    Ejemplos
    ________

    Para utilizar este metodo definimos un array,

    X = np.arange(1,7)
    X = X.reshape(-1,2)

    Y llamamos a la función con sus respectivos parametros,
    aunque debemos definir trim, dependiendo del resultado
    esperado,

    lagmat(X, maxlag=2, trim="forward", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.]])

    lagmat(X, maxlag=2, trim="backward", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])

    lagmat(X, maxlag=2, trim="forward", original='in')
    lagmat(X,2)[2:]
    """
    maxlag_aux = int_like(maxlag, 'maxlag')
    use_pandas = bool_like(use_pandas, 'use_pandas')
    trim = string_like(trim, 'trim', optional=True,
                       options=('forward', 'backward', 'both', 'none'))
    original = string_like(original, 'original', options=('ex', 'sep', 'in'))

    # TODO:  allow list of lags additional to maxlag
    orig = x
    x = array_like(x, 'x', ndim=2, dtype=None)
    is_pandas = _is_using_pandas(orig, None) and use_pandas
    trim = 'none' if trim is None else trim
    trim = trim.lower()
    if is_pandas and trim in ('none', 'backward'):
        raise ValueError("trim cannot be 'none' or 'forward' when used on "
                         "Series or DataFrames")

    dropidx = 0
    nobs, nvar = x.shape
    if original in ['ex', 'sep']:
        dropidx = nvar
    if maxlag_aux >= nobs:
        raise ValueError("maxlag_aux should be < nobs")
    lm = np.zeros((nobs + maxlag_aux, nvar * (maxlag_aux + 1)))
    for k in range(0, int(maxlag_aux + 1)):
        lm[maxlag_aux - k:nobs + maxlag_aux - k, nvar * (maxlag_aux - k):nvar * (maxlag_aux - k + 1)] = x

    if trim in ('none', 'forward'):
        startobs = 0
    elif trim in ('backward', 'both'):
        startobs = maxlag_aux
    else:
        raise ValueError('trim option not valid')

    if trim in ('none', 'backward'):
        stopobs = len(lm)
    else:
        stopobs = nobs

    if is_pandas:
        x = orig
        x_columns = x.columns if isinstance(x, DataFrame) else [x.name]
        columns = [str(col) for col in x_columns]
        for lag in range(maxlag_aux):
            lag_str = str(lag + 1)
            columns.extend([str(col) + '.L.' + lag_str for col in x_columns])
        lm = DataFrame(lm[:stopobs], index=x.index, columns=columns)
        lags = lm.iloc[startobs:]
        if original in ('sep', 'ex'):
            leads = lags[x_columns]
            lags = lags.drop(x_columns, 1)
    else:
        lags = lm[startobs:stopobs, dropidx:]
        if original == 'sep':
            leads = lm[startobs:stopobs, :dropidx]

    if original == 'sep':
        return lags, leads
    else:
        return lags
