import numpy as np
import pandas as pd
from scipy.special import logsumexp
#from Markov_Switching_FAE.Funciones import lagmat
from Funciones import lagmat
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitchingResults, MarkovSwitchingParams
from statsmodels.tsa.regime_switching import markov_switching
#from Markov_Switching_FAE.markov_switching import MarkovSwitchingResults, MarkovSwitchingParams
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.statespace.tools import find_best_blas_type, prepare_exog
from statsmodels.tools.tools import Bunch
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.statespace.tools import unconstrain_stationary_univariate, constrain_stationary_univariate
from statsmodels.tsa.regime_switching._hamilton_filter import (
    shamilton_filter_log, dhamilton_filter_log, chamilton_filter_log,
    zhamilton_filter_log)
from statsmodels.tsa.regime_switching._kim_smoother import (
    skim_smoother_log, dkim_smoother_log, ckim_smoother_log, zkim_smoother_log)

prefix_hamilton_filter_log_map = {
    's': shamilton_filter_log, 'd': dhamilton_filter_log,
    'c': chamilton_filter_log, 'z': zhamilton_filter_log
}
prefix_kim_smoother_log_map = {
    's': skim_smoother_log, 'd': dkim_smoother_log,
    'c': ckim_smoother_log, 'z': zkim_smoother_log
}


def cy_kim_smoother_log(regime_transition, predicted_joint_probabilities,
                        filtered_joint_probabilities):
    """
    Kim smoother in log space using Cython inner loop.
    Parameters
    ----------
    regime_transition : ndarray
        Matrix of regime transition probabilities, shaped either
        (k_regimes, k_regimes, 1) or if there are time-varying transition
        probabilities (k_regimes, k_regimes, nobs).
    predicted_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t-1}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t-1
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    filtered_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    Returns
    -------
    smoothed_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_T] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on all information.
        Shaped (k_regimes,) * (order + 1) + (nobs,).
    smoothed_marginal_probabilities : ndarray
        Array containing Pr[S_t=s_t | Y_T] - the probability of being in each
        regime conditional on all information. Shaped (k_regimes, nobs).
    """
    # Dimensions
    k_regimes = filtered_joint_probabilities.shape[0]
    nobs = filtered_joint_probabilities.shape[-1]
    order = filtered_joint_probabilities.ndim - 2
    dtype = filtered_joint_probabilities.dtype
    # Storage
    smoothed_joint_probabilities = np.zeros(
        (k_regimes,) * (order + 1) + (nobs,), dtype=dtype)
    # Get appropriate subset of transition matrix
    if regime_transition.shape[-1] == nobs + order:
        regime_transition = regime_transition[..., order:]
    # Convert to log space1
    regime_transition = np.log(np.maximum(regime_transition, 1e-20))
    # Run Cython smoother iterations
    prefix, dtype, _ = find_best_blas_type((
        regime_transition, predicted_joint_probabilities,
        filtered_joint_probabilities))
    func = prefix_kim_smoother_log_map[prefix]
    func(nobs, k_regimes, order, regime_transition,
         predicted_joint_probabilities.reshape(k_regimes ** (order + 1), nobs),
         filtered_joint_probabilities.reshape(k_regimes ** (order + 1), nobs),
         smoothed_joint_probabilities.reshape(k_regimes ** (order + 1), nobs))
    # Convert back from log space
    smoothed_joint_probabilities = np.exp(smoothed_joint_probabilities)
    # Get smoothed marginal probabilities S_t | T by integrating out
    # S_{t-k+1}, S_{t-k+2}, ..., S_{t-1}
    smoothed_marginal_probabilities = smoothed_joint_probabilities
    for i in range(1, smoothed_marginal_probabilities.ndim - 1):
        smoothed_marginal_probabilities = np.sum(
            smoothed_marginal_probabilities, axis=-2)
    return smoothed_joint_probabilities, smoothed_marginal_probabilities


def cy_hamilton_filter_log(initial_probabilities, regime_transition,
                           conditional_loglikelihoods, model_order):
    """
    Hamilton filter in log space using Cython inner loop.

    Parameters
    ----------
    initial_probabilities : ndarray
        Array of initial probabilities, shaped (k_regimes,) giving the
        distribution of the regime process at time t = -order where order
        is a nonnegative integer.
    regime_transition : ndarray
        Matrix of regime transition probabilities, shaped either
        (k_regimes, k_regimes, 1) or if there are time-varying transition
        probabilities (k_regimes, k_regimes, nobs + order).  Entry [i, j,
        t] contains the probability of moving from j at time t-1 to i at
        time t, so each matrix regime_transition[:, :, t] should be left
        stochastic.  The first order entries and initial_probabilities are
        used to produce the initial joint distribution of dimension order +
        1 at time t=0.
    conditional_loglikelihoods : ndarray
        Array of loglikelihoods conditional on the last `order+1` regimes,
        shaped (k_regimes,)*(order + 1) + (nobs,).

    Returns
    -------
    filtered_marginal_probabilities : ndarray
        Array containing Pr[S_t=s_t | Y_t] - the probability of being in each
        regime conditional on time t information. Shaped (k_regimes, nobs).
    predicted_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t-1}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t-1
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    joint_loglikelihoods : ndarray
        Array of loglikelihoods condition on time t information,
        shaped (nobs,).
    filtered_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    """

    # Dimensions
    k_regimes = len(initial_probabilities)
    nobs = conditional_loglikelihoods.shape[-1]
    order = conditional_loglikelihoods.ndim - 2
    dtype = conditional_loglikelihoods.dtype

    # Check for compatible shapes.
    incompatible_shapes = (
            regime_transition.shape[-1] not in (1, nobs + model_order)
            or regime_transition.shape[:2] != (k_regimes, k_regimes)
            or conditional_loglikelihoods.shape[0] != k_regimes)
    if incompatible_shapes:
        raise ValueError('Arguments do not have compatible shapes')

    # Convert to log space
    initial_probabilities = np.log(initial_probabilities)
    regime_transition = np.log(np.maximum(regime_transition, 1e-20))

    # Storage
    # Pr[S_t = s_t | Y_t]
    filtered_marginal_probabilities = (
        np.zeros((k_regimes, nobs), dtype=dtype))
    # Pr[S_t = s_t, ... S_{t-r} = s_{t-r} | Y_{t-1}]
    # Has k_regimes^(order+1) elements
    predicted_joint_probabilities = np.zeros(
        (k_regimes,) * (order + 1) + (nobs,), dtype=dtype)
    # log(f(y_t | Y_{t-1}))
    joint_loglikelihoods = np.zeros((nobs,), dtype)
    # Pr[S_t = s_t, ... S_{t-r+1} = s_{t-r+1} | Y_t]
    # Has k_regimes^order elements
    filtered_joint_probabilities = np.zeros(
        (k_regimes,) * (order + 1) + (nobs + 1,), dtype=dtype)

    # Initial probabilities
    filtered_marginal_probabilities[:, 0] = initial_probabilities
    tmp = np.copy(initial_probabilities)
    shape = (k_regimes, k_regimes)
    transition_t = 0
    for i in range(order):
        if regime_transition.shape[-1] > 1:
            transition_t = i
        tmp = np.reshape(regime_transition[..., transition_t],
                         shape + (1,) * i) + tmp
    filtered_joint_probabilities[..., 0] = tmp

    # Get appropriate subset of transition matrix
    if regime_transition.shape[-1] > 1:
        regime_transition = regime_transition[..., model_order:]

    # Run Cython filter iterations
    prefix, dtype, _ = find_best_blas_type((
        regime_transition, conditional_loglikelihoods, joint_loglikelihoods,
        predicted_joint_probabilities, filtered_joint_probabilities))
    func = prefix_hamilton_filter_log_map[prefix]
    func(nobs, k_regimes, order, regime_transition,
         conditional_loglikelihoods.reshape(k_regimes ** (order + 1), nobs),
         joint_loglikelihoods,
         predicted_joint_probabilities.reshape(k_regimes ** (order + 1), nobs),
         filtered_joint_probabilities.reshape(k_regimes ** (order + 1), nobs + 1))

    # Save log versions for smoother
    predicted_joint_probabilities_log = predicted_joint_probabilities
    filtered_joint_probabilities_log = filtered_joint_probabilities

    # Convert out of log scale
    predicted_joint_probabilities = np.exp(predicted_joint_probabilities)
    filtered_joint_probabilities = np.exp(filtered_joint_probabilities)

    # S_t | t
    filtered_marginal_probabilities = filtered_joint_probabilities[..., 1:]
    for i in range(1, filtered_marginal_probabilities.ndim - 1):
        filtered_marginal_probabilities = np.sum(
            filtered_marginal_probabilities, axis=-2)

    return (filtered_marginal_probabilities, predicted_joint_probabilities,
            joint_loglikelihoods, filtered_joint_probabilities[..., 1:],
            predicted_joint_probabilities_log,
            filtered_joint_probabilities_log[..., 1:])


class HamiltonFilterResults(object):
    """
    Resultados de aplicar el 'Hamilton Filter' a un state space model.
    Parameters
    ----------
    model : Representation
        A Statespace representation
    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_regimes : int
        The number of unobserved regimes.
    regime_transition : ndarray
        The regime transition matrix.
    initialization : str
        Initialization method for regime probabilities.
    initial_probabilities : ndarray
        Initial regime probabilities
    conditional_loglikelihoods : ndarray
        The loglikelihood values at each time period, conditional on regime.
    predicted_joint_probabilities : ndarray
        Predicted joint probabilities at each time period.
    filtered_marginal_probabilities : ndarray
        Filtered marginal probabilities at each time period.
    filtered_joint_probabilities : ndarray
        Filtered joint probabilities at each time period.
    joint_loglikelihoods : ndarray
        The likelihood values at each time period.
    llf_obs : ndarray
        The loglikelihood values at each time period.
    """

    def __init__(self, model, result):
        self.model = model
        self.nobs = model.nobs
        self.order = model.order
        self.k_regimes = model.k_regimes
        attributes = ['regime_transition', 'initial_probabilities',
                      'conditional_loglikelihoods',
                      'predicted_joint_probabilities',
                      'filtered_marginal_probabilities',
                      'filtered_joint_probabilities',
                      'joint_loglikelihoods']
        for name in attributes:
            setattr(self, name, getattr(result, name))
        self.initialization = model._initialization
        self.llf_obs = self.joint_loglikelihoods
        self.llf = np.sum(self.llf_obs)
        # Subset transition if necessary (e.g. for Markov autoregression)
        if self.regime_transition.shape[-1] > 1 and self.order > 0:
            self.regime_transition = self.regime_transition[..., self.order:]
        # Cache for predicted marginal probabilities
        self._predicted_marginal_probabilities = None

    @property
    def predicted_marginal_probabilities(self):
        if self._predicted_marginal_probabilities is None:
            self._predicted_marginal_probabilities = (
                self.predicted_joint_probabilities)
            for i in range(self._predicted_marginal_probabilities.ndim - 2):
                self._predicted_marginal_probabilities = np.sum(
                    self._predicted_marginal_probabilities, axis=-2)
        return self._predicted_marginal_probabilities

    @property
    def expected_durations(self):
        """
        (array) Expected duration of a regime, possibly time-varying.
        """
        diag = np.diagonal(self.regime_transition)
        expected_durations = np.zeros_like(diag)
        degenerate = np.any(diag == 1, axis=1)
        # For non-degenerate states, use the usual computation
        expected_durations[~degenerate] = 1 / (1 - diag[~degenerate])
        # For degenerate states, everything is np.nan, except for the one
        # state that is np.inf.
        expected_durations[degenerate] = np.nan
        expected_durations[diag == 1] = np.inf
        return expected_durations.squeeze()


class KimSmootherResults(HamiltonFilterResults):
    """
    Resultados de aplicar el 'Kim Smoother' a un Markov Switching model.
    Parameters
    ----------
    model : MarkovSwitchingModel
        The model object.
    result : dict
        A dictionary containing two keys: 'smoothd_joint_probabilities' and
        'smoothed_marginal_probabilities'.
    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    """

    def __init__(self, model, result):
        super(KimSmootherResults, self).__init__(model, result)
        attributes = ['smoothed_joint_probabilities',
                      'smoothed_marginal_probabilities']
        for name in attributes:
            setattr(self, name, getattr(result, name))


# noinspection PyTypeChecker,SpellCheckingInspection,PyAbstractClass,PyProtectedMember
class MarkovSwitching_FAE(tsbase.TimeSeriesModel):

    def __init__(self, endog, k_regimes, order, trend='c', exog=None,
                 exog_tvtp=None, switching_ar=True, switching_trend=True,
                 switching_exog=False, switching_variance=False,
                 dates=None, freq=None, missing='none'):

        # Properties
        self.switching_ar = switching_ar
        self.trend = trend
        self.switching_trend = switching_trend
        self.switching_exog = switching_exog
        self.switching_variance = switching_variance
        self.k_regimes = k_regimes
        self.tvtp = exog_tvtp is not None
        self.order = order

        # Switching options
        if self.switching_ar is True or self.switching_ar is False:
            self.switching_ar = [self.switching_ar] * order
        elif not len(self.switching_ar) == order:
            raise ValueError('Invalid iterable passed to `switching_ar`.')

        # Exogenous data
        self.k_exog, exog = prepare_exog(exog)

        # Trend
        nobs = len(endog)
        self.k_trend = 0
        self._k_exog = self.k_exog
        trend_exog = None
        if trend == 'c':
            trend_exog = np.ones((nobs, 1))
            self.k_trend = 1
        elif trend == 't':
            trend_exog = (np.arange(nobs) + 1)[:, np.newaxis]
            self.k_trend = 1
        elif trend == 'ct':
            trend_exog = np.c_[np.ones((nobs, 1)),
                               (np.arange(nobs) + 1)[:, np.newaxis]]
            self.k_trend = 2
        if trend_exog is not None:
            exog = trend_exog if exog is None else np.c_[trend_exog, exog]
            self._k_exog += self.k_trend

        # Exogenous data
        self.k_tvtp, self.exog_tvtp = prepare_exog(exog_tvtp)

        # Initialize the base model, tsbase.TimeSeriesModel
        super(MarkovSwitching_FAE, self).__init__(endog, exog, missing=missing)

        # Dimensions
        self.nobs = self.endog.shape[0]

        # Sanity checks
        if self.endog.ndim > 1 and self.endog.shape[1] > 1:
            raise ValueError('Must have univariate endogenous data.')
        if self.k_regimes < 2:
            raise ValueError('Markov switching models must have at least two'
                             ' regimes.')
        if not (self.exog_tvtp is None or self.exog_tvtp.shape[0] == self.nobs):
            raise ValueError('Time-varying transition probabilities exogenous'
                             ' array must have the same number of observations'
                             ' as the endogenous array.')

        self.parameters = markov_switching.MarkovSwitchingParams(self.k_regimes)
        k_transition = self.k_regimes - 1
        if self.tvtp:
            k_transition *= self.k_tvtp
        self.parameters['regime_transition'] = [1] * k_transition

        # Internal model properties: default is steady-state initialization
        self._initialization = 'steady-state'
        self._initial_probabilities = None

        # Switching options
        if self.switching_trend is True or self.switching_trend is False:
            self.switching_trend = [self.switching_trend] * self.k_trend
        elif not len(self.switching_trend) == self.k_trend:
            raise ValueError('Invalid iterable passed to `switching_trend`.')
        if self.switching_exog is True or self.switching_exog is False:
            self.switching_exog = [self.switching_exog] * self.k_exog
        elif not len(self.switching_exog) == self.k_exog:
            raise ValueError('Invalid iterable passed to `switching_exog`.')

        self.switching_coeffs = (np.r_[self.switching_trend, self.switching_exog].astype(bool).tolist())

        # Parameters
        self.parameters['exog'] = self.switching_coeffs
        self.parameters['variance'] = [1] if self.switching_variance else [0]

        # Sanity checks
        if self.nobs <= self.order:
            raise ValueError('Must have more observations than the order of'
                             ' the autoregression.')

        # Autoregressive exog
        self.exog_ar = lagmat(endog, self.order)[self.order:]

        # Reshape other datasets
        self.nobs -= self.order
        self.orig_endog = self.endog
        self.endog = self.endog[self.order:]
        if self._k_exog > 0:
            self.orig_exog = self.exog
            self.exog = self.exog[self.order:]

        # Reset the ModelData datasets
        self.data.endog, self.data.exog = (
            self.data._convert_endog_exog(self.endog, self.exog))

        # Reset indexes, if provided
        if self.data.row_labels is not None:
            self.data._cache['row_labels'] = (
                self.data.row_labels[self.order:])
        if self._index is not None:
            if self._index_generated:
                self._index = self._index[:-self.order]
            else:
                self._index = self._index[self.order:]

        # Parameters
        self.parameters['autoregressive'] = self.switching_ar
        # Cache an array for holding slices
        self._predict_slices = [slice(None, None, None)] * (self.order + 1)

    @property
    def param_names(self):
        """
        (lista de str) Lista de nombres para los parámetros.
        """
        param_names = np.zeros(self.k_params, dtype=object)

        # Transition probabilities
        if self.tvtp:
            # TODO add support for exog_tvtp_names
            param_names[self.parameters['regime_transition']] = [
                'p[%d->%d].tvtp%d' % (j, i, k)
                for i in range(self.k_regimes - 1)
                for k in range(self.k_tvtp)
                for j in range(self.k_regimes)
            ]
        else:
            param_names[self.parameters['regime_transition']] = [
                'p[%d->%d]' % (j, i)
                for i in range(self.k_regimes - 1)
                for j in range(self.k_regimes)]

        # Regression coefficients
        if np.any(self.switching_coeffs):
            for i in range(self.k_regimes):
                param_names[self.parameters[i, 'exog']] = [
                    '%s[%d]' % (exog_name, i) for exog_name in self.exog_names]
        else:
            param_names[self.parameters['exog']] = self.exog_names

        # Variances
        if self.switching_variance:
            for i in range(self.k_regimes):
                param_names[self.parameters[i, 'variance']] = 'sigma2[%d]' % i
        else:
            param_names[self.parameters['variance']] = 'sigma2'

        # Autoregressive
        if np.any(self.switching_ar):
            for i in range(self.k_regimes):
                param_names[self.parameters[i, 'autoregressive']] = [
                    'ar.L%d[%d]' % (j + 1, i) for j in range(self.order)]
        else:
            param_names[self.parameters['autoregressive']] = [
                'ar.L%d' % (j + 1) for j in range(self.order)]

        return param_names.tolist()

    def _em_variance(self, result, endog, exog, betas, tmp=None):
        """
        EM step for variances
        """
        k_exog = 0 if exog is None else exog.shape[1]

        if self.switching_variance:
            variance = np.zeros(self.k_regimes)
            for i in range(self.k_regimes):
                if k_exog > 0:
                    resid = endog - np.dot(exog, betas[i])
                else:
                    resid = endog
                variance[i] = (
                        np.sum(resid ** 2 *
                               result.smoothed_marginal_probabilities[i]) /
                        np.sum(result.smoothed_marginal_probabilities[i]))
        else:
            variance = 0
            if tmp is None:
                tmp = np.sqrt(result.smoothed_marginal_probabilities)
            for i in range(self.k_regimes):
                tmp_endog = tmp[i] * endog
                if k_exog > 0:
                    tmp_exog = tmp[i][:, np.newaxis] * exog
                    resid = tmp_endog - np.dot(tmp_exog, betas[i])
                else:
                    resid = tmp_endog
                variance += np.sum(resid ** 2)
            variance /= self.nobs
        return variance

    @property
    def k_params(self):
        """
        (s ints) Entrega el número de parametros en el modelo.
        """
        return self.parameters.k_params

    @property
    def start_params(self):
        """
        (array) Parámetros iniciales para la estimación de máxima verosimilitud.
        """
        params = np.zeros(self.k_params, dtype=np.float64)

        # Transition probabilities
        if self.tvtp:
            params[self.parameters['regime_transition']] = 0.
        else:
            params[self.parameters['regime_transition']] = 1. / self.k_regimes

        endog = self.endog.copy()
        if self._k_exog > 0 and self.order > 0:
            exog = np.c_[self.exog, self.exog_ar]
        elif self._k_exog > 0:
            exog = self.exog
        elif self.order > 0:
            exog = self.exog_ar
        if self._k_exog > 0 or self.order > 0:
            beta = np.dot(np.linalg.pinv(exog), endog)
            variance = np.var(endog - np.dot(exog, beta))
        else:
            variance = np.var(endog)
        # Regression coefficients
        if self._k_exog > 0:
            if np.any(self.switching_coeffs):
                for i in range(self.k_regimes):
                    params[self.parameters[i, 'exog']] = (
                            beta[:self._k_exog] * (i / self.k_regimes))
            else:
                params[self.parameters['exog']] = beta[:self._k_exog]
        # Autoregressive
        if self.order > 0:
            if np.any(self.switching_ar):
                for i in range(self.k_regimes):
                    params[self.parameters[i, 'autoregressive']] = (
                            beta[self._k_exog:] * (i / self.k_regimes))
            else:
                params[self.parameters['autoregressive']] = beta[self._k_exog:]
        # Variance
        if self.switching_variance:
            params[self.parameters['variance']] = (
                np.linspace(variance / 10., variance, num=self.k_regimes))
        else:
            params[self.parameters['variance']] = variance
        return params

    def _em_regime_transition(self, result):
        """
        EM step for regime transition probabilities
        """
        # Marginalize the smoothed joint probabilites to just S_t, S_{t-1} | T
        tmp = result.smoothed_joint_probabilities
        for i in range(tmp.ndim - 3):
            tmp = np.sum(tmp, -2)
        smoothed_joint_probabilities = tmp
        # Transition parameters (recall we're not yet supporting TVTP here)
        k_transition = len(self.parameters[0, 'regime_transition'])
        regime_transition = np.zeros((self.k_regimes, k_transition))
        for i in range(self.k_regimes):  # S_{t_1}
            for j in range(self.k_regimes - 1):  # S_t
                regime_transition[i, j] = (
                        np.sum(smoothed_joint_probabilities[j, i]) /
                        np.sum(result.smoothed_marginal_probabilities[i]))
            delta = np.sum(regime_transition[i]) - 1
            if delta > 0:
                warnings.warn('Invalid regime transition probabilities'
                              ' estimated in EM iteration; probabilities have'
                              ' been re-scaled to continue estimation.',
                              EstimationWarning)
                regime_transition[i] /= 1 + delta + 1e-6
        return regime_transition

    def _em_iteration(self, params0):
        """
        EM iteration

        Notes
        -----
        La iteración EM en esta clase base, solo realiza EM para las probabilidades de transición no-TVTP
        (time-varying transition probabilities).
        """
        params1 = np.zeros(params0.shape,
                           dtype=np.promote_types(np.float64, params0.dtype))

        # Smooth at the given parameters
        result = self.smooth(params0, transformed=True, return_raw=True)

        # The EM with TVTP is not yet supported, just return the previous
        # iteration parameters
        if self.tvtp:
            params1[self.parameters['regime_transition']] = (
                params0[self.parameters['regime_transition']])
        else:
            regime_transition = self._em_regime_transition(result)
            for i in range(self.k_regimes):
                params1[self.parameters[i, 'regime_transition']] = (
                    regime_transition[i])

        tmp = np.sqrt(result.smoothed_marginal_probabilities)
        # Regression coefficients
        coeffs = None
        if self._k_exog > 0:
            coeffs = self._em_exog(result, self.endog, self.exog,
                                   self.parameters.switching['exog'], tmp)
            for i in range(self.k_regimes):
                params1[self.parameters[i, 'exog']] = coeffs[i]
        # Variances
        params1[self.parameters['variance']] = self._em_variance(result, self.endog, self.exog, coeffs, tmp)
        # Regression coefficients
        coeffs = None
        if self._k_exog > 0:
            coeffs = self._em_exog(result, self.endog, self.exog, self.parameters.switching['exog'], tmp)
            for i in range(self.k_regimes):
                params1[self.parameters[i, 'exog']] = coeffs[i]
        # Autoregressive
        if self.order > 0:
            if self._k_exog > 0:
                ar_coeffs, variance = self._em_autoregressive(
                    result, coeffs)
            else:
                ar_coeffs = self._em_exog(
                    result, self.endog, self.exog_ar,
                    self.parameters.switching['autoregressive'])
                variance = self._em_variance(result, self.endog, self.exog_ar, ar_coeffs, tmp)

            for i in range(self.k_regimes):
                params1[self.parameters[i, 'autoregressive']] = ar_coeffs[i]
            params1[self.parameters['variance']] = variance

        return result, params1

    def _em_exog(self, result, endog, exog, switching, tmp=None):
        """
        EM step para los coeficientes de regression.
        """
        k_exog = exog.shape[1]
        coeffs = np.zeros((self.k_regimes, k_exog))
        # First, estimate non-switching coefficients
        if not np.all(switching):
            nonswitching_exog = exog[:, ~switching]
            nonswitching_coeffs = (
                np.dot(np.linalg.pinv(nonswitching_exog), endog))
            coeffs[:, ~switching] = nonswitching_coeffs
            endog = endog - np.dot(nonswitching_exog, nonswitching_coeffs)

        # Next, get switching coefficients
        if np.any(switching):
            switching_exog = exog[:, switching]
            if tmp is None:
                tmp = np.sqrt(result.smoothed_marginal_probabilities)
            for i in range(self.k_regimes):
                tmp_endog = tmp[i] * endog
                tmp_exog = tmp[i][:, np.newaxis] * switching_exog
                coeffs[i, switching] = (np.dot(np.linalg.pinv(tmp_exog), tmp_endog))

        return coeffs

    def _em_autoregressive(self, result, betas, tmp=None):
        """
        EM step para los coeficientes autoregresivos y varianzas.
        """
        if tmp is None:
            tmp = np.sqrt(result.smoothed_marginal_probabilities)

        resid = np.zeros((self.k_regimes, self.nobs + self.order))
        resid[:] = self.orig_endog
        if self._k_exog > 0:
            for i in range(self.k_regimes):
                resid[i] -= np.dot(self.orig_exog, betas[i])

        # The difference between this and `_em_exog` is that here we have a
        # different endog and exog for each regime
        coeffs = np.zeros((self.k_regimes,) + (self.order,))
        variance = np.zeros((self.k_regimes,))
        exog = np.zeros((self.nobs, self.order))
        for i in range(self.k_regimes):
            endog = resid[i, self.order:]
            exog = lagmat(resid[i], self.order)[self.order:]
            tmp_endog = tmp[i] * endog
            tmp_exog = tmp[i][:, None] * exog

            coeffs[i] = np.dot(np.linalg.pinv(tmp_exog), tmp_endog)

            if self.switching_variance:
                tmp_resid = endog - np.dot(exog, coeffs[i])
                variance[i] = (np.sum(
                    tmp_resid ** 2 * result.smoothed_marginal_probabilities[i]) /
                               np.sum(result.smoothed_marginal_probabilities[i]))
            else:
                tmp_resid = tmp_endog - np.dot(tmp_exog, coeffs[i])
                variance[i] = np.sum(tmp_resid ** 2)
            # Variances
        if not self.switching_variance:
            variance = variance.sum() / self.nobs
        return coeffs, variance

    def _fit_em(self, start_params=None, transformed=True, cov_type='none',
                cov_kwds=None, maxiter=50, tolerance=1e-6, full_output=True,
                return_params=False):
        if start_params is None:
            start_params = self.start_params
            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)
        if not transformed:
            start_params = self.transform_params(start_params)
        # Perform expectation-maximization
        llf = []
        params = [start_params]
        i = 0
        delta = 0
        while i < maxiter and (i < 2 or (delta > tolerance)):
            out = self._em_iteration(params[-1])
            llf.append(out[0].llf)
            params.append(out[1])
            if i > 0:
                delta = 2 * (llf[-1] - llf[-2]) / np.abs((llf[-1] + llf[-2]))
            i += 1
        # Just return the fitted parameters if requested
        if return_params:
            result = params[-1]
        return result

    def untransform_params(self, constrained):
        """
        Transforme los parámetros 'constrained' usados en la likelihood evaluation a parámetros 'unconstrained' usados
        por el optimizador.
        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.

        Notes
        -----
        In the base class, this only untransforms the transition-probability-
        related parameters.
        """
        unconstrained = np.array(constrained, copy=True)
        unconstrained = unconstrained.astype(np.promote_types(np.float64, unconstrained.dtype))

        # Nothing to do for transition probabilities if TVTP
        if self.tvtp:
            unconstrained[self.parameters['regime_transition']] = (constrained[self.parameters['regime_transition']])
        # Otherwise reverse logistic transformation
        else:
            for i in range(self.k_regimes):
                s = self.parameters[i, 'regime_transition']
                if self.k_regimes == 2:
                    unconstrained[s] = -np.log(1. / constrained[s] - 1)
                else:
                    from scipy.optimize import root
                    out = root(self._untransform_logistic,
                               np.zeros(unconstrained[s].shape,
                                        unconstrained.dtype),
                               args=(constrained[s],))
                    if not out['success']:
                        raise ValueError('Could not untransform parameters.')
                    unconstrained[s] = out['x']

        # Nothing to do for regression coefficients
        unconstrained[self.parameters['exog']] = (constrained[self.parameters['exog']])
        # Force variances to be positive
        unconstrained[self.parameters['variance']] = (constrained[self.parameters['variance']] ** 0.5)

        for i in range(self.k_regimes):
            s = self.parameters[i, 'autoregressive']
            unconstrained[s] = unconstrain_stationary_univariate(
                constrained[s])

        return unconstrained

    def transform_params(self, unconstrained):
        """
        Transforma parametros 'unconsrained' usados por el optimizador a parametros 'constrained' usados en la evaluacion
        likelihood.
        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evaluation.

        Notes
        -----
        In the base class, this only transforms the transition-probability-
        related parameters.
        """
        constrained = np.array(unconstrained, copy=True)
        constrained = constrained.astype(
            np.promote_types(np.float64, constrained.dtype))

        # Nothing to do for transition probabilities if TVTP
        if self.tvtp:
            constrained[self.parameters['regime_transition']] = (unconstrained[self.parameters['regime_transition']])
        # Otherwise do logistic transformation
        else:
            # Transition probabilities
            for i in range(self.k_regimes):
                tmp1 = unconstrained[self.parameters[i, 'regime_transition']]
                tmp2 = np.r_[0, tmp1]
                constrained[self.parameters[i, 'regime_transition']] = np.exp(
                    tmp1 - logsumexp(tmp2))

        # Ningun cambio para los coeficientes de regression.
        constrained[self.parameters['exog']] = (unconstrained[self.parameters['exog']])
        # Imponiendo condicion para varianza positiva.
        constrained[self.parameters['variance']] = (unconstrained[self.parameters['variance']] ** 2)

        # Autoregressive
        for i in range(self.k_regimes):
            s = self.parameters[i, 'autoregressive']
            constrained[s] = constrain_stationary_univariate(unconstrained[s])

        return constrained

    def regime_transition_matrix(self, params, exog_tvtp=None):
        """
        Construir la matriz de transición estocástica izquierda
        Notes
        -----
        This matrix will either be shaped (k_regimes, k_regimes, 1) or if there
        are time-varying transition probabilities, it will be shaped
        (k_regimes, k_regimes, nobs).

        The (i,j)th element of this matrix is the probability of transitioning
        from regime j to regime i; thus the previous regime is represented in a
        column and the next regime is represented by a row.

        It is left-stochastic, meaning that each column sums to one (because
        it is certain that from one regime (j) you will transition to *some
        other regime*).
        """
        params = np.array(params, ndmin=1)
        if not self.tvtp:
            regime_transition_matrix = np.zeros(
                (self.k_regimes, self.k_regimes, 1),
                dtype=np.promote_types(np.float64, params.dtype))
            regime_transition_matrix[:-1, :, 0] = np.reshape(
                params[self.parameters['regime_transition']],
                (self.k_regimes - 1, self.k_regimes))
            regime_transition_matrix[-1, :, 0] = (
                    1 - np.sum(regime_transition_matrix[:-1, :, 0], axis=0))
        else:
            regime_transition_matrix = (
                self._regime_transition_matrix_tvtp(params, exog_tvtp))
        return regime_transition_matrix

    def initial_probabilities(self, params, regime_transition=None):
        """
        Recuperar probabilidades iniciales
        """
        params = np.array(params, ndmin=1)
        if self._initialization == 'steady-state':
            # if regime_transition is None:
            # regime_transition = self.regime_transition_matrix(params)
            if regime_transition.ndim == 3:
                regime_transition = regime_transition[..., 0]
            m = regime_transition.shape[0]
            A = np.c_[(np.eye(m) - regime_transition).T, np.ones(m)].T
            try:
                probabilities = np.linalg.pinv(A)[:, -1]
            except np.linalg.LinAlgError:
                raise RuntimeError('Steady-state probabilities could not be constructed.')
        elif self._initialization == 'known':
            probabilities = self._initial_probabilities
        else:
            raise RuntimeError('Invalid initialization method selected.')
        # Condición para las probabilidades alejadas de cero (filtros in log space)
        probabilities = np.maximum(probabilities, 1e-20)
        return probabilities

    def predict_conditional(self, params):
        """
        In-sample prediction, conditional on the current and previous regime
        Parameters
        ----------
        params : array_like
            Array of parameters at which to create predictions.
        Returns
        -------
        predict : array_like
            Array of predictions conditional on current, and possibly past,
            regimes
        """
        params = np.array(params, ndmin=1)

        # Prediction is based on:
        # y_t = x_t beta^{(S_t)} +
        #       \phi_1^{(S_t)} (y_{t-1} - x_{t-1} beta^{(S_t-1)}) + ...
        #       \phi_p^{(S_t)} (y_{t-p} - x_{t-p} beta^{(S_t-p)}) + eps_t
        if self._k_exog > 0:
            xb = []
            for i in range(self.k_regimes):
                coeffs = params[self.parameters[i, 'exog']]
                xb.append(np.dot(self.orig_exog, coeffs))

        predict = np.zeros(
            (self.k_regimes,) * (self.order + 1) + (self.nobs,), dtype=np.promote_types(np.float64, params.dtype))
        # Iterate over S_{t} = i
        for i in range(self.k_regimes):
            ar_coeffs = params[self.parameters[i, 'autoregressive']]

            # y_t - x_t beta^{(S_t)}
            ix = self._predict_slices[:]
            ix[0] = i
            ix = tuple(ix)
            if self._k_exog > 0:
                predict[ix] += xb[i][self.order:]

            # Iterate over j = 2, .., p
            for j in range(1, self.order + 1):
                for k in range(self.k_regimes):
                    # This gets a specific time-period / regime slice:
                    # S_{t} = i, S_{t-j} = k, across all other time-period /
                    # regime slices.
                    ix = self._predict_slices[:]
                    ix[0] = i
                    ix[j] = k
                    ix = tuple(ix)

                    start = self.order - j
                    end = -j
                    if self._k_exog > 0:
                        predict[ix] += ar_coeffs[j - 1] * (self.orig_endog[start:end] - xb[k][start:end])
                    else:
                        predict[ix] += ar_coeffs[j - 1] * (self.orig_endog[start:end])
        return predict

    def _resid(self, params):
        return self.endog - self.predict_conditional(params)

    def _conditional_loglikelihoods(self, params):
        """
        Calcula loglikelihoods condicional del régimen del período actual y los últimos sel.order regímen
        """
        # Get the residuals
        resid = self._resid(params)
        # Compute the conditional likelihoods
        variance = params[self.parameters['variance']].squeeze()
        if self.switching_variance:
            variance = np.reshape(variance, (self.k_regimes, 1, 1))
        conditional_loglikelihoods = (-0.5 * resid ** 2 / variance - 0.5 * np.log(2 * np.pi * variance))

        return conditional_loglikelihoods

    def _filter(self, params, regime_transition=None):
        # Get the regime transition matrix if not provided
        if regime_transition is None:
            regime_transition = self.regime_transition_matrix(params)
        # Get the initial probabilities
        initial_probabilities = self.initial_probabilities(params, regime_transition)
        # Compute the conditional likelihoods
        conditional_loglikelihoods = self._conditional_loglikelihoods(params)
        # Apply the filter
        return ((regime_transition, initial_probabilities,
                 conditional_loglikelihoods) +
                cy_hamilton_filter_log(
                    initial_probabilities, regime_transition,
                    conditional_loglikelihoods, self.order))

    def _smooth(self, params, predicted_joint_probabilities_log,
                filtered_joint_probabilities_log, regime_transition=None):
        # Get the regime transition matrix
        if regime_transition is None:
            regime_transition = self.regime_transition_matrix(params)

        # Apply the smoother
        return cy_kim_smoother_log(regime_transition,
                                   predicted_joint_probabilities_log,
                                   filtered_joint_probabilities_log)

    @property
    def _res_classes(self):
        return {'fit': (MarkovSwitchingResults, MarkovSwitchingResultsWrapper)}

    def _wrap_results(self, params, result, return_raw, cov_type=None,
                      cov_kwds=None, results_class=None, wrapper_class=None):
        if not return_raw:
            # Wrap in a results object
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            if cov_kwds is not None:
                result_kwargs['cov_kwds'] = cov_kwds

            if results_class is None:
                results_class = self._res_classes['fit'][0]
            if wrapper_class is None:
                wrapper_class = self._res_classes['fit'][1]

            res = results_class(self, params, result, **result_kwargs)
            result = wrapper_class(res)
        return result

    def smooth(self, params, transformed=True, cov_type=None, cov_kwds=None, return_raw=False, results_class=None,
               results_wrapper_class=None):
        """
        Aplicación de 'Kim smoother' y 'Hamilton filter'

        Parameters
        ----------
        params : array_like
            Array of parameters at which to perform filtering.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        cov_type : str, optional
            See `fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `fit` for a description of required keywords for alternative
            covariance estimators
        return_raw : bool,optional
            Whether or not to return only the raw Hamilton filter output or a
            full results object. Default is to return a full results object.
        results_class : type, optional
            A results class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.
        results_wrapper_class : type, optional
            A results wrapper class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.
        Returns
        -------
        MarkovSwitchingResults
        """
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)
        # Save the parameter names
        self.data.param_names = self.param_names
        # Hamilton filter
        names = ['regime_transition', 'initial_probabilities',
                 'conditional_loglikelihoods',
                 'filtered_marginal_probabilities',
                 'predicted_joint_probabilities', 'joint_loglikelihoods',
                 'filtered_joint_probabilities',
                 'predicted_joint_probabilities_log',
                 'filtered_joint_probabilities_log']
        result = Bunch(**dict(zip(names, self._filter(params))))
        # Kim smoother
        out = self._smooth(params, result.predicted_joint_probabilities_log,
                           result.filtered_joint_probabilities_log)
        result['smoothed_joint_probabilities'] = out[0]
        result['smoothed_marginal_probabilities'] = out[1]
        result = KimSmootherResults(self, result)

        # Wrap in a results object
        return self._wrap_results(params, result, return_raw, cov_type, cov_kwds, results_class, results_wrapper_class)

    def loglikeobs(self, params, transformed=True):
        """
        Loglikelihood evaluation para cada periodo
        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)
        if not transformed:
            params = self.transform_params(params)
        results = self._filter(params)
        return results[5]

    def loglike(self, params, transformed=True):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        return np.sum(self.loglikeobs(params, transformed))

    def score(self, params, transformed=True):
        """
        Compute the score function at params.
        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)
        return approx_fprime_cs(params, self.loglike, args=(transformed,))

    def score_obs(self, params, transformed=True):
        """
        Calcule el puntaje por observación, evaluado en los parámetros
        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)
        return approx_fprime_cs(params, self.loglikeobs, args=(transformed,))

    def hessian(self, params, transformed=True):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters
        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the Hessian
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)

        return approx_hess_cs(params, self.loglike)

    def fit(self, start_params=None, transformed=True, cov_type='approx',
            cov_kwds=None, method='bfgs', maxiter=100, full_output=1, disp=0,
            callback=None, return_params=False, em_iter=5, search_reps=0,
            search_iter=5, search_scale=1., **kwargs):
        if start_params is None:
            start_params = self.start_params
            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)
        # Get better start params through EM algorithm
        if em_iter and not self.tvtp:
            start_params = self._fit_em(start_params, transformed=transformed,
                                        maxiter=em_iter, tolerance=0,
                                        return_params=True)
            transformed = True

        if transformed:
            start_params = self.untransform_params(start_params)

        # Maximum likelihood estimation by scoring
        fargs = (False,)
        mlefit = super(MarkovSwitching_FAE, self).fit(start_params, method=method,
                                                           fargs=fargs,
                                                           maxiter=maxiter,
                                                           full_output=full_output,
                                                           disp=disp, callback=callback,
                                                           skip_hessian=True, **kwargs)

        # Just return the fitted parameters if requested
        if return_params:
            result = self.transform_params(mlefit.params)
        # Otherwise construct the results class if desired
        else:
            result = self.smooth(mlefit.params, transformed=False, cov_type=cov_type, cov_kwds=cov_kwds)
            result.mlefit = mlefit
            result.mle_retvals = mlefit.mle_retvals
            result.mle_settings = mlefit.mle_settings
        return result

    def summary(self, alpha=.05, start=None, title=None, model_name=None):
        """
        Resumen del modelo.
        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.
        title : str, optional
            The title of the summary table.
        model_name : str
            The name of the model used. Default is to use model class name.
        display_params : bool, optional
            Whether or not to display tables of estimated parameters. Default
            is True. Usually only used internally.
        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.
        """
        from statsmodels.iolib.summary import Summary

        # Model specification results
        model = self.model
        if title is None:
            title = 'Markov Switching Model Results'
        if start is None:
            start = 0
        if self.data.dates is not None:
            dates = self.data.dates
            d = dates[start]
            sample = ['%02d-%02d-%02d' % (d.month, d.day, d.year)]
            d = dates[-1]
            sample += ['- ' + '%02d-%02d-%02d' % (d.month, d.day, d.year)]
        else:
            sample = [str(start), ' - ' + str(self.model.nobs)]
        # Standardize the model name as a list of str
        if model_name is None:
            model_name = model.__class__.__name__

        # Create the tables
        if not isinstance(model_name, list):
            model_name = [model_name]

        top_left = [('Dep. Variable:', None)]
        top_left.append(('Model:', [model_name[0]]))
        for i in range(1, len(model_name)):
            top_left.append(('', ['+ ' + model_name[i]]))
        top_left += [
            ('Date:', None),
            ('Time:', None),
            ('Sample:', [sample[0]]),
            ('', [sample[1]])
        ]
        top_right = [
            ('No. Observations:', [self.model.nobs]),
            ('Log Likelihood', ["%#5.3f" % self.llf]),
            ('AIC', ["%#5.3f" % self.aic]),
            ('BIC', ["%#5.3f" % self.bic]),
            ('HQIC', ["%#5.3f" % self.hqic])
        ]

        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))

        summary = Summary()
        summary.add_table_2cols(self, gleft=top_left, gright=top_right, title=title)

        # Make parameters tables for each regime
        from statsmodels.iolib.summary import summary_params
        import re

        def make_table(self, mask, title, strip_end=True):
            res = (self, self.params[mask], self.bse[mask], self.tvalues[mask], self.pvalues[mask],
                   self.conf_int(alpha)[mask])

            param_names = [
                re.sub(r'\[\d+\]$', '', name) for name in
                np.array(self.data.param_names)[mask].tolist()
            ]

            return summary_params(res, yname=None, xname=param_names, alpha=alpha, use_t=False, title=title)

        params = model.parameters
        regime_masks = [[] for i in range(model.k_regimes)]
        other_masks = {}
        for key, switching in params.switching.items():
            k_params = len(switching)
            if key == 'regime_transition':
                continue
            other_masks[key] = []

            for i in range(k_params):
                if switching[i]:
                    for j in range(self.k_regimes):
                        regime_masks[j].append(params[j, key][i])
                else:
                    other_masks[key].append(params[0, key][i])

        for i in range(self.k_regimes):
            mask = regime_masks[i]
            if len(mask) > 0:
                table = make_table(self, mask, 'Regime %d parameters' % i)
                summary.tables.append(table)

        mask = []
        for key, _mask in other_masks.items():
            mask.extend(_mask)
        if len(mask) > 0:
            table = make_table(self, mask, 'Non-switching parameters')
            summary.tables.append(table)
        # Transition parameters
        mask = params['regime_transition']
        table = make_table(self, mask, 'Regime transition parameters')
        summary.tables.append(table)
        # Add warnings/notes, added to text format only
        etext = []
        if hasattr(self, 'cov_type') and 'description' in self.cov_kwds:
            etext.append(self.cov_kwds['description'])
        if self._rank < len(self.params):
            etext.append("Covariance matrix is singular or near-singular,"
                         " with condition number %6.3g. Standard errors may be"
                         " unstable." % np.linalg.cond(self.cov_params()))

        if etext:
            etext = ["[{0}] {1}".format(i + 1, text)
                     for i, text in enumerate(etext)]
            etext.insert(0, "Warnings:")
            summary.add_extra_txt(etext)
        return summary


class MarkovSwitchingResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'cov_params_approx': 'cov',
        'cov_params_default': 'cov',
        'cov_params_opg': 'cov',
        'cov_params_robust': 'cov',
    }
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {
        'forecast': 'dates',
    }
    _wrap_methods = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods)

wrap.populate_wrapper(MarkovSwitchingResultsWrapper,  # noqa:E305
                      MarkovSwitchingResults)
