import math
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import betaln, gammaln


def sigmoid(x):
    """sigmoid function

    Returns: np.array(a) containing all the sigmoid values of input array x
    """
    a = []
    for i in np.nditer(x):
        if i >= 0:
            a.append(1.0/(1+math.exp(-i)))
        else:
            a.append(math.exp(i)/(1+math.exp(i)))
    return np.array(a)

def cost_fun_for_W(W, endog, exog, phi):
    """cost function used for optimizing the weights

        Z (np.array): sigmoid values of current prediction
        k (np.array): the number of positive events
        n (np.array): the number of total events
        phi (double): overdispersion value, equals to 1/(1 + alpha + beta)
        a (double): alpha
        b (double): beta

    Returns:
        np.array: value of cost function under current parameter values

    Note: 
        Some constants occured in the cost function have been ignored here as it won't affect the function trend. 
        This function is same as the function cost_fun_for_W. Seperating them into two is just for preventing the scipy.optimize.minimize to mess up the parameter of optimization.
    """
    Z = sigmoid(np.inner(W, exog))
    k = np.array(endog[:, 0])
    n = np.array(endog[:, 0] + endog[:, 1])
    a = Z * (1/phi - 1)
    b = (1 - Z) * (1/phi - 1)
    cost = np.sum(betaln(k + a, n - k + b) - betaln(a, b))
    return -(cost/np.size(Z))

def cost_fun_for_phi(phi, endog, exog, W):
    """cost function used for optimizing phi

        Z (np.array): sigmoid values of current prediction
        k (np.array): the number of positive events
        n (np.array): the number of total events
        phi (double): overdispersion value, equals to 1/(1 + alpha + beta)
        a (double): alpha
        b (double): beta

    Returns:
        np.array: value of cost function under current parameter values

    Note: 
        Some constants occured in the cost function have been ignored here as it won't affect the function trend. 
        This function is same as the function cost_fun_for_phi. Seperating them into two is just for preventing the scipy.optimize.minimize to mess up the parameter of optimization.
    """
    Z = sigmoid(np.inner(W, exog))
    k = np.array(endog[:, 0])
    n = np.array(endog[:, 0] + endog[:, 1])
    a = Z * (1/phi - 1)
    b = (1 - Z) * (1/phi - 1)
    cost = np.sum(betaln(k + a, n - k + b) - betaln(a, b))
    return -(cost/np.size(Z))

def predict(self):
    """predict the output given the parameters

        Z (np.array): sigmoid values of current prediction
        n (np.array): the number of total events
        pred (list): a (2 * number of cases) list of predicted beta-binomial distribution with current optimized weight
        ypred (np.array): ypred in the type of np.array

    Returns:
        ypred(np.array): predicted beta-binomial distributions
    """
    W = self.W
    Z = sigmoid(np.dot(self.exog, W))
    n = np.array(self.endog[:, 0] + self.endog[:, 1])
    pred = [(Z[i] * n[i], (1 - Z[i]) * n[i]) for i in range (len(Z))]
    ypred = np.array(pred)
    return ypred

def get_loglikelihood(self):
    """calculate the log likelihood after optimization

        Z (np.array): sigmoid values of current prediction
        k (np.array): the number of positive events
        n (np.array): the number of total events
        phi (double): overdispersion value, equals to 1/(1 + alpha + beta)
        a (double): alpha
        b (double): beta

        Returns: 
            the log-likelihood value
    """
    W = self.W
    phi = self.phi
    Z = sigmoid(np.dot(self.exog, W))
    k = np.array(self.endog[:, 0])
    n = np.array(self.endog[:, 0] + self.endog[:, 1])
    a = Z * (1/phi - 1)
    b = (1 - Z) * (1/phi - 1)
    return np.sum(betaln(k + a, n - k + b) - betaln(a, b)) + np.sum(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

def fit_trial(self, phi, W):
    """optimize the parameter once

    First optimize the weight (fix the value of phi), then optimize phi (fix the weight to the optimized one)

        bounds_W (list): the bounds set for optimizing W using scipy.optimize.minimize
        res1: OptimizeResult of W
        W (np.array): optimized result
        bounds_W (list): the bound set for optimizing phi using scipy.optimize.minimize
        res2: OptimizeResult of phi
    
    Returns:
        np.array, np.array: the optimized result of the weight and the phi

    Note: 
        for more about the OptimizeResult using scipy.optimize.minimize, please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    bounds_W = [(-np.inf, np.inf) for i in range (self.exog.shape[1])] 
    res1 = minimize(fun = cost_fun_for_W, x0 = W, args = (self.endog, self.exog, phi), bounds = bounds_W, method = self.method) 
    W = res1.x
    bounds_phi = [(0, 1)]
    res2 = minimize(fun = cost_fun_for_phi, x0 = phi, args = (self.endog, self.exog, W), bounds = bounds_phi, method = self.method) 
    return res1.x, res2.x

def get_LL(self, W, phi):
    """calculate the log likelihood during optimization

        Z (np.array): sigmoid values of current prediction
        k (np.array): the number of positive events
        n (np.array): the number of total events
        phi (double): overdispersion value, equals to 1/(1 + alpha + beta)
        a (double): alpha
        b (double): beta

    Returns: 
        double: the log-likelihood value

    Note:
        The calculation is same as the function get_loglikelihood. But, during the optimization process, the intermediate values of W and phi will not be assigned to self. Therefore, this additional function is necessary as it contains additional args W and phi.
    """
    Z = sigmoid(np.dot(self.exog, W))
    k = np.array(self.endog[:, 0])
    n = np.array(self.endog[:, 0] + self.endog[:, 1])
    a = Z * (1/phi - 1)
    b = (1 - Z) * (1/phi - 1)
    return np.sum(betaln(k + a, n - k + b) - betaln(a, b)) + np.sum(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

def fit(self):
    """ beta-binomial regression

    An expectation-maximization like method for beta-binomial regression. The function will keep optimization until the negative log-likelihood change is smaller than 1e-7

    NLL_trend (list): store the negative log-likelihood after each optimization trial

    Returns:
        np.array, np.array: the optimized value of W and phi

    """
    phi = 0.1
    W = np.array([0 for i in range (self.exog.shape[1])])
    NLL_trend = []
    while True:
        if (len(NLL_trend) > 1):
            if (abs(NLL_trend[-1] - NLL_trend[-2]) < 1e-7):
                break
        W, phi = fit_trial(self, phi, W)
        NLL_trend.append(-get_LL(self, W, phi))
    return W, phi

class BetaBinomial():
    def __init__(self, endog, exog, fit_intercept = True, method = 'Nelder-Mead'):
        if fit_intercept:
            exog = sm.add_constant(exog, prepend = False)
        self.exog = exog
        self.endog = endog
        self.method = method
        self.W, self.phi = fit(self)
        self.predict = predict(self)
        self.LL = get_loglikelihood(self)
        if self.LL > 0:
            self.LL = np.nan
            print("Positive LL is obtained by error in optimization. Change another optimizer to avoid this problem.") 
