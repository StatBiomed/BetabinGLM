import math
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import betaln, gammaln


# sigmoid function
def sigmoid(x):
    a = []
    for i in np.nditer(x):
        if i >= 0:
            a.append(1.0/(1+math.exp(-i)))
        else:
            a.append(math.exp(i)/(1+math.exp(i)))
    return np.array(a)

# cost function used for optimizing the weights
def cost_fun_for_W(W, endog, exog, phi):
    Z = sigmoid(np.inner(W, exog))
    k = np.array(endog[:, 0])
    n = np.array(endog[:, 0] + endog[:, 1])
    a = Z * (1/phi - 1)
    b = (1 - Z) * (1/phi - 1)
    cost = np.sum(betaln(k + a, n - k + b) - betaln(a, b))
    return -(cost/np.size(Z))

# cost function used for optimizing phi
def cost_fun_for_phi(phi, endog, exog, W):
    Z = sigmoid(np.inner(W, exog))
    k = np.array(endog[:, 0])
    n = np.array(endog[:, 0] + endog[:, 1])
    a = Z * (1/phi - 1)
    b = (1 - Z) * (1/phi - 1)
    cost = np.sum(betaln(k + a, n - k + b) - betaln(a, b))
    return -(cost/np.size(Z))

# predict the output given the parameters
def predict(self):
    W = self.W
    Z = sigmoid(np.dot(self.exog, W))
    n = np.array(self.endog[:, 0] + self.endog[:, 1])
    pred = [(Z[i] * n[i], (1 - Z[i]) * n[i]) for i in range (len(Z))]
    ypred = np.array(pred)
    return ypred

# calculate the log likelihood after optimization
def get_loglikelihood(self):
    W = self.W
    phi = self.phi
    Z = sigmoid(np.dot(self.exog, W))
    k = np.array(self.endog[:, 0])
    n = np.array(self.endog[:, 0] + self.endog[:, 1])
    a = Z * (1/phi - 1)
    b = (1 - Z) * (1/phi - 1)
    return np.sum(betaln(k + a, n - k + b) - betaln(a, b)) + np.sum(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

# optimize the parameters once
def fit_trial(self, phi, W):
    bounds_W = [(-np.inf, np.inf) for i in range (self.exog.shape[1])] 
    res1 = minimize(fun = cost_fun_for_W, x0 = W, args = (self.endog, self.exog, phi), bounds = bounds_W, method = self.method) 
    W = res1.x
    bounds_phi = [(0, 1)]
    res2 = minimize(fun = cost_fun_for_phi, x0 = phi, args = (self.endog, self.exog, W), bounds = bounds_phi, method = self.method) 
    return res1.x, res2.x

#  calculate the log likelihood during optimization
def get_LL(self, W, phi):
    Z = sigmoid(np.dot(self.exog, W))
    k = np.array(self.endog[:, 0])
    n = np.array(self.endog[:, 0] + self.endog[:, 1])
    a = Z * (1/phi - 1)
    b = (1 - Z) * (1/phi - 1)
    return np.sum(betaln(k + a, n - k + b) - betaln(a, b)) + np.sum(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

# beta-binomial regression
def fit(self):
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

class betabin():
    def __init__(self, endog, exog, fit_intercept = True, method = 'Nelder-Mead'):
        if fit_intercept:
            exog = sm.add_constant(exog, prepend = False)
        self.exog = exog
        self.endog = endog
        self.method = method
        self.phi, self.W = fit(self)
        self.predict = predict(self)
        self.NLL = -get_loglikelihood(self)
        if self.NLL > 0:
            self.NLL = np.nan
            print("Positive LL is obtained by error in optimization. Change another optimizer to avoid this problem.") 
