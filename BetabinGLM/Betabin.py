import numpy as np
import scipy as sp
import statsmodels.api as sm

## Sigmoid function
def sigmoid(x):
    return (1/(1 + np.exp(-x)))

## Dot product of weight and independent variables
def pro(w, x):
    return sigmoid(np.inner(w, x))

## Coefficient alpha in beta-binomial prior
def alpha(p, phi):
    return (((1 / phi) - 1) * p)

## Coefficient beta in beta-binomial prior
def beta(p, phi):
    return (((1 / phi) - 1) * (1 - p))

## Average cost of beta-binomial regression excluding the constant 
def cost(f, exog, endog):
    cos = 0
    for n in range(len(exog)):
        p = pro(f[:-1], exog[n])
        a = alpha(p, f[-1])
        b = beta(p, f[-1])
        cos += sp.special.betaln(endog[n][0] + a, endog[n][1] + b) - sp.special.betaln(a, b)
    return - cos / len(exog)

## Predict the output given the parameters
def predict(f, exog, endog):
    p_pred = sigmoid(np.matmul(exog, f[:-1]))
    return np.concatenate(([p_pred * (endog[:, 0] + endog[:, 1])], [(1 - p_pred) * (endog[:, 0] + endog[:, 1])]), axis = 0).T

## log-likelihood of the data
def LL(f, exog, endog):
    return - cost(f, exog, endog) * len(exog) + sum(sp.special.gammaln(endog[:,0] + endog[:,1] + 1) - sp.special.gammaln(endog[:,0] + 1) - sp.special.gammaln(endog[:,1] + 1))

# Finding the best initial guess for phi
def initial(params, exog, endog):
    guess = np.arange(0.99999, 10, 0.1)
    cos = [0]
    while np.argmin(cos) == 0:
        guess = guess / 10
        cos = []
        for m in range(len(guess)): 
            f = np.concatenate((params, [guess[m]]))
            cos.append(cost(f, exog, endog))
    f = np.concatenate((params, [guess[np.argmin(cos)]]))
    return f

# Beta-binomial regression
def fit(self):
    glm_binom = sm.GLM(self.endog, self.exog, family = sm.families.Binomial())
    res = glm_binom.fit()
        
    bnds = []
    for m in range(len(res.params)):
        bnds.append((- np.inf, np.inf))
    bnds.append((0,1))
    bnds_t = tuple(bnds)
        
    f = initial(res.params, self.exog, self.endog)
    ress = sp.optimize.minimize(cost, f, args = (self.exog, self.endog), method = self.method, bounds = bnds_t)
    return ress.x

class betabin:
    def __init__(self, exog, endog, fit_intercept = True, method = "SLSQP"):
        self.n_features_in_ = len(exog[0])
        self.intercept_ = fit_intercept
        if self.intercept_:
            exog = sm.add_constant(exog, prepend = False)
        self.exog = exog
        self.endog = endog
        self.method = method
        self.params = fit(self)
        self.predict = predict(self.params, self.exog, self.endog)
        LL_try = LL(self.params, self.exog, self.endog)
        if LL_try > 0:
            self.LL = np.nan
            print("Positive LL is obtained by error in optimization. Change another optimizer to avoid this problem") 
        else:
            self.LL = LL_try
