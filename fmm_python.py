#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#implements the fully endogenized finite mixture model described here: https://github.com/syedmfuad/fmm

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load data
path = "XXXXXXXXXXXX"
data = pd.read_csv(path + "HHdata.csv")

# Convert Price to Y
Y = data["Price"] / 100000

# House attributes
X = pd.DataFrame({
    "Intercept": 1,
    "Square Foot": data["SquareFoot"] / 1000,
    "Lot Size": data["Lot"] / 1000,
    "House Age": data["HouseAge"],
    "Garage": data["Garage"],
    "Bird": data["ExpBird"]
})

# Demographic variables/mixing variables
Z = pd.DataFrame({
    "Intercept": 1,
    "Educ": data["Educ"],
    "Inc": data["Inc"] / 10000,
    "Age": data["Age"],
    "HHSize": data["HHSize"]
})

# Initialize OLS
ols_agg = np.linalg.lstsq(X.values, Y, rcond=None)[0]

# Starting values for the hedonic estimates/betas for each type
beta_start = ols_agg

# Starting values for the gamma estimates for the demographic variables
gamma_start = np.array([0.01] * Z.shape[1])

# Starting values for sigma
sigma_start = np.sqrt(np.mean(np.square(Y - np.dot(X, beta_start))))

# Collecting initializing values
val_start = np.concatenate((beta_start, gamma_start, sigma_start))

vals = val_start
types = 2

# Convergence criteria
Iter_conv = 0.0001
j = types

# Number of independent variables or beta estimates
niv = X.shape[1]

# Number of demographic variables
gvs = Z.shape[1]

# Row dimension of aggregate
n = X.shape[0]
conv_cg = 5000
conv_cb = 5000

# Define probability density function FnOne
def FnOne(par, x, y):
    mean = np.dot(x, par[1:])
    return np.exp(-(y - mean) ** 2 / (2 * par[0] ** 2)) / (par[0] * np.sqrt(2 * np.pi))

# Define the max probability densities over type probabilities (FnTwo)
def FnTwo(par, d, x, y):
    pdy = np.zeros((n, j))
    b = par[:niv * j].reshape((niv, j))
    s = par[niv * j:niv * j + j]
    for i in range(j):
        pdy[:, i] = FnOne(np.concatenate(([s[i]], b[:, i])), x, y)
    return np.sum(np.log(np.dot(d, pdy)))

# Define the logit for gamma estimates (FnThree)
def FnThree(g, z):
    L = np.exp(np.dot(z, g))
    return L

# Define the max gamma estimates and type probabilities (FnFour)
def FnFour(par, d, z, y):
    L = np.zeros((n, j))
    L[:, 0] = 1
    for m in range(j - 1):
        L[:, m + 1] = FnThree(par[m * gvs:(m + 1) * gvs], z)
    Pi = L / np.sum(L, axis=1, keepdims=True)
    return np.sum(np.sum(np.log(d * Pi)))

# Mixing algorithm
def FMM(par, X, Z, y):
    b = par[:niv * j].reshape((niv, j))
    g = par[niv * j:niv * j + j * gvs]
    s = par[niv * j + j * gvs:]
    L = np.zeros((n, j))
    f = np.zeros((n, j))
    d = np.zeros((n, j))
    iter = 0

    while np.abs(conv_cg) + np.abs(conv_cb) > Iter_conv:
        beta_old = b.copy()
        gamma_old = g.copy()
        iter += 1

        for i in range(j):
            f[:, i] = FnOne(np.concatenate(([s[i]], b[:, i])), X, Y)

        for i in range(j - 1):
            L[:, 0] = 0
            L[:, i + 1] = np.dot(Z, g[i * gvs:(i + 1) * gvs])

        P = np.exp(L) / (1 + np.sum(np.exp(L), axis=1, keepdims=True))

        for i in range(n):
            d[i, :] = P[i, :] * f[i, :] / np.sum(P[i, :] * f[i, :])

        b1 = b.flatten()
        par1 = np.concatenate((b1, s))
        beta_m = minimize(FnTwo, par1, args=(d, X, Y), method='BFGS', options={'disp': False})
        b = beta_m.x[:niv * j].reshape((niv, j))
        s = beta_m.x[niv * j:]
        gam_m = minimize(FnFour, g, args=(d, Z, Y), method='BFGS', options={'disp': False})
        g = gam_m.x

        conv_cg = np.sum(np.abs(g - gamma_old))
        conv_cb = np.sum(np.abs(b - beta_old))

        par2 = np.concatenate((b1, s))
        LL = FnTwo(par2, d, X, Y) + FnFour(g, d, Z, Y)

        bvector = b.flatten()
        vals_fin = np.concatenate((bvector, g, s))
        dvector = d

    out_pars = {"vals_fin": vals_fin, "i_type": d}
    return out_pars

# Calling FMM
mix = FMM(val_start, X, Z, Y)

# Final updating
d = mix["i_type"]
b = mix["vals_fin"][:niv * j].reshape((niv, j))
g = mix["vals_fin"][niv * j:niv * j + j * gvs]
s = mix["vals_fin"][niv * j + j * gvs:]

b1 = b.flatten()
par3 = np.concatenate((b1, s))

# Standard errors for beta
def hessian(FnTwo, par, d, x, y):
    h = np.zeros((par.shape[0], par.shape[0]))
    eps = 1e-6
    for i in range(par.shape[0]):
        for j in range(par.shape[0]):
            par_plus = par.copy()
            par_plus[i] += eps
            par_plus[j] += eps
            h[i, j] = (FnTwo(par_plus, d, x, y) - FnTwo(par_plus, d, x, y) - FnTwo(par, d, x, y) + FnTwo(par, d, x, y)) / (eps ** 2)
    return h

hess = hessian(FnTwo, par3, d, X, Y)
inv_hess = np.linalg.inv(hess)
bse = np.sqrt(np.diagonal(inv_hess))

# Standard errors for gamma
def hessian(FnFour, par, d, z, y):
    h = np.zeros((par.shape[0], par.shape[0]))
    eps = 1e-6
    for i in range(par.shape[0]):
        for j in range(par.shape[0]):
            par_plus = par.copy()
            par_plus[i] += eps
            par_plus[j] += eps
            h[i, j] = (FnFour(par_plus, d, z, y) - FnFour(par_plus, d, z, y) - FnFour(par, d, z, y) + FnFour(par, d, z, y)) / (eps ** 2)
    return h

hess = hessian(FnFour, g, d, Z, Y)
inv_hess = np.linalg.inv(hess)
gse = np.sqrt(np.diagonal(inv_hess))

LL = FnTwo(par2, d, X, Y) + FnFour(g, d, Z, Y)

