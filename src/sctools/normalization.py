#   Author: Cyrille L. Delley
#
#
#
#
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse, io
from scipy.stats.mstats import gmean
from scipy.linalg import helmert
from sklearn.decomposition import TruncatedSVD
import statsmodels.api as sm
import anndata as ad

def GLM_regression(andat, covariables:list, antibodies:list=[], components:int=1, offset:bool=False, force_zeros:bool=False):
    """Perform general linear model regression of each Ab vector with provided covariables.
    currently supported covariables are a subset from: [IgG1','amplicon','ab_raw','ab_umi']
    Covariables are SVD transformed and the first N components are retained (default 1)

    currently uses linear regression on log transformed data and a Gaussian error model."""
    
    if antibodies == []:
        antibodies = list(andat.var.index)
    
    # add pseudo count, log transform, normalize and column center the covariable matrix before SVD 
    covar = np.log(np.array(andat.obs[covariables])+1)
    covar = covar / covar.max(axis=0)
    U,s,Vt= np.linalg.svd(covar - covar.mean(axis=0), full_matrices=False)

    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.plot(s, 'o')
    plt.show()

    # use left eigenvectors for regression, correlation is scale invariant
    if offset:
        factors = sm.add_constant(U[:,:components], prepend=True)
    else:
        factors = U[:,:components]

    # if some Ab's have all zero counts or were not mesured, the regression cannot be performed on the raw matrix
    # skip columns with all zero in regression
    Ab_filter = andat.X.sum(axis=0) != 0

    fit_val = np.zeros([len(U), len(antibodies)])
    residuals = np.zeros([len(U), len(antibodies)])
    params = np.zeros([len(antibodies),factors.shape[1]])
    for i in range(len(antibodies)):
        if Ab_filter[i]:
            linear_model_result = sm.GLM(np.log(andat.X[:,i]+1), factors, family=sm.families.Gaussian()).fit()
            fit_val[:,i] = linear_model_result.fittedvalues
            residuals[:,i] = linear_model_result.resid_response
            params[i] = linear_model_result.params


    andat_new = ad.AnnData(X=residuals, obs=andat.obs, var=andat[:,antibodies].var)
    if force_zeros:
        zero_pos = andat.X == 0
        for v in zero_pos:
            andat_new[:,v].X == 1.
    return andat_new

def compositional_transform(andat_raw, add_pseudocount:bool=False, features:list=[]):
    """calculated the three Aitchison geometry transforms for the Ab counts.
    - alr uses the IgG1 counts as universal reference
    - ilr contrasts are based on the SVD of clr

    if add_pseudocount is set to true add 1 to prevent zero division, otherwise
    cells with zero counts in the denominator create inf and have to be filtered
    before downstream analysis, e.g.:
        clr_filter = np.isfinite(clr).all(axis=1)
        clr = clr[clr_filter,:]"""
        
    if features == []:
        features = list(andat_raw.var.index)    
    
    if add_pseudocount:
        clr_data = np.log((1+andat_raw[:,features].X)/gmean(andat_raw[:,features].X+1, axis=1).reshape(-1,1))
        alr_data = np.log((1+andat_raw[:,features].X)/(andat_raw[:,features].X[:,-1]+1).reshape(-1,1))

        U,s,Vt = np.linalg.svd(clr_data, full_matrices=False)
        ilr_data = np.dot(U*s,helmert(len(s)).T)
    else:
        clr_data = np.log(andat_raw[:,features].X/gmean(andat_raw[:,features].X, axis=1).reshape(-1,1))
        alr_data = np.log(andat_raw[:,features].X/(andat_raw[:,features].X[:,-1].reshape(-1,1)))

        finite_clr = np.isfinite(clr_data).all(axis=1)
        U,s,Vt = np.linalg.svd(clr_data[finite_clr,:], full_matrices=False)
        ilr_data = np.dot(U*s,helmert(len(s)).T)
    return alr_data, clr_data, ilr_data
