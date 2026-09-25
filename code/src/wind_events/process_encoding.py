"""Train-only encodings for independently observed physical-process decoding.

Wang and Oates (2015), arXiv:1506.00327, angular summation/difference fields:
GASF is even under global sign reversal, whereas GADF is odd. With the
study's zero event-start coordinate, the corresponding GADF row equals x.
Block-energy balancing and protected DCT coordinates are explicit study
ablations, not claims to have invented angular difference fields.
"""
import numpy as np
from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .representation import gasf,polarity_bit


def gadf(x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all() or (np.abs(x)>1+1e-6).any():raise ValueError('Finite unit-domain paths required')
    x=np.clip(x,-1,1);c=np.sqrt(np.maximum(0,1-x*x))
    return c[:,:,None]*x[:,None,:]-x[:,:,None]*c[:,None,:]


class ProcessEncodings:
    def fit(self,x):
        x=np.asarray(x,float)
        self.raw_scale=StandardScaler().fit(x)
        self.raw_pca=PCA(12,svd_solver='full').fit(self.raw_scale.transform(x))
        g=gasf(x).reshape(len(x),-1);a=gadf(x).reshape(len(x),-1)
        self.g_scale=StandardScaler().fit(g);self.a_scale=StandardScaler().fit(a)
        g=self.g_scale.transform(g);a=self.a_scale.transform(a);r=self.raw_scale.transform(x)
        self.energy=np.sqrt([np.mean(np.sum(z*z,axis=1)) for z in [g,a,r]])
        self.g_pca=PCA(6,svd_solver='randomized',random_state=41).fit(g)
        self.a_pca=PCA(6,svd_solver='randomized',random_state=41).fit(a)
        self.dual_pca=PCA(6,svd_solver='randomized',random_state=41).fit(np.c_[g/self.energy[0],a/self.energy[1]])
        self.balanced_pca=PCA(6,svd_solver='randomized',random_state=41).fit(np.c_[g/self.energy[0],r/self.energy[2]])
        return self

    def transform(self,x):
        x=np.asarray(x,float);r=self.raw_scale.transform(x)
        g=self.g_scale.transform(gasf(x).reshape(len(x),-1))
        a=self.a_scale.transform(gadf(x).reshape(len(x),-1))
        gp=self.g_pca.transform(g);rp=self.raw_pca.transform(r)
        coefficients=dct(x[:,4:21],type=2,norm='ortho',axis=1)
        return {'raw25':x,'raw_pca6':rp[:,:6],'raw_pca12':rp,
            'gaf6':gp,'gaf5_bit':np.c_[gp[:,:5],polarity_bit(x)],
            'gadf6':self.a_pca.transform(a),
            'dual_gaf6':self.dual_pca.transform(np.c_[g/self.energy[0],a/self.energy[1]]),
            'balanced_signed6':self.balanced_pca.transform(np.c_[g/self.energy[0],r/self.energy[2]]),
            'phase_protected6':np.c_[gp[:,:3],coefficients[:,:3]],
            'phase_protected12':np.c_[gp,coefficients[:,:6]]}
