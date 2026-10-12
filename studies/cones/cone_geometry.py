"""Stable full-dimensional normalized cone gaps for archived finite fits."""
import base64
import numpy as np


def unpack(blob):
    if blob['dtype'] != 'float32-le':
        raise ValueError('unexpected coordinate dtype')
    return np.frombuffer(base64.b64decode(blob['base64']),dtype='<f4').reshape(blob['shape']).astype(np.float64)


def gaps(u,rho,k):
    u=np.asarray(u,dtype=np.float64)
    rho=np.asarray(rho,dtype=np.float64)
    if u.ndim!=2 or rho.shape!=(len(u),) or not np.isfinite(u).all() or not np.isfinite(rho).all():
        raise ValueError('nonfinite or invalid cone state')
    if np.any(rho<=0) or np.any(2*k/np.sinh(rho)>=1):
        raise ValueError('invalid cone aperture domain')
    c=np.clip(u@u.T,-1,1)
    one_minus=np.sum((u[:,None,:]-u[None,:,:])**2,axis=-1)/2
    transverse_direction=np.linalg.norm(u[None,:,:]-c[:,:,None]*u[:,None,:],axis=-1)
    axial=np.sinh(rho[None,:]-rho[:,None])-np.cosh(rho[:,None])*np.sinh(rho[None,:])*one_minus
    transverse=np.sinh(rho[None,:])*transverse_direction
    length=np.hypot(axial,transverse)
    np.fill_diagonal(length,1)
    cos_angle=axial/length
    angle_loss=np.where(axial>=0,(transverse/length)**2/np.maximum(1+cos_angle,1e-300),1-cos_angle)
    sine=2*k/np.sinh(rho)
    clearance=sine*sine/(1+np.sqrt(1-sine*sine))
    gap=angle_loss/clearance[:,None]-1
    np.fill_diagonal(gap,np.nan)
    return gap


def state_gap(state,k):
    u=unpack(state['directions'])
    u/=np.linalg.norm(u,axis=1,keepdims=True)
    rho=unpack(state['rho'])
    return gaps(u,rho,k)
