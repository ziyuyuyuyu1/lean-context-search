"""Differentiable stable tangent-component cones; isolated follow-up variant."""
import math

import torch

K = 0.1
RHO_MIN, RHO_MAX = 0.4, 2.0
NORM_MARGIN = 0.1


def cone_terms(us, ur, rs, rr, geometry, *, k=K):
    """Vectors are unit directions; source/result arrays have matching leading shape."""
    if us.shape != ur.shape or us.shape[-1] != 128 or rs.shape != rr.shape or rs.shape != us.shape[:-1]:
        raise ValueError('Cone shape mismatch')
    if not 0 < k < RHO_MIN / 2 or not bool(torch.isfinite(us).all() and torch.isfinite(ur).all()
                                            and torch.isfinite(rs).all() and torch.isfinite(rr).all()):
        raise ArithmeticError('Nonfinite or invalid cone parameters')
    if bool(((rs < RHO_MIN-1e-6) | (rs > RHO_MAX+1e-6) |
             (rr < RHO_MIN-1e-6) | (rr > RHO_MAX+1e-6)).any()):
        raise ArithmeticError('Radius outside declared bounds')
    c=(us*ur).sum(-1)
    if bool((c.abs()>1+1e-4).any()):raise ArithmeticError('Invalid direction cosine')
    c=c.clamp(-1,1)  # Only tiny float error after the explicit domain check.
    t=0.5*(us-ur).square().sum(-1)
    transverse_direction=torch.linalg.vector_norm(ur-c[...,None]*us,dim=-1)
    if geometry=='euclidean':
        axial=(rr-rs)-rr*t
        transverse=rr*transverse_direction
        sine=2*k/rs
    elif geometry=='hyperbolic':
        axial=torch.sinh(rr-rs)-torch.cosh(rs)*torch.sinh(rr)*t
        transverse=torch.sinh(rr)*transverse_direction
        sine=2*k/torch.sinh(rs)
    else:raise ValueError('Unknown geometry')
    if bool(((sine<=0)|(sine>=1)).any()):raise ArithmeticError('Aperture outside valid domain')
    aperture_cos=torch.sqrt(1-sine.square())
    clearance=sine.square()/(1+aperture_cos)
    tangent_length=torch.hypot(axial,transverse)
    threshold=1e-6 if us.dtype==torch.float32 else 1e-12
    if bool((tangent_length<=threshold).any()):
        raise ArithmeticError('Distinct/too-close points: undefined angular relation')
    angle_cos=axial/tangent_length
    # Make the inactive rationalized branch finite at an exactly reversed ray.
    rational_denominator=torch.where(axial>=0,1+angle_cos,torch.ones_like(angle_cos))
    one_minus_angle=torch.where(axial>=0,
                                (transverse/tangent_length).square()/rational_denominator,
                                1-angle_cos)
    normalized_gap=one_minus_angle/clearance-1
    raw_gap=normalized_gap*clearance
    if not bool(torch.isfinite(normalized_gap).all()):raise ArithmeticError('Nonfinite normalized gap')
    return {'normalized_gap':normalized_gap,'raw_gap':raw_gap,'clearance':clearance,
            'angle_cos':angle_cos,'aperture_cos':aperture_cos,
            'axial':axial,'transverse':transverse,'tangent_length':tangent_length,
            'cosine_clamp_count':int((c.abs()>=1).sum().detach().cpu())}


def signed_loss(positive,negative,margin=NORM_MARGIN):
    if positive.numel()==0 or negative.numel()==0 or margin<=0:raise ValueError('Both labels and positive margin required')
    return .5*torch.relu(positive+margin).mean()+.5*torch.relu(margin-negative).mean()


def old_signed_loss(positive,negative,margin=.02):
    return .5*torch.relu(positive+margin).mean()+.5*torch.relu(margin-negative).mean()


def clearance_bound(rho,geometry,k=K):
    rho=torch.as_tensor(rho)
    sine=2*k/(rho if geometry=='euclidean' else torch.sinh(rho))
    return sine.square()/(1+torch.sqrt(1-sine.square()))
