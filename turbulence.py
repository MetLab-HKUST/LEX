""" Conventional turbulence schemes """

import numpy as np
import jax.numpy as jnp
import namelist_n_constants as nl
import pressure_equations as pres_eqn
import one_step_integration as one


def compute_smag(rho, km, kh, s11, s22, s33, s12, s13, s23, theta, qv, x3d, y3d, z3d, x3d4u, y3d4v, z3d4w):
    """ Compute the subgrid-scale turbulence tendencies using the Smagorinsky model """
    tau11 = -2.0 * km * s11    # density included
    tau22 = -2.0 * km * s22
    tau33 = -2.0 * km * s33
    tau12 = -2.0 * km * s12
    tau13 = -2.0 * km * s13
    tau23 = -2.0 * km * s23

    dtau11_dx = (tau11[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
                 tau11[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) / (
                 x3d[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
                 x3d[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    dtau12_dy8v = (tau12[nl.ngx-1:-(nl.ngx-1), nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                   tau12[nl.ngx-1:-(nl.ngx-1), nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz]) / (
                   y3d[nl.ngx-1:-(nl.ngx-1), nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                   y3d[nl.ngx-1:-(nl.ngx-1), nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz])
    dtau12_dy = 0.25 * (dtau12_dy8v[0:-1, 0:-1, :] + dtau12_dy8v[1:, 1:, :] +
                        dtau12_dy8v[0:-1, 1:, :] + dtau12_dy8v[1:, 0:-1, :])
    tau13_8w = pres_eqn.interpolate_scalar2w_0(tau13[:, :, nl.ngz:-nl.ngz])
    dtau13_dz8s = (tau13_8w[:, :, 1:] - tau13_8w[:, :, 0:-1]) / (
                   z3d4w[:, :, nl.ngz+1:-nl.ngz] - z3d4w[:, :, nl.ngz:-(nl.ngz+1)])
    dtau13_dz = 0.5 * (dtau13_dz8s[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, :] -
                       dtau13_dz8s[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, :])
    sgs_u = -(dtau11_dx + dtau12_dy + dtau13_dz) / (0.5 * (
              rho[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] +
              rho[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]))
    
    dtau22_dy = (tau22[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                 tau22[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz]) / (
                 y3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                 y3d[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz])
    dtau12_dx8u = (tau12[nl.ngx:-(nl.ngx-1), nl.ngy-1:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                   tau12[nl.ngx-1:-nl.ngx, nl.ngy-1:-(nl.ngy-1), nl.ngz:-nl.ngz]) / (
                   x3d[nl.ngx:-(nl.ngx-1), nl.ngy-1:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                   x3d[nl.ngx-1:-nl.ngx, nl.ngy-1:-(nl.ngy-1), nl.ngz:-nl.ngz])
    dtau12_dx = 0.25 * (dtau12_dx8u[0:-1, 0:-1, :] + dtau12_dx8u[1:, 1:, :] +
                        dtau12_dx8u[0:-1, 1:, :] + dtau12_dx8u[1:, 0:-1, :])
    tau23_8w = pres_eqn.interpolate_scalar2w_0(tau23[:, :, nl.ngz:-nl.ngz])
    dtau23_dz8s = (tau23_8w[:, :, 1:] - tau23_8w[:, :, 0:-1]) / (
                   z3d4w[:, :, nl.ngz+1:-nl.ngz] - z3d4w[:, :, nl.ngz:-(nl.ngz+1)])
    dtau23_dz = 0.5 * (dtau23_dz8s[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, :] -
                       dtau23_dz8s[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), :])
    sgs_v = -(dtau12_dx + dtau22_dy + dtau23_dz) / (0.5 * (
              rho[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] +
              rho[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz]))

    dtau33_dz = (tau33[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
                 tau33[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, 0:-nl.ngz]) / (
                 z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
                 z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, 0:-nl.ngz])
    dtau13_dx8s = (tau13[nl.ngx+1:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
                   tau13[nl.ngx-1:-(nl.ngx+1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) / (
                   x3d[nl.ngx+1:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
                   x3d[nl.ngx-1:-(nl.ngx+1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    dtau13_dx = pres_eqn.interpolate_scalar2w_0(dtau13_dx8s)              
    dtau23_dy8s = (tau23[nl.ngx:-nl.ngx, nl.ngy+1:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                   tau23[nl.ngx:-nl.ngx, nl.ngy-1:-(nl.ngy+1), nl.ngz:-nl.ngz]) / (
                   y3d[nl.ngx:-nl.ngx, nl.ngy+1:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                   y3d[nl.ngx:-nl.ngx, nl.ngy-1:-(nl.ngy+1), nl.ngz:-nl.ngz])
    dtau23_dy = pres_eqn.interpolate_scalar2w_0(dtau23_dy8s)
    rho8w = pres_eqn.interpolate_scalar2w(rho[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    sgs_w = -(dtau13_dx + dtau23_dy + dtau33_dz) / rho8w
    sgs_w = sgs_w.at[:, :, (0,-1)].set(0.0)

    # calculate the SGS tendency for theta
    rho8u, rho8v, rho8w = stagger_scalar(rho)
    kh8u, kh8v, kh8w = stagger_scalar(kh)
    dth_dx, dth_dy, dth_dz = compute_scalar_gradient(theta, x3d, y3d, z3d)
    tau_th_x = -rho8u * kh8u * dth_dx  
    tau_th_y = -rho8v * kh8v * dth_dy
    tau_th_z = -rho8w * kh8w * dth_dz
    tau_th_z = tau_th_z.at[:, :, (0,-1)].set(0.0)
    sgs_theta = -((tau_th_x[1:,:,:] - tau_th_x[0:-1,:,:]) / (
                   x3d4u[nl.ngx+1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
                   x3d4u[nl.ngx:-(nl.ngx+1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) +
                  (tau_th_y[:,1:,:] - tau_th_y[:,0:-1,:]) / (
                   y3d4v[nl.ngx:-nl.ngx, nl.ngy+1:-nl.ngy, nl.ngz:-nl.ngz] -
                   y3d4v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy+1), nl.ngz:-nl.ngz]) +
                  (tau_th_z[:,:,1:] - tau_th_z[:,:,0:-1]) / (
                   z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] -
                   z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)])) / rho[
                       nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]

    # calculate the SGS tendency for qv
    dqv_dx, dqv_dy, dqv_dz = compute_scalar_gradient(qv, x3d, y3d, z3d)
    tau_qv_x = -rho8u * kh8u * dqv_dx  
    tau_qv_y = -rho8v * kh8v * dqv_dy
    tau_qv_z = -rho8w * kh8w * dqv_dz
    tau_qv_z = tau_qv_z.at[:, :, (0,-1)].set(0.0)
    sgs_qv = -((tau_qv_x[1:,:,:] - tau_qv_x[0:-1,:,:]) / (
                   x3d4u[nl.ngx+1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
                   x3d4u[nl.ngx:-(nl.ngx+1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) +
                  (tau_qv_y[:,1:,:] - tau_qv_y[:,0:-1,:]) / (
                   y3d4v[nl.ngx:-nl.ngx, nl.ngy+1:-nl.ngy, nl.ngz:-nl.ngz] -
                   y3d4v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy+1), nl.ngz:-nl.ngz]) +
                  (tau_qv_z[:,:,1:] - tau_qv_z[:,:,0:-1]) / (
                   z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] -
                   z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)])) / rho[
                       nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]

    return sgs_u, sgs_v, sgs_w, sgs_theta, sgs_qv
    

def compute_deformation(rho, u, v, w, x3d, y3d, z3d, x3d4u, y3d4v, z3d4w):
    """ Compute teh strain rate terms

    S_ij = 0.5 * ( d(u_i)/d(x_j) + d(u_j)/d(x_i) )
    (note: multiplied by density herein) and then uses these variables to calculate deformation (which does not
    include the density factor).
    """
    s11 = rho[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * (
            u[nl.ngx+1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
            u[nl.ngx:-(nl.ngx+1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) / (
            x3d4u[nl.ngx+1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
            x3d4u[nl.ngx:-(nl.ngx+1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    s22 = rho[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * (
            v[nl.ngx:-nl.ngx, nl.ngy+1:-nl.ngy, nl.ngz:-nl.ngz] -
            v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy+1), nl.ngz:-nl.ngz]) / (
            y3d4v[nl.ngx:-nl.ngx, nl.ngy+1:-nl.ngy, nl.ngz:-nl.ngz] -
            y3d4v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy+1), nl.ngz:-nl.ngz])
    s33 = rho[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * (
            w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] -
            w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)]) / (
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] -
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)])
    # s11, s22, s33 are at scalar points
    dudy_edge = (u[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                 u[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz]) / (0.5 * (
                 y3d[nl.ngx:-(nl.ngx-1), nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                 y3d[nl.ngx:-(nl.ngx-1), nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz] +
                 y3d[nl.ngx-1:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                 y3d[nl.ngx-1:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz]))
    dvdx_edge = (v[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
                 v[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) / (0.5 * (
                 x3d[nl.ngx:-(nl.ngx-1), nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
                 x3d[nl.ngx-1:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] +
                 x3d[nl.ngx:-(nl.ngx-1), nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz] -
                 x3d[nl.ngx-1:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz]))
    s12_edge = 0.5 * (dudy_edge + dvdx_edge)
    s12 = rho[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * (0.25 * (
        s12_edge[0:-1, 0:-1, :] + s12_edge[1:, 1:, :] + s12_edge[0:-1, 1:, :] + s12_edge[1:, 0:-1, :]))
    # put s12 to scalar points as well
    dudz_edge = (u[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] -
                 u[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)]) / (0.5 * (
                 z3d[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] -
                 z3d[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)] +
                 z3d[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] -
                 z3d[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)]))
    dwdx_edge = (w[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz+1:-(nl.ngz+1)] -
                 w[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-(nl.ngz+1)]) / (0.5 * (
                 x3d[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] -
                 x3d[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] +
                 x3d[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)] -
                 x3d[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)]))
    s13_edge = 0.5 * (dudz_edge + dwdx_edge)
    bottom0, top0 = pres_eqn.extrapolate_bottom_top(s13_edge)    # bottom and top are at scalar levels
    bottom = 0.5 * (bottom0[1:, :, :] + bottom0[0:-1, :, :])
    top = 0.5 * (top0[1:, :, :] + top0[0:-1, :, :])
    s13_part = 0.25 * (s13_edge[0:-1, :, 0:-1] + s13_edge[1:, :, 1:] + s13_edge[0:-1, :, 1:] + s13_edge[1:, :, 0:-1])
    s13 = jnp.concatenate((bottom, s13_part, top), axis=2)
    s13 = rho[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * s13
    # put s13 to scalar points
    dvdz_edge = (v[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] -
                 v[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)]) / (0.5 * (
                 z3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz+1:-nl.ngz] -
                 z3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-(nl.ngz+1)] +
                 z3d[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz+1:-nl.ngz] -
                 z3d[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-(nl.ngz+1)]))
    dwdy_edge = (w[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz+1:-(nl.ngz+1)]  -
                 w[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz+1:-(nl.ngz+1)] ) / (0.5 * (
                 y3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz+1:-nl.ngz] -
                 y3d[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz+1:-nl.ngz] +
                 y3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-(nl.ngz+1)] -
                 y3d[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-(nl.ngz+1)]))
    s23_edge = 0.5 * (dvdz_edge + dwdy_edge)
    bottom0, top0 = pres_eqn.extrapolate_bottom_top(s23_edge)    # bottom and top are at scalar levels
    bottom = 0.5 * (bottom0[:, 1:, :] + bottom0[:, 0:-1, :])
    top = 0.5 * (top0[:, 1:, :] + top0[:, 0:-1, :])
    s23_part = 0.25 * (s23_edge[:, 0:-1, 0:-1] + s23_edge[:, 1:, 1:] + s23_edge[:, 0:-1, 1:] + s23_edge[:, 1:, 0:-1])
    s23 = jnp.concatenate((bottom, s23_part, top), axis=2)
    s23 = rho[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * s23
    # put s23 to scalar points
    deformation = (2.0 * (s11**2 + s22**2 + s33**2) + 4.0 * (s12**2 + s13**2 + s23**2)) / (
        rho[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])**2

    s11 = one.padding3_array(s11)
    s22 = one.padding3_array(s22)
    s33 = one.padding3_array(s33)
    s12 = one.padding3_array(s12)
    s13 = one.padding3_array(s13)
    s23 = one.padding3_array(s23)
    deformation = one.padding3_array(deformation)

    return s11, s22, s33, s12, s13, s23, deformation


def compute_nm(theta, qv, z3d):
    """ Calculate the squared Brunt-Vaisala frequency """
    theta_rho =  theta * (1.0 + nl.repsm1*qv)
    n28w = jnp.log(theta_rho[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] /
                 theta_rho[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)]) / (
                 z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz+1:-nl.ngz] -
                 z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz+1)]) * nl.g
    bottom, top = pres_eqn.extrapolate_bottom_top(n28w)    # bottom and top are at scalar levels
    n2_part = 0.5 * (n28w[:, :, 0:-1] + n28w[:, :, 1:])
    n2 = jnp.concatenate((bottom, n2_part, top), axis=2)
    n2 = one.padding3_array(n2)

    return n2


def compute_k(s2, n2, x3d4u, y3d4v, z3d4w):
    """ Calculate the Smagorinsky eddy viscosity and diffusivity """
    prandtl = 1.0/3.0
    cs = 0.18
    grid_scale = jnp.cbrt((x3d4u[1:, :, :] -
                          x3d4u[0:-1, :, :]) * (
                          y3d4v[:, 1:, :] -
                          y3d4v[:, 0:-1, :]) * (
                          z3d4w[:, :, 1:] -
                          z3d4w[:, :, 0:-1]))

    s2n2 = s2 - n2/prandtl
    s2n2 = jnp.where(s2n2>0.0, s2n2, 0.0)

    km = (cs * grid_scale)**2 * jnp.sqrt(s2n2)
    kh = km/prandtl
    return km, kh


def compute_scalar_gradient(scalar, x3d, y3d, z3d):
    """ Calculate the gradient for a scalar

    Assuming the input arrays have ghost points.
    """
    ds_dx8u = (scalar[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
               scalar[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) / (
               x3d[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
               x3d[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
   
    ds_dy8v = (scalar[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
               scalar[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz]) / (
               y3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
               y3d[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz])

    ds_dz8w = (scalar[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
               scalar[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, 0:-nl.ngz]) / (
               z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
               z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, 0:-nl.ngz])
    ds_dz8w = ds_dz8w.at[:, :, (0, -1)].set(0.0)
    return ds_dx8u, ds_dy8v, ds_dz8w


def stagger_scalar(rho):
    """ Put a scalar to staggered grid points """
    rho8u = 0.5 * (rho[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + 
                   rho[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    rho8v = 0.5 * (rho[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] +
                   rho[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz])
    rho8w = pres_eqn.interpolate_scalar2w(rho[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    return rho8u, rho8v, rho8w
