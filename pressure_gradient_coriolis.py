""" Compute pressure gradient and Coriolis force """

import jax.numpy as jnp
import namelist_n_constants as nl


def pressure_gradient_force(pi0, pip, theta, x3d, y3d, z3d):
    """ Calculate the pressure gradient force for the momentum equations

    Assuming the input arrays have ghost points.
    """
    pi_total = pip    # if pi0 is horizontally uniform
    # pi_total = pi0 + pip    
    # if pi0 is not horizontally uniform, it is probably better to calculate the pressure gradient force due to
    # base state in advance, instead of repeating the calculation here.
    dpi_dx = (pi_total[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
              pi_total[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) / (
              x3d[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
              x3d[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    pres_grad4u = -nl.Cp * 0.5 * (theta[nl.ngx:-(nl.ngx-1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] +
                                  theta[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) * dpi_dx

    dpi_dy = (pi_total[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
              pi_total[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz]) / (
              y3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] -
              y3d[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz])
    pres_grad4v = -nl.Cp * 0.5 * (theta[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] +
                                  theta[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz]) * dpi_dy

    dpi_dz = (pip[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
              pip[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, 0:-nl.ngz]) / (
              z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
              z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, 0:-nl.ngz])
    pres_grad4w = -nl.Cp * 0.5 * (theta[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] +
                                  theta[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, 0:-nl.ngz]) * dpi_dz
    pres_grad4w = pres_grad4w.at[:, :, (0, -1)].set(0.0)
    return pres_grad4u, pres_grad4v, pres_grad4w


def calculate_coriolis_force(u, v):
    """ Calculate the Coriolis force, putting them at u and v points """
    fv = nl.fCor * 0.25 * (v[nl.ngx-1:-nl.ngx, nl.ngy:-(nl.ngy+1), nl.ngz:-nl.ngz] +
                           v[nl.ngx-1:-nl.ngx, nl.ngy+1:-nl.ngy, nl.ngz:-nl.ngz] +
                           v[nl.ngx:-(nl.ngx-1), nl.ngy:-(nl.ngy+1), nl.ngz:-nl.ngz] +
                           v[nl.ngx:-(nl.ngx-1), nl.ngy+1:-nl.ngy, nl.ngz:-nl.ngz]
                           )

    fu = nl.fCor * 0.25 * (u[nl.ngx:-(nl.ngx+1), nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz] +
                           u[nl.ngx:-(nl.ngx+1), nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] +
                           u[nl.ngx+1:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz] +
                           u[nl.ngx+1:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz]
                           )
    # du/dt = fv + ...
    # dv/dt = -fu + ...
    return fu, fv


def calculate_coriolis_force_with_lspgrad(u, v):
    """ Calculate the Coriolis force, putting them at u and v points """
    fv = nl.fCor * (0.25 * (v[nl.ngx-1:-nl.ngx, nl.ngy:-(nl.ngy+1), nl.ngz:-nl.ngz] +
                           v[nl.ngx-1:-nl.ngx, nl.ngy+1:-nl.ngy, nl.ngz:-nl.ngz] +
                           v[nl.ngx:-(nl.ngx-1), nl.ngy:-(nl.ngy+1), nl.ngz:-nl.ngz] +
                           v[nl.ngx:-(nl.ngx-1), nl.ngy+1:-nl.ngy, nl.ngz:-nl.ngz]
                           ) - 0.0)

    fu = nl.fCor * (0.25 * (u[nl.ngx:-(nl.ngx+1), nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz] +
                           u[nl.ngx:-(nl.ngx+1), nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz] +
                           u[nl.ngx+1:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz] +
                           u[nl.ngx+1:-nl.ngx, nl.ngy:-(nl.ngy-1), nl.ngz:-nl.ngz]
                           ) - 1.0)
    # du/dt = fv + ...
    # dv/dt = -fu + ...
    return fu, fv

