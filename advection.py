""" Functions for calculating the advection terms """

import jax
import jax.numpy as jnp
import numpy as np
import namelist_n_constants as nl


def advection_scalar(rho0, scalar, u, v, w, weps, flow_divergence, scalar_sfc_flux, x3d4u, y3d4v, z3d4w):
    """ Compute the advection term for a scalar

    3rd-order WENO is used to compute the vertical flux, and 5th-order WENO is used to compute the horizontal fluxes.
    """
    flux_z = vertical_flux_scalar2(weps, rho0, w, scalar)
    flux_z = flux_z.at[:, :, 0].set(scalar_sfc_flux)  # set lower boundary condition

    flux_x, flux_y = horizontal_flux_scalar(weps, rho0, u, v, scalar)

    scalar_convergence = -((flux_x[1:, :, :] - flux_x[0:-1, :, :]) /
                           (x3d4u[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
                            x3d4u[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) +
                           (flux_y[:, 1:, :] - flux_y[:, 0:-1, :]) /
                           (y3d4v[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz:-nl.ngz] -
                            y3d4v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz:-nl.ngz]) +
                           (flux_z[:, :, 1:] - flux_z[:, :, 0:-1]) /
                           (z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz + 1:-nl.ngz] -
                            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz + 1)])
                           )

    adv_tendency = (scalar_convergence + scalar[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] *
                    flow_divergence[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]
                    ) / rho0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]

    return adv_tendency


def advection_u(rho0, u, v, w, weps, flow_divergence, u_sfc_flux, x3d, y3d4v, z3d4w):
    """ Compute the advection term for u velocity

    3rd-order WENO is used to compute the vertical flux, and 5th-order WENO is used to compute the horizontal fluxes.
    """
    flux_z = vertical_flux_u2(weps, rho0, w, u)
    _, flx_size_y = jnp.shape(u_sfc_flux)
    tau_x_west = jnp.reshape(u_sfc_flux[-1, :], (1, flx_size_y))
    tau_x_east = jnp.reshape(u_sfc_flux[0, :], (1, flx_size_y))
    tau_x = jnp.concatenate((tau_x_west, u_sfc_flux, tau_x_east), axis=0)
    tau_x8u = 0.5 * (tau_x[0:-1, :] + tau_x[1:, :])
    flux_z = flux_z.at[:, :, 0].set(tau_x8u)  # set lower boundary condition

    flux_x, flux_y = horizontal_flux_u(weps, rho0, u, v)

    u_convergence = -((flux_x[1:, :, :] - flux_x[0:-1, :, :]) /
                      (x3d[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
                       x3d[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) +
                      (flux_y[:, 1:, :] - flux_y[:, 0:-1, :]) / (0.5 * (
                       (y3d4v[nl.ngx:-(nl.ngx - 1), nl.ngy + 1:-nl.ngy, nl.ngz:-nl.ngz] -
                        y3d4v[nl.ngx:-(nl.ngx - 1), nl.ngy:-(nl.ngy + 1), nl.ngz:-nl.ngz]) +
                       (y3d4v[nl.ngx - 1:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz:-nl.ngz] -
                        y3d4v[nl.ngx - 1:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz:-nl.ngz]))) +
                      (flux_z[:, :, 1:] - flux_z[:, :, 0:-1]) / (0.5 * (
                       (z3d4w[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz + 1:-nl.ngz] -
                        z3d4w[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz + 1)]) +
                       (z3d4w[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz + 1:-nl.ngz] -
                        z3d4w[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz + 1)])))
                      )

    divergence8u = 0.5 * (flow_divergence[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] +
                          flow_divergence[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    rho08u = 0.5 * (rho0[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] +
                    rho0[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])

    adv_tendency = (u_convergence + u[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * divergence8u) / rho08u

    return adv_tendency


def advection_v(rho0, u, v, w, weps, flow_divergence, v_sfc_flux, x3d4u, y3d, z3d4w):
    """ Compute the advection term for v velocity

    3rd-order WENO is used to compute the vertical flux, and 5th-order WENO is used to compute the horizontal fluxes.
    """
    flux_z = vertical_flux_v2(weps, rho0, w, v)
    flx_size_x, _ = jnp.shape(v_sfc_flux)
    tau_y_south = jnp.reshape(v_sfc_flux[:, -1], (flx_size_x, 1))
    tau_y_north = jnp.reshape(v_sfc_flux[:, 0], (flx_size_x, 1))
    tau_y = jnp.concatenate((tau_y_south, v_sfc_flux, tau_y_north), axis=1)
    tau_y8v = 0.5 * (tau_y[:, 0:-1] + tau_y[:, 1:])
    flux_z = flux_z.at[:, :, 0].set(tau_y8v)  # set lower boundary condition

    flux_x, flux_y = horizontal_flux_v(weps, rho0, u, v)

    v_convergence = -((flux_x[1:, :, :] - flux_x[0:-1, :, :]) / (0.5 * (
                       (x3d4u[nl.ngx + 1:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz] -
                        x3d4u[nl.ngx:-(nl.ngx + 1), nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz]) +
                       (x3d4u[nl.ngx + 1:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz] -
                        x3d4u[nl.ngx:-(nl.ngx + 1), nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz]))) +
                      (flux_y[:, 1:, :] - flux_y[:, 0:-1, :]) /
                      (y3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz] -
                       y3d[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz]) +
                      (flux_z[:, :, 1:] - flux_z[:, :, 0:-1]) / (0.5 * (
                       (z3d4w[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz + 1:-nl.ngz] -
                        z3d4w[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-(nl.ngz + 1)]) +
                       (z3d4w[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz + 1:-nl.ngz] -
                        z3d4w[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-(nl.ngz + 1)])))
                      )

    divergence8v = 0.5 * (flow_divergence[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz] +
                          flow_divergence[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz])
    rho08v = 0.5 * (rho0[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz] +
                    rho0[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz])

    adv_tendency = (v_convergence + v[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * divergence8v) / rho08v

    return adv_tendency


def advection_w(rho0, u, v, w, weps, flow_divergence, x3d4u, y3d4v, z3d):
    """ Compute the advection term for v velocity

    3rd-order WENO is used to compute the vertical flux, and 5th-order WENO is used to compute the horizontal fluxes.
    """
    flux_z = vertical_flux_w2(weps, rho0, w)
    flux_x, flux_y = horizontal_flux_w(weps, rho0, u, v, w)

    w_convergence = -((flux_x[1:, :, :] - flux_x[0:-1, :, :]) / (0.5 * (
                       (x3d4u[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz-1:-nl.ngz] -
                        x3d4u[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy, nl.ngz-1:-nl.ngz]) +
                       (x3d4u[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
                        x3d4u[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy, nl.ngz:]))) +
                      (flux_y[:, 1:, :] - flux_y[:, 0:-1, :]) / (0.5 * (
                       (y3d4v[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz-1:-nl.ngz] -
                        y3d4v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz-1:-nl.ngz]) +
                       (y3d4v[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz:] -
                        y3d4v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz:]))) +
                      (flux_z[:, :, 1:] - flux_z[:, :, 0:-1]) /
                      (z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
                       z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz-1:-nl.ngz])
                      )

    divergence8w = 0.5 * (flow_divergence[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz-1:-nl.ngz] +
                          flow_divergence[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:])
    rho08w = 0.5 * (rho0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz-1:-nl.ngz] +
                    rho0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:])

    adv_tendency = (w_convergence + w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * divergence8w) / rho08w
    adv_tendency = adv_tendency.at[:, :, 0].set(0.0)  # set lower boundary tendency to zero, correct for flat terrain
    adv_tendency = adv_tendency.at[:, :, -1].set(0.0)  # set top boundary tendency to zero

    return adv_tendency


def advection_pi(rho0, pi0, pip, u, v, w, x3d, y3d, z3d, cc1, cc2):
    """ low-order scheme to compute the advection term for pi'.
     
    Because we need to compute it in acoustic steps, we use a low-order scheme here 
    to save computational cost.
    """
    # if base state pi0 is not horizontally uniform, we should compuate dpi0_dx and dpi0_dy
    # and add them to the total gradient
    dpi_dx = (pip[nl.ngx:-nl.ngx+1, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] - 
              pip[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) / (
                  x3d[nl.ngx:-nl.ngx+1, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] - 
                  x3d[nl.ngx-1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])

    dpi_dy = (pip[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy+1, nl.ngz:-nl.ngz] -
              pip[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz]) / (
                  y3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy+1, nl.ngz:-nl.ngz] -
                  y3d[nl.ngx:-nl.ngx, nl.ngy-1:-nl.ngy, nl.ngz:-nl.ngz])

    dpi0_dz = (pi0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
               pi0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz-1:-nl.ngz]) / (
                   z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
                   z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz-1:-nl.ngz])

    dpip_dz = (pip[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
               pip[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz-1:-nl.ngz]) / (
                   z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
                   z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz-1:-nl.ngz])
    dpi_dz = dpi0_dz + dpip_dz
               
    u_dpi_dx = u[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * dpi_dx
    v_dpi_dy = v[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * dpi_dy
    w_dpi_dz = w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * dpi_dz
    w_dpi_dz = w_dpi_dz.at[:, :, 0].set(0.0)  # set lower and upper boundary flux to zero, correct for flat terrain
    w_dpi_dz = w_dpi_dz.at[:, :, -1].set(0.0) 

    rho08w = 0.5 * (rho0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz-1:-nl.ngz] +
                    rho0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:])

    adv_pi_tendency = -(0.5 * (u_dpi_dx[1:, :, :] + u_dpi_dx[0:-1, :, :]) +
                        0.5 * (v_dpi_dy[:, 1:, :] + v_dpi_dy[:, 0:-1, :]) +
                        (cc1 * w_dpi_dz[:, :, 0:-1] * rho08w[:, :, 0:-1] +
                         cc2 * w_dpi_dz[:, :, 1:] * rho08w[:, :, 1:])
                         ) / rho0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]

    return adv_pi_tendency


def get_divergence(rho0, u, v, w, x3d4u, y3d4v, z3d4w):
    """ Compute the divergence of (rho0*u, rho0*v, rho0*w) """
    rho8u_part = 0.5 * (rho0[0:-1, :, :] + rho0[1:, :, :])
    rho8v_part = 0.5 * (rho0[:, 0:-1, :] + rho0[:, 1:, :])
    rho8w_part = 0.5 * (rho0[:, :, 0:-1] + rho0[:, :, 1:])
    _, u_size_y, u_size_z = rho8u_part.shape
    v_size_x, _, v_size_z = rho8v_part.shape
    west = jnp.reshape(rho8u_part[-1, :, :], (1, u_size_y, u_size_z))
    east = jnp.reshape(rho8u_part[0, :, :], (1, u_size_y, u_size_z))
    rho8u = jnp.concatenate((west, rho8u_part, east), axis=0)  # periodic boundary
    south = jnp.reshape(rho8v_part[:, -1, :], (v_size_x, 1, v_size_z))
    north = jnp.reshape(rho8v_part[:, 0, :], (v_size_x, 1, v_size_z))
    rho8v = jnp.concatenate((south, rho8v_part, north), axis=1)  # periodic boundary
    w_size_x, w_size_y, _ = rho8w_part.shape
    zero4w = jnp.zeros((w_size_x, w_size_y, 1))
    rho8w = jnp.concatenate((zero4w, rho8w_part, zero4w), axis=2)

    rho_u = rho8u * u
    rho_v = rho8v * v
    rho_w = rho8w * w

    div_x = (rho_u[1:, :, :] - rho_u[0:-1, :, :]) / (x3d4u[1:, :, :] - x3d4u[0:-1, :, :])
    div_y = (rho_v[:, 1:, :] - rho_v[:, 0:-1, :]) / (y3d4v[:, 1:, :] - y3d4v[:, 0:-1, :])
    div_z = (rho_w[:, :, 1:] - rho_w[:, :, 0:-1]) / (z3d4w[:, :, 1:] - z3d4w[:, :, 0:-1])
    div_rho_u = div_x + div_y + div_z  # ghost points kept

    return div_rho_u


def get_divergence2(u, v, w, x3d4u, y3d4v, z3d4w):
    """ Compute the divergence of (u, v, w) """
    div_x = (u[1:, :, :] - u[0:-1, :, :]) / (x3d4u[1:, :, :] - x3d4u[0:-1, :, :])
    div_y = (v[:, 1:, :] - v[:, 0:-1, :]) / (y3d4v[:, 1:, :] - y3d4v[:, 0:-1, :])
    div_z = (w[:, :, 1:] - w[:, :, 0:-1]) / (z3d4w[:, :, 1:] - z3d4w[:, :, 0:-1])
    div_u = div_x + div_y + div_z  # ghost points kept

    return div_u


def get_2d_divergence(rho0, u, v, x3d4u, y3d4v):
    """ Compute the 2D divergence of (rho0*u, rho0*v) """
    rho8u_part = 0.5 * (rho0[0:-1, :, :] + rho0[1:, :, :])
    rho8v_part = 0.5 * (rho0[:, 0:-1, :] + rho0[:, 1:, :])
    _, u_size_y, u_size_z = rho8u_part.shape
    v_size_x, _, v_size_z = rho8v_part.shape
    west = jnp.reshape(rho8u_part[-1, :, :], (1, u_size_y, u_size_z))
    east = jnp.reshape(rho8u_part[0, :, :], (1, u_size_y, u_size_z))
    rho8u = jnp.concatenate((west, rho8u_part, east), axis=0)  # periodic boundary
    south = jnp.reshape(rho8v_part[:, -1, :], (v_size_x, 1, v_size_z))
    north = jnp.reshape(rho8v_part[:, 0, :], (v_size_x, 1, v_size_z))
    rho8v = jnp.concatenate((south, rho8v_part, north), axis=1)  # periodic boundary
    rho_u = rho8u * u
    rho_v = rho8v * v

    div_x = (rho_u[1:, :, :] - rho_u[0:-1, :, :]) / (x3d4u[1:, :, :] - x3d4u[0:-1, :, :])
    div_y = (rho_v[:, 1:, :] - rho_v[:, 0:-1, :]) / (y3d4v[:, 1:, :] - y3d4v[:, 0:-1, :])
    div_rho_u = div_x + div_y

    return div_rho_u


def vertical_flux_scalar(weps, rho0, w, scalar):
    """ Vertical scalar flux using 3rd-order WENO

    Original WENO3 from Jiang adn Shu, 1996, JCP
    """
    # If w >= 0
    b1up = (scalar[:, :, 0:-3] - scalar[:, :, 1:-2]) ** 2
    b2up = (scalar[:, :, 1:-2] - scalar[:, :, 2:-1]) ** 2
    # # Original WENO
    # w1up = (1.0 / 3.0) / (weps + b1up) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up) ** 2
    # Improved smoothness indicators (Borges et al., 2008, JCP)
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b2up)/(b1up+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b2up)/(b2up+weps))**2)
    weno3_upward = (w1up * ((-1.0 / 2.0) * scalar[:, :, 0:-3] + (3.0 / 2.0) * scalar[:, :, 1:-2]) +
                    w2up * ((1.0 / 2.0) * scalar[:, :, 1:-2] + (1.0 / 2.0) * scalar[:, :, 2:-1])
                    ) / (w1up + w2up)
    weno3_upward = weno3_upward.at[:, :, 0].set(0.5 * (scalar[:, :, 1] + scalar[:, :, 2]))
    # reset reconstruction at the second level (w point), assuming ngz=1
    # If w < 0
    b1dn = (scalar[:, :, 3:] - scalar[:, :, 2:-1]) ** 2
    b2dn = (scalar[:, :, 2:-1] - scalar[:, :, 1:-2]) ** 2
    # # Original WENO
    # w1dn = (1.0 / 3.0) / (weps + b1dn) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn) ** 2
    # Improved smoothness indicators
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b2dn)/(b1dn+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b2dn)/(b2dn+weps))**2)
    weno3_downward = (w1dn * ((-1.0 / 2.0) * scalar[:, :, 3:] + (3.0 / 2.0) * scalar[:, :, 2:-1]) +
                      w2dn * ((1.0 / 2.0) * scalar[:, :, 2:-1] + (1.0 / 2.0) * scalar[:, :, 1:-2])
                      ) / (w1dn + w2dn)
    weno3_downward = weno3_downward.at[:, :, -1].set(0.5 * (scalar[:, :, -2] + scalar[:, :, -3]))
    # reset reconstruction at the penultimate level (w point), assuming ngz=1

    rho0w = 0.5 * (rho0[:, :, 1:-2] + rho0[:, :, 2:-1]) * w[:, :, 2:-2]  # rho0*w at w points
    flux = jax.lax.select(rho0w >= 0.0, rho0w * weno3_upward, rho0w * weno3_downward)
    # Concatenate zero flux at bottom and top. It can be modified later by a boundary condition function
    zero4w = jnp.zeros((nl.nx, nl.ny, 1))
    vertical_flux = jnp.concatenate((zero4w, flux[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, :], zero4w), axis=2)
    # nz+1 layers, ghost points discarded

    return vertical_flux


def vertical_flux_scalar2(weps, rho0, w, scalar):
    """ Vertical scalar flux using 5th-order WENO

    Original WENO5 from Jiang adn Shu, 1996, JCP
    """
    # If w >= 0, upward
    b1up = ((13.0 / 12.0) * (scalar[:,:,0:-5] - 2.0*scalar[:,:,1:-4] + scalar[:,:,2:-3]) ** 2
           + 0.25 * (scalar[:,:,0:-5] - 4.0*scalar[:,:,1:-4] + 3.0*scalar[:,:,2:-3]) ** 2)
    b2up = ((13.0 / 12.0) * (scalar[:,:,1:-4] - 2.0*scalar[:,:,2:-3] + scalar[:,:,3:-2]) ** 2
           + 0.25 * (scalar[:,:,1:-4] - scalar[:,:,3:-2]) ** 2)
    b3up = ((13.0 / 12.0) * (scalar[:,:,2:-3] - 2.0*scalar[:,:,3:-2] + scalar[:,:,4:-1]) ** 2
           + 0.25 * (3.0 * scalar[:,:,2:-3] - 4.0 * scalar[:,:,3:-2] + scalar[:,:,4:-1]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1up = 0.1 / (weps + b1up) ** 2
    # w2up = 0.6 / (weps + b2up) ** 2
    # w3up = 0.3 / (weps + b3up) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP)
    w1up = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b1up+weps))**2)       
    w2up = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b2up+weps))**2)
    w3up = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b3up+weps))**2)
    weno5_upward = ((w1up * (
            (2.0/6.0) * scalar[:,:,0:-5] + (-7.0/6.0)*scalar[:,:,1:-4] + (11.0/6.0) * scalar[:,:,2:-3]) +
                w2up * ((-1.0/6.0) * scalar[:,:,1:-4] + (5.0/6.0) * scalar[:,:,2:-3] + (2.0/6.0) * scalar[:,:,3:-2]) +
                w3up * ((2.0/6.0) * scalar[:,:,2:-3] + (5.0/6.0) * scalar[:,:,3:-2] + (-1.0/6.0) * scalar[:,:,4:-1])) /
               (w1up + w2up + w3up))
    # use WENO3 and centered difference for near-boundary points
    b1up_lower = (scalar[:, :, 1:2] - scalar[:, :, 2:3]) ** 2
    b2up_lower = (scalar[:, :, 2:3] - scalar[:, :, 3:4]) ** 2
    # # Original WENO
    # w1up = (1.0 / 3.0) / (weps + b1up_lower) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up_lower) ** 2
    # Improved smoothness indicators (Borges et al., 2008, JCP)
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_lower-b2up_lower)/(b1up_lower+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_lower-b2up_lower)/(b2up_lower+weps))**2)
    weno3_upward = (w1up * ((-1.0 / 2.0) * scalar[:, :, 1:2] + (3.0 / 2.0) * scalar[:, :, 2:3]) +
                    w2up * ((1.0 / 2.0) * scalar[:, :, 2:3] + (1.0 / 2.0) * scalar[:, :, 3:4])
                    ) / (w1up + w2up)
    weno5_upward = weno5_upward.at[:, :, 0:1].set(weno3_upward)
    scalar_l1 = 0.5 * (scalar[:, :, 1:2] + scalar[:, :, 2:3])
    b1up_upper = (scalar[:, :, -4:-3] - scalar[:, :, -3:-2]) ** 2
    b2up_upper = (scalar[:, :, -3:-2] - scalar[:, :, -2:-1]) ** 2
    # # Original WENO
    # w1up = (1.0 / 3.0) / (weps + b1up_upper) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up_upper) ** 2
    # Improved smoothness indicators (Borges et al., 2008, JCP)
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_upper-b2up_upper)/(b1up_upper+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_upper-b2up_upper)/(b2up_upper+weps))**2)
    scalar_u1 = (w1up * ((-1.0 / 2.0) * scalar[:, :, -4:-3] + (3.0 / 2.0) * scalar[:, :, -3:-2]) +
               w2up * ((1.0 / 2.0) * scalar[:, :, -3:-2] + (1.0 / 2.0) * scalar[:, :, -2:-1])
               ) / (w1up + w2up)
    zero4w = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, 1))
    scalar_upward = jnp.concatenate((zero4w,
                                     scalar_l1,
                                     weno5_upward,
                                     scalar_u1,
                                     zero4w), axis=2)
    
    # If w < 0, downward
    b1dn = ((13.0 / 12.0) * (scalar[:,:,5:] - 2.0 * scalar[:,:,4:-1] + scalar[:,:,3:-2]) ** 2
           + 0.25 * (scalar[:,:,5:] - 4.0 * scalar[:,:,4:-1] + 3.0 * scalar[:,:,3:-2]) ** 2)
    b2dn = ((13.0 / 12.0) * (scalar[:,:,4:-1] - 2.0 * scalar[:,:,3:-2] + scalar[:,:,2:-3]) ** 2
           + 0.25 * (scalar[:,:,4:-1] - scalar[:,:,2:-3]) ** 2)
    b3dn = ((13.0 / 12.0) * (scalar[:,:,3:-2] - 2.0 * scalar[:,:,2:-3] + scalar[:,:,1:-4]) ** 2
           + 0.25 * (3.0 * scalar[:,:,3:-2] - 4.0 * scalar[:,:,2:-3] + scalar[:,:,1:-4]) ** 2)
    # w1dn = 0.1 / (weps + b1dn) ** 2
    # w2dn = 0.6 / (weps + b2dn) ** 2
    # w3dn = 0.3 / (weps + b3dn) ** 2
    w1dn = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b1dn+weps))**2)       
    w2dn = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b2dn+weps))**2)
    w3dn = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b3dn+weps))**2)
    weno5_downward = ((w1dn * (
            (2.0/6.0) * scalar[:,:,5:] + (-7.0/6.0) * scalar[:,:,4:-1] + (11.0/6.0)*scalar[:,:,3:-2]) +
                w2dn * ((-1.0/6.0) * scalar[:,:,4:-1] + (5.0/6.0) * scalar[:,:,3:-2] + (2.0/6.0) * scalar[:,:,2:-3]) +
                w3dn * ((2.0/6.0) * scalar[:,:,3:-2] + (5.0/6.0) * scalar[:,:,2:-3] + (-1.0/6.0) * scalar[:,:,1:-4])) /
               (w1dn + w2dn + w3dn))
    # use WENO3 and centered difference for near-boundary points
    b1dn_upper = (scalar[:, :, -2:-1] - scalar[:, :, -3:-2]) ** 2
    b2dn_upper = (scalar[:, :, -3:-2] - scalar[:, :, -4:-3]) ** 2
    # # Original WENO
    # w1dn = (1.0 / 3.0) / (weps + b1dn_upper) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn_upper) ** 2
    # Improved smoothness indicators
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_upper-b2dn_upper)/(b1dn_upper+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_upper-b2dn_upper)/(b2dn_upper+weps))**2)
    weno3_downward = (w1dn * ((-1.0 / 2.0) * scalar[:, :, -2:-1] + (3.0 / 2.0) * scalar[:, :, -3:-2]) +
                      w2dn * ((1.0 / 2.0) * scalar[:, :, -3:-2] + (1.0 / 2.0) * scalar[:, :, -4:-3])
                      ) / (w1dn + w2dn)
    weno5_downward = weno5_downward.at[:, :, -1:].set(weno3_downward)
    scalar_u1 = 0.5 * (scalar[:, :, -2:-1] + scalar[:, :, -3:-2])
    b1dn_lower = (scalar[:, :, 3:4] - scalar[:, :, 2:3]) ** 2
    b2dn_lower = (scalar[:, :, 2:3] - scalar[:, :, 1:2]) ** 2
    # # Original WENO
    # w1dn = (1.0 / 3.0) / (weps + b1dn_lower) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn_lower) ** 2
    # Improved smoothness indicators
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_lower-b2dn_lower)/(b1dn_lower+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_lower-b2dn_lower)/(b2dn_lower+weps))**2)
    scalar_l1 = (w1dn * ((-1.0 / 2.0) * scalar[:, :, 3:4] + (3.0 / 2.0) * scalar[:, :, 2:3]) +
                      w2dn * ((1.0 / 2.0) * scalar[:, :, 2:3] + (1.0 / 2.0) * scalar[:, :, 1:2])
                      ) / (w1dn + w2dn)
    scalar_downward = jnp.concatenate((zero4w,
                                       scalar_l1,
                                       weno5_downward,
                                       scalar_u1,
                                       zero4w), axis=2)

    rho0w = 0.5 * (rho0[:, :, 0:-1] + rho0[:, :, 1:]) * w[:, :, 1:-1]  # rho0*w at w points
    flux = jax.lax.select(rho0w >= 0.0, rho0w * scalar_upward, rho0w * scalar_downward)
    # We have zero flux at bottom and top. It can be modified later by a boundary condition function
    # nz+1 layers, ghost points discarded
    vertical_flux = flux[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, :]
    return vertical_flux


def horizontal_flux_scalar(weps, rho0, u, v, scalar):
    """ Horizontal scalar fluxes using 5th-order WENO

    Original WENO from Jiang adn Shu, 1996, JCP
    """
    # Compute x-direction flux first
    # If u >= 0, westerly
    b1w = ((13.0 / 12.0) * (scalar[0:-5, :, :] - 2.0 * scalar[1:-4, :, :] + scalar[2:-3, :, :]) ** 2
           + 0.25 * (scalar[0:-5, :, :] - 4.0 * scalar[1:-4, :, :] + 3.0 * scalar[2:-3, :, :]) ** 2)
    b2w = ((13.0 / 12.0) * (scalar[1:-4, :, :] - 2.0 * scalar[2:-3, :, :] + scalar[3:-2, :, :]) ** 2
           + 0.25 * (scalar[1:-4, :, :] - scalar[3:-2, :, :]) ** 2)
    b3w = ((13.0 / 12.0) * (scalar[2:-3, :, :] - 2.0 * scalar[3:-2, :, :] + scalar[4:-1, :, :]) ** 2
           + 0.25 * (3.0 * scalar[2:-3, :, :] - 4.0 * scalar[3:-2, :, :] + scalar[4:-1, :, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1w = 0.1 / (weps + b1w) ** 2
    # w2w = 0.6 / (weps + b2w) ** 2
    # w3w = 0.3 / (weps + b3w) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP)
    w1w = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b1w+weps))**2)       
    w2w = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b2w+weps))**2)
    w3w = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b3w+weps))**2)    
    weno5_w = ((w1w * (
            (2.0 / 6.0) * scalar[0:-5, :, :] + (-7.0 / 6.0) * scalar[1:-4, :, :] + (11.0 / 6.0) * scalar[2:-3, :, :]) +
                w2w * ((-1.0 / 6.0) * scalar[1:-4, :, :] + (5.0 / 6.0) * scalar[2:-3, :, :] + (2.0 / 6.0) * scalar[3:-2, :, :]) +
                w3w * ((2.0 / 6.0) * scalar[2:-3, :, :] + (5.0 / 6.0) * scalar[3:-2, :, :] + (-1.0 / 6.0) * scalar[4:-1, :, :])) /
               (w1w + w2w + w3w))
    # If u < 0, easterly
    b1e = ((13.0 / 12.0) * (scalar[5:, :, :] - 2.0 * scalar[4:-1, :, :] + scalar[3:-2, :, :]) ** 2
           + 0.25 * (scalar[5:, :, :] - 4.0 * scalar[4:-1, :, :] + 3.0 * scalar[3:-2, :, :]) ** 2)
    b2e = ((13.0 / 12.0) * (scalar[4:-1, :, :] - 2.0 * scalar[3:-2, :, :] + scalar[2:-3, :, :]) ** 2
           + 0.25 * (scalar[4:-1, :, :] - scalar[2:-3, :, :]) ** 2)
    b3e = ((13.0 / 12.0) * (scalar[3:-2, :, :] - 2.0 * scalar[2:-3, :, :] + scalar[1:-4, :, :]) ** 2
           + 0.25 * (3.0 * scalar[3:-2, :, :] - 4.0 * scalar[2:-3, :, :] + scalar[1:-4, :, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1e = 0.1 / (weps + b1e) ** 2
    # w2e = 0.6 / (weps + b2e) ** 2
    # w3e = 0.3 / (weps + b3e) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP)
    w1e = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b1e+weps))**2)       
    w2e = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b2e+weps))**2)
    w3e = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b3e+weps))**2) 
    weno5_e = ((w1e * (
            (2.0 / 6.0) * scalar[5:, :, :] + (-7.0 / 6.0) * scalar[4:-1, :, :] + (11.0 / 6.0) * scalar[3:-2, :, :]) +
                w2e * ((-1.0 / 6.0) * scalar[4:-1, :, :] + (5.0 / 6.0) * scalar[3:-2, :, :] + (2.0 / 6.0) * scalar[2:-3, :, :]) +
                w3e * ((2.0 / 6.0) * scalar[3:-2, :, :] + (5.0 / 6.0) * scalar[2:-3, :, :] + (-1.0 / 6.0) * scalar[1:-4, :, :])) /
               (w1e + w2e + w3e))

    rho0u = 0.5 * (rho0[2:-3, :, :] + rho0[3:-2, :, :]) * u[3:-3, :, :]  # rho0*u at u points
    flux_x = jax.lax.select(rho0u >= 0.0, rho0u * weno5_w, rho0u * weno5_e)
    horizontal_flux_x = flux_x[:, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]  # discard ghost points

    # Compute y-direction flux
    # If v >=0, southerly
    b1s = ((13.0 / 12.0) * (scalar[:, 0:-5, :] - 2.0 * scalar[:, 1:-4, :] + scalar[:, 2:-3, :]) ** 2
           + 0.25 * (scalar[:, 0:-5, :] - 4.0 * scalar[:, 1:-4, :] + 3.0 * scalar[:, 2:-3, :]) ** 2)
    b2s = ((13.0 / 12.0) * (scalar[:, 1:-4, :] - 2.0 * scalar[:, 2:-3, :] + scalar[:, 3:-2, :]) ** 2
           + 0.25 * (scalar[:, 1:-4, :] - scalar[:, 3:-2, :]) ** 2)
    b3s = ((13.0 / 12.0) * (scalar[:, 2:-3, :] - 2.0 * scalar[:, 3:-2, :] + scalar[:, 4:-1, :]) ** 2
           + 0.25 * (3.0 * scalar[:, 2:-3, :] - 4.0 * scalar[:, 3:-2, :] + scalar[:, 4:-1, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1s = 0.1 / (weps + b1s) ** 2
    # w2s = 0.6 / (weps + b2s) ** 2
    # w3s = 0.3 / (weps + b3s) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP)
    w1s = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b1s+weps))**2)
    w2s = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b2s+weps))**2)
    w3s = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b3s+weps))**2)
    weno5_s = ((w1s * (
            (2.0 / 6.0) * scalar[:, 0:-5, :] + (-7.0 / 6.0) * scalar[:, 1:-4, :] + (11.0 / 6.0) * scalar[:, 2:-3, :]) +
                w2s * ((-1.0 / 6.0) * scalar[:, 1:-4, :] + (5.0 / 6.0) * scalar[:, 2:-3, :] + (2.0 / 6.0) * scalar[:, 3:-2, :]) +
                w3s * ((2.0 / 6.0) * scalar[:, 2:-3, :] + (5.0 / 6.0) * scalar[:, 3:-2, :] + (-1.0 / 6.0) * scalar[:, 4:-1, :])) /
               (w1s + w2s + w3s))
    # If v < 0, northerly
    b1n = ((13.0 / 12.0) * (scalar[:, 5:, :] - 2.0 * scalar[:, 4:-1, :] + scalar[:, 3:-2, :]) ** 2
           + 0.25 * (scalar[:, 5:, :] - 4.0 * scalar[:, 4:-1, :] + 3.0 * scalar[:, 3:-2, :]) ** 2)
    b2n = ((13.0 / 12.0) * (scalar[:, 4:-1, :] - 2.0 * scalar[:, 3:-2, :] + scalar[:, 2:-3, :]) ** 2
           + 0.25 * (scalar[:, 4:-1, :] - scalar[:, 2:-3, :]) ** 2)
    b3n = ((13.0 / 12.0) * (scalar[:, 3:-2, :] - 2.0 * scalar[:, 2:-3, :] + scalar[:, 1:-4, :]) ** 2
           + 0.25 * (3.0 * scalar[:, 3:-2, :] - 4.0 * scalar[:, 2:-3, :] + scalar[:, 1:-4, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1n = 0.1 / (weps + b1n) ** 2
    # w2n = 0.6 / (weps + b2n) ** 2
    # w3n = 0.3 / (weps + b3n) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP)
    w1n = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b1n+weps))**2)
    w2n = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b2n+weps))**2)
    w3n = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b3n+weps))**2)
    weno5_n = ((w1n * (
            (2.0 / 6.0) * scalar[:, 5:, :] + (-7.0 / 6.0) * scalar[:, 4:-1, :] + (11.0 / 6.0) * scalar[:, 3:-2, :]) +
                w2n * ((-1.0 / 6.0) * scalar[:, 4:-1, :] + (5.0 / 6.0) * scalar[:, 3:-2, :] + (2.0 / 6.0) * scalar[:, 2:-3, :]) +
                w3n * ((2.0 / 6.0) * scalar[:, 3:-2, :] + (5.0 / 6.0) * scalar[:, 2:-3, :] + (-1.0 / 6.0) * scalar[:, 1:-4, :])) /
               (w1n + w2n + w3n))

    rho0v = 0.5 * (rho0[:, 2:-3, :] + rho0[:, 3:-2, :]) * v[:, 3:-3, :]  # rho0*v at v points
    flux_y = jax.lax.select(rho0v >= 0.0, rho0v * weno5_s, rho0v * weno5_n)
    horizontal_flux_y = flux_y[nl.ngx:-nl.ngx, :, nl.ngz:-nl.ngz]  # discard ghost points

    return horizontal_flux_x, horizontal_flux_y


def vertical_flux_u(weps, rho0, w, u):
    """ Vertical u-momentum flux using 3rd-order WENO

    Original WENO3 from Jiang adn Shu, 1996, JCP
    """
    # If w >= 0
    b1up = (u[:, :, 0:-3] - u[:, :, 1:-2]) ** 2
    b2up = (u[:, :, 1:-2] - u[:, :, 2:-1]) ** 2
    # w1up = (1.0 / 3.0) / (weps + b1up) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up) ** 2
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b2up)/(b1up+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b2up)/(b2up+weps))**2)
    weno3_upward = (w1up * ((-1.0 / 2.0) * u[:, :, 0:-3] + (3.0 / 2.0) * u[:, :, 1:-2]) +
                    w2up * ((1.0 / 2.0) * u[:, :, 1:-2] + (1.0 / 2.0) * u[:, :, 2:-1])
                    ) / (w1up + w2up)
    weno3_upward = weno3_upward.at[:, :, 0].set(0.5 * (u[:, :, 1] + u[:, :, 2]))  # reset level 1
    # If w < 0
    b1dn = (u[:, :, 3:] - u[:, :, 2:-1]) ** 2
    b2dn = (u[:, :, 2:-1] - u[:, :, 1:-2]) ** 2
    # w1dn = (1.0 / 3.0) / (weps + b1dn) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn) ** 2
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b2dn)/(b1dn+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b2dn)/(b2dn+weps))**2)
    weno3_downward = (w1dn * ((-1.0 / 2.0) * u[:, :, 3:] + (3.0 / 2.0) * u[:, :, 2:-1]) +
                      w2dn * ((1.0 / 2.0) * u[:, :, 2:-1] + (1.0 / 2.0) * u[:, :, 1:-2])
                      ) / (w1dn + w2dn)
    weno3_downward = weno3_downward.at[:, :, -1].set(0.5 * (u[:, :, -2] + u[:, :, -3]))
    # reset reconstruction at the penultimate level (w point), assuming ngz=1
    
    rho0w = 0.5 * (rho0[:, :, 1:-2] + rho0[:, :, 2:-1]) * w[:, :, 2:-2]
    rho0w8u = 0.5 * (rho0w[0:-1, :, :] + rho0w[1:, :, :])
    # rho0*w above u point at the w level
    flux = jax.lax.select(rho0w8u >= 0.0, rho0w8u * weno3_upward[1:-1, :, :],
                          rho0w8u * weno3_downward[1:-1, :, :])
    # Concatenate zero flux at bottom and top. It can be modified later by a boundary condition function
    zero4w = jnp.zeros((nl.nx + 1, nl.ny, 1))
    vertical_flux = jnp.concatenate((zero4w, flux[nl.ngx - 1:-(nl.ngx - 1), nl.ngy:-nl.ngy, :], zero4w), axis=2)
    # nz+1 layers, ghost points discarded

    return vertical_flux


def vertical_flux_u2(weps, rho0, w, u):
    """ Vertical u-momentum flux using 5th-order WENO

    Original WENO5 from Jiang adn Shu, 1996, JCP
    """
    # If w >= 0, upward
    b1up = ((13.0 / 12.0) * (u[:,:,0:-5] - 2.0*u[:,:,1:-4] + u[:,:,2:-3]) ** 2
           + 0.25 * (u[:,:,0:-5] - 4.0*u[:,:,1:-4] + 3.0*u[:,:,2:-3]) ** 2)
    b2up = ((13.0 / 12.0) * (u[:,:,1:-4] - 2.0*u[:,:,2:-3] + u[:,:,3:-2]) ** 2
           + 0.25 * (u[:,:,1:-4] - u[:,:,3:-2]) ** 2)
    b3up = ((13.0 / 12.0) * (u[:,:,2:-3] - 2.0*u[:,:,3:-2] + u[:,:,4:-1]) ** 2
           + 0.25 * (3.0 * u[:,:,2:-3] - 4.0 * u[:,:,3:-2] + u[:,:,4:-1]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1up = 0.1 / (weps + b1up) ** 2
    # w2up = 0.6 / (weps + b2up) ** 2
    # w3up = 0.3 / (weps + b3up) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP)
    w1up = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b1up+weps))**2)       
    w2up = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b2up+weps))**2)
    w3up = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b3up+weps))**2)
    weno5_upward = ((w1up * (
            (2.0/6.0) * u[:,:,0:-5] + (-7.0/6.0)*u[:,:,1:-4] + (11.0/6.0) * u[:,:,2:-3]) +
                w2up * ((-1.0/6.0) * u[:,:,1:-4] + (5.0/6.0) * u[:,:,2:-3] + (2.0/6.0) * u[:,:,3:-2]) +
                w3up * ((2.0/6.0) * u[:,:,2:-3] + (5.0/6.0) * u[:,:,3:-2] + (-1.0/6.0) * u[:,:,4:-1])) /
               (w1up + w2up + w3up))
    # use WENO3 and centered difference for near-boundary points
    b1up_lower = (u[:, :, 1:2] - u[:, :, 2:3]) ** 2
    b2up_lower = (u[:, :, 2:3] - u[:, :, 3:4]) ** 2
    # # Original WENO
    # w1up = (1.0 / 3.0) / (weps + b1up_lower) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up_lower) ** 2
    # Improved smoothness indicators (Borges et al., 2008, JCP)
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_lower-b2up_lower)/(b1up_lower+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_lower-b2up_lower)/(b2up_lower+weps))**2)
    weno3_upward = (w1up * ((-1.0 / 2.0) * u[:, :, 1:2] + (3.0 / 2.0) * u[:, :, 2:3]) +
                    w2up * ((1.0 / 2.0) * u[:, :, 2:3] + (1.0 / 2.0) * u[:, :, 3:4])
                    ) / (w1up + w2up)
    weno5_upward = weno5_upward.at[:, :, 0:1].set(weno3_upward)
    u_l1 = 0.5 * (u[:, :, 1:2] + u[:, :, 2:3])
    b1up_upper = (u[:, :, -4:-3] - u[:, :, -3:-2]) ** 2
    b2up_upper = (u[:, :, -3:-2] - u[:, :, -2:-1]) ** 2
    # # Original WENO
    # w1up = (1.0 / 3.0) / (weps + b1up_upper) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up_upper) ** 2
    # Improved smoothness indicators (Borges et al., 2008, JCP)
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_upper-b2up_upper)/(b1up_upper+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_upper-b2up_upper)/(b2up_upper+weps))**2)
    u_u1 = (w1up * ((-1.0 / 2.0) * u[:, :, -4:-3] + (3.0 / 2.0) * u[:, :, -3:-2]) +
               w2up * ((1.0 / 2.0) * u[:, :, -3:-2] + (1.0 / 2.0) * u[:, :, -2:-1])
               ) / (w1up + w2up)
    zero4w = jnp.zeros((nl.nx+1+2*nl.ngx, nl.ny+2*nl.ngy, 1))
    u_upward = jnp.concatenate((zero4w,
                                u_l1,
                                weno5_upward,
                                u_u1,
                                zero4w), axis=2)
    
    # If w < 0, downward
    b1dn = ((13.0 / 12.0) * (u[:,:,5:] - 2.0 * u[:,:,4:-1] + u[:,:,3:-2]) ** 2
           + 0.25 * (u[:,:,5:] - 4.0 * u[:,:,4:-1] + 3.0 * u[:,:,3:-2]) ** 2)
    b2dn = ((13.0 / 12.0) * (u[:,:,4:-1] - 2.0 * u[:,:,3:-2] + u[:,:,2:-3]) ** 2
           + 0.25 * (u[:,:,4:-1] - u[:,:,2:-3]) ** 2)
    b3dn = ((13.0 / 12.0) * (u[:,:,3:-2] - 2.0 * u[:,:,2:-3] + u[:,:,1:-4]) ** 2
           + 0.25 * (3.0 * u[:,:,3:-2] - 4.0 * u[:,:,2:-3] + u[:,:,1:-4]) ** 2)
    # w1dn = 0.1 / (weps + b1dn) ** 2
    # w2dn = 0.6 / (weps + b2dn) ** 2
    # w3dn = 0.3 / (weps + b3dn) ** 2
    w1dn = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b1dn+weps))**2)       
    w2dn = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b2dn+weps))**2)
    w3dn = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b3dn+weps))**2)
    weno5_downward = ((w1dn * (
            (2.0/6.0) * u[:,:,5:] + (-7.0/6.0) * u[:,:,4:-1] + (11.0/6.0)*u[:,:,3:-2]) +
                w2dn * ((-1.0/6.0) * u[:,:,4:-1] + (5.0/6.0) * u[:,:,3:-2] + (2.0/6.0) * u[:,:,2:-3]) +
                w3dn * ((2.0/6.0) * u[:,:,3:-2] + (5.0/6.0) * u[:,:,2:-3] + (-1.0/6.0) * u[:,:,1:-4])) /
               (w1dn + w2dn + w3dn))
    # use WENO3 and centered difference for near-boundary points
    b1dn_upper = (u[:, :, -2:-1] - u[:, :, -3:-2]) ** 2
    b2dn_upper = (u[:, :, -3:-2] - u[:, :, -4:-3]) ** 2
    # # Original WENO
    # w1dn = (1.0 / 3.0) / (weps + b1dn_upper) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn_upper) ** 2
    # Improved smoothness indicators
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_upper-b2dn_upper)/(b1dn_upper+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_upper-b2dn_upper)/(b2dn_upper+weps))**2)
    weno3_downward = (w1dn * ((-1.0 / 2.0) * u[:, :, -2:-1] + (3.0 / 2.0) * u[:, :, -3:-2]) +
                      w2dn * ((1.0 / 2.0) * u[:, :, -3:-2] + (1.0 / 2.0) * u[:, :, -4:-3])
                      ) / (w1dn + w2dn)
    weno5_downward = weno5_downward.at[:, :, -1:].set(weno3_downward)
    u_u1 = 0.5 * (u[:, :, -2:-1] + u[:, :, -3:-2])
    b1dn_lower = (u[:, :, 3:4] - u[:, :, 2:3]) ** 2
    b2dn_lower = (u[:, :, 2:3] - u[:, :, 1:2]) ** 2
    # # Original WENO
    # w1dn = (1.0 / 3.0) / (weps + b1dn_lower) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn_lower) ** 2
    # Improved smoothness indicators
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_lower-b2dn_lower)/(b1dn_lower+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_lower-b2dn_lower)/(b2dn_lower+weps))**2)
    u_l1 = (w1dn * ((-1.0 / 2.0) * u[:, :, 3:4] + (3.0 / 2.0) * u[:, :, 2:3]) +
                      w2dn * ((1.0 / 2.0) * u[:, :, 2:3] + (1.0 / 2.0) * u[:, :, 1:2])
                      ) / (w1dn + w2dn)
    u_downward = jnp.concatenate((zero4w,
                                  u_l1,
                                  weno5_downward,
                                  u_u1,
                                  zero4w), axis=2)
    
    rho0w = 0.5 * (rho0[:, :, 0:-1] + rho0[:, :, 1:]) * w[:, :, 1:-1]
    rho0w8u = 0.5 * (rho0w[0:-1, :, :] + rho0w[1:, :, :])
    # rho0*w above u point at the w level
    flux = jax.lax.select(rho0w8u >= 0.0, rho0w8u * u_upward[1:-1, :, :],
                          rho0w8u * u_downward[1:-1, :, :])
    # We have zero flux at bottom and top. It can be modified later by a boundary condition function
    vertical_flux = flux[nl.ngx - 1:-(nl.ngx - 1), nl.ngy:-nl.ngy, :]
    # nz+1 layers, ghost points discarded
    return vertical_flux


def horizontal_flux_u(weps, rho0, u, v):
    """ Horizontal u-momentum fluxes using 5th-order WENO

    Original WENO from Jiang adn Shu, 1996, JCP
    """
    # Compute x-direction flux first
    # If u >= 0, westerly
    b1w = ((13.0 / 12.0) * (u[0:-5, :, :] - 2.0 * u[1:-4, :, :] + u[2:-3, :, :]) ** 2
           + 0.25 * (u[0:-5, :, :] - 4.0 * u[1:-4, :, :] + 3.0 * u[2:-3, :, :]) ** 2)
    b2w = ((13.0 / 12.0) * (u[1:-4, :, :] - 2.0 * u[2:-3, :, :] + u[3:-2, :, :]) ** 2
           + 0.25 * (u[1:-4, :, :] - u[3:-2, :, :]) ** 2)
    b3w = ((13.0 / 12.0) * (u[2:-3, :, :] - 2.0 * u[3:-2, :, :] + u[4:-1, :, :]) ** 2
           + 0.25 * (3.0 * u[2:-3, :, :] - 4.0 * u[3:-2, :, :] + u[4:-1, :, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1w = 0.1 / (weps + b1w) ** 2
    # w2w = 0.6 / (weps + b2w) ** 2
    # w3w = 0.3 / (weps + b3w) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP) 
    w1w = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b1w+weps))**2)       
    w2w = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b2w+weps))**2)
    w3w = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b3w+weps))**2)
    weno5_w = ((w1w * ((2.0 / 6.0) * u[0:-5, :, :] + (-7.0 / 6.0) * u[1:-4, :, :] + (11.0 / 6.0) * u[2:-3, :, :]) +
                w2w * ((-1.0 / 6.0) * u[1:-4, :, :] + (5.0 / 6.0) * u[2:-3, :, :] + (2.0 / 6.0) * u[3:-2, :, :]) +
                w3w * ((2.0 / 6.0) * u[2:-3, :, :] + (5.0 / 6.0) * u[3:-2, :, :] + (-1.0 / 6.0) * u[4:-1, :, :])) /
               (w1w + w2w + w3w))
    # If u < 0, easterly
    b1e = ((13.0 / 12.0) * (u[5:, :, :] - 2.0 * u[4:-1, :, :] + u[3:-2, :, :]) ** 2
           + 0.25 * (u[5:, :, :] - 4.0 * u[4:-1, :, :] + 3.0 * u[3:-2, :, :]) ** 2)
    b2e = ((13.0 / 12.0) * (u[4:-1, :, :] - 2.0 * u[3:-2, :, :] + u[2:-3, :, :]) ** 2
           + 0.25 * (u[4:-1, :, :] - u[2:-3, :, :]) ** 2)
    b3e = ((13.0 / 12.0) * (u[3:-2, :, :] - 2.0 * u[2:-3, :, :] + u[1:-4, :, :]) ** 2
           + 0.25 * (3.0 * u[3:-2, :, :] - 4.0 * u[2:-3, :, :] + u[1:-4, :, :]) ** 2)
    # w1e = 0.1 / (weps + b1e) ** 2
    # w2e = 0.6 / (weps + b2e) ** 2
    # w3e = 0.3 / (weps + b3e) ** 2
    w1e = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b1e+weps))**2)       
    w2e = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b2e+weps))**2)
    w3e = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b3e+weps))**2)
    weno5_e = ((w1e * ((2.0 / 6.0) * u[5:, :, :] + (-7.0 / 6.0) * u[4:-1, :, :] + (11.0 / 6.0) * u[3:-2, :, :]) +
                w2e * ((-1.0 / 6.0) * u[4:-1, :, :] + (5.0 / 6.0) * u[3:-2, :, :] + (2.0 / 6.0) * u[2:-3, :, :]) +
                w3e * ((2.0 / 6.0) * u[3:-2, :, :] + (5.0 / 6.0) * u[2:-3, :, :] + (-1.0 / 6.0) * u[1:-4, :, :])) /
               (w1e + w2e + w3e))

    rho0u = 0.5 * (u[2:-3, :, :] + u[3:-2, :, :]) * rho0[2:-2, :, :]
    # rho0*u at scalar points
    flux_x = jax.lax.select(rho0u >= 0.0, rho0u * weno5_w, rho0u * weno5_e)
    horizontal_flux_x = flux_x[:, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]  # discard ghost points

    # Compute y-direction flux
    # If v >=0, southerly
    b1s = ((13.0 / 12.0) * (u[:, 0:-5, :] - 2.0 * u[:, 1:-4, :] + u[:, 2:-3, :]) ** 2
           + 0.25 * (u[:, 0:-5, :] - 4.0 * u[:, 1:-4, :] + 3.0 * u[:, 2:-3, :]) ** 2)
    b2s = ((13.0 / 12.0) * (u[:, 1:-4, :] - 2.0 * u[:, 2:-3, :] + u[:, 3:-2, :]) ** 2
           + 0.25 * (u[:, 1:-4, :] - u[:, 3:-2, :]) ** 2)
    b3s = ((13.0 / 12.0) * (u[:, 2:-3, :] - 2.0 * u[:, 3:-2, :] + u[:, 4:-1, :]) ** 2
           + 0.25 * (3.0 * u[:, 2:-3, :] - 4.0 * u[:, 3:-2, :] + u[:, 4:-1, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1s = 0.1 / (weps + b1s) ** 2
    # w2s = 0.6 / (weps + b2s) ** 2
    # w3s = 0.3 / (weps + b3s) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP) 
    w1s = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b1s+weps))**2)       
    w2s = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b2s+weps))**2)
    w3s = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b3s+weps))**2)
    weno5_s = ((w1s * ((2.0 / 6.0) * u[:, 0:-5, :] + (-7.0 / 6.0) * u[:, 1:-4, :] + (11.0 / 6.0) * u[:, 2:-3, :]) +
                w2s * ((-1.0 / 6.0) * u[:, 1:-4, :] + (5.0 / 6.0) * u[:, 2:-3, :] + (2.0 / 6.0) * u[:, 3:-2, :]) +
                w3s * ((2.0 / 6.0) * u[:, 2:-3, :] + (5.0 / 6.0) * u[:, 3:-2, :] + (-1.0 / 6.0) * u[:, 4:-1, :])) /
               (w1s + w2s + w3s))
    # If v < 0, northerly
    b1n = ((13.0 / 12.0) * (u[:, 5:, :] - 2.0 * u[:, 4:-1, :] + u[:, 3:-2, :]) ** 2
           + 0.25 * (u[:, 5:, :] - 4.0 * u[:, 4:-1, :] + 3.0 * u[:, 3:-2, :]) ** 2)
    b2n = ((13.0 / 12.0) * (u[:, 4:-1, :] - 2.0 * u[:, 3:-2, :] + u[:, 2:-3, :]) ** 2
           + 0.25 * (u[:, 4:-1, :] - u[:, 2:-3, :]) ** 2)
    b3n = ((13.0 / 12.0) * (u[:, 3:-2, :] - 2.0 * u[:, 2:-3, :] + u[:, 1:-4, :]) ** 2
           + 0.25 * (3.0 * u[:, 3:-2, :] - 4.0 * u[:, 2:-3, :] + u[:, 1:-4, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1n = 0.1 / (weps + b1n) ** 2
    # w2n = 0.6 / (weps + b2n) ** 2
    # w3n = 0.3 / (weps + b3n) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP) 
    w1n = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b1n+weps))**2)       
    w2n = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b2n+weps))**2)
    w3n = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b3n+weps))**2)    
    weno5_n = ((w1n * ((2.0 / 6.0) * u[:, 5:, :] + (-7.0 / 6.0) * u[:, 4:-1, :] + (11.0 / 6.0) * u[:, 3:-2, :]) +
                w2n * ((-1.0 / 6.0) * u[:, 4:-1, :] + (5.0 / 6.0) * u[:, 3:-2, :] + (2.0 / 6.0) * u[:, 2:-3, :]) +
                w3n * ((2.0 / 6.0) * u[:, 3:-2, :] + (5.0 / 6.0) * u[:, 2:-3, :] + (-1.0 / 6.0) * u[:, 1:-4, :])) /
               (w1n + w2n + w3n))

    rho0v = 0.5 * (rho0[:, 2:-3, :] + rho0[:, 3:-2, :]) * v[:, 3:-3, :]  # rho0*v at v points
    rho0v8u = 0.5 * (rho0v[0:-1, :, :] + rho0v[1:, :, :])  # rho0*v at corner points, to the south of u points
    flux_y = jax.lax.select(rho0v8u >= 0.0, rho0v8u * weno5_s[1:-1, :, :], rho0v8u * weno5_n[1:-1, :, :])
    horizontal_flux_y = flux_y[nl.ngx - 1:-(nl.ngx - 1), :, nl.ngz:-nl.ngz]  # discard ghost points

    return horizontal_flux_x, horizontal_flux_y


def vertical_flux_v(weps, rho0, w, v):
    """ Vertical v-momentum flux using 3rd-order WENO

    Original WENO3 from Jiang adn Shu, 1996, JCP
    """
    # If w >= 0
    b1up = (v[:, :, 0:-3] - v[:, :, 1:-2]) ** 2
    b2up = (v[:, :, 1:-2] - v[:, :, 2:-1]) ** 2
    # w1up = (1.0 / 3.0) / (weps + b1up) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up) ** 2
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b2up)/(b1up+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b2up)/(b2up+weps))**2)
    weno3_upward = (w1up * ((-1.0 / 2.0) * v[:, :, 0:-3] + (3.0 / 2.0) * v[:, :, 1:-2]) +
                    w2up * ((1.0 / 2.0) * v[:, :, 1:-2] + (1.0 / 2.0) * v[:, :, 2:-1])
                    ) / (w1up + w2up)
    weno3_upward = weno3_upward.at[:, :, 0].set(0.5 * (v[:, :, 1] + v[:, :, 2]))  # reset level 1
    # If w < 0
    b1dn = (v[:, :, 3:] - v[:, :, 2:-1]) ** 2
    b2dn = (v[:, :, 2:-1] - v[:, :, 1:-2]) ** 2
    # w1dn = (1.0 / 3.0) / (weps + b1dn) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn) ** 2
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b2dn)/(b1dn+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b2dn)/(b2dn+weps))**2)
    weno3_downward = (w1dn * ((-1.0 / 2.0) * v[:, :, 3:] + (3.0 / 2.0) * v[:, :, 2:-1]) +
                      w2dn * ((1.0 / 2.0) * v[:, :, 2:-1] + (1.0 / 2.0) * v[:, :, 1:-2])
                      ) / (w1dn + w2dn)
    weno3_downward = weno3_downward.at[:, :, -1].set(0.5 * (v[:, :, -2] + v[:, :, -3]))
    # reset reconstruction at the penultimate level (w point), assuming ngz=1

    rho0w = 0.5 * (rho0[:, :, 1:-2] + rho0[:, :, 2:-1]) * w[:, :, 2:-2]
    rho0w8v = 0.5 * (rho0w[:, 0:-1, :] + rho0w[:, 1:, :])
    # rho0*w above v point at the w level
    flux = jax.lax.select(rho0w8v >= 0.0, rho0w8v * weno3_upward[:, 1:-1, :],
                          rho0w8v * weno3_downward[:, 1:-1, :])
    # Concatenate zero flux at bottom and top. It can be modified later by a boundary condition function
    zero4w = jnp.zeros((nl.nx, nl.ny + 1, 1))
    vertical_flux = jnp.concatenate((zero4w, flux[nl.ngx:-nl.ngx, nl.ngy - 1:-(nl.ngy - 1), :], zero4w), axis=2)
    # nz+1 layers, ghost points discarded

    return vertical_flux


def vertical_flux_v2(weps, rho0, w, v):
    """ Vertical v-momentum flux using 5th-order WENO

    Original WENO5 from Jiang adn Shu, 1996, JCP
    """
    # If w >= 0, upward
    b1up = ((13.0 / 12.0) * (v[:,:,0:-5] - 2.0*v[:,:,1:-4] + v[:,:,2:-3]) ** 2
           + 0.25 * (v[:,:,0:-5] - 4.0*v[:,:,1:-4] + 3.0*v[:,:,2:-3]) ** 2)
    b2up = ((13.0 / 12.0) * (v[:,:,1:-4] - 2.0*v[:,:,2:-3] + v[:,:,3:-2]) ** 2
           + 0.25 * (v[:,:,1:-4] - v[:,:,3:-2]) ** 2)
    b3up = ((13.0 / 12.0) * (v[:,:,2:-3] - 2.0*v[:,:,3:-2] + v[:,:,4:-1]) ** 2
           + 0.25 * (3.0 * v[:,:,2:-3] - 4.0 * v[:,:,3:-2] + v[:,:,4:-1]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1up = 0.1 / (weps + b1up) ** 2
    # w2up = 0.6 / (weps + b2up) ** 2
    # w3up = 0.3 / (weps + b3up) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP)
    w1up = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b1up+weps))**2)       
    w2up = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b2up+weps))**2)
    w3up = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b3up+weps))**2)
    weno5_upward = ((w1up * (
            (2.0/6.0) * v[:,:,0:-5] + (-7.0/6.0)*v[:,:,1:-4] + (11.0/6.0) * v[:,:,2:-3]) +
                w2up * ((-1.0/6.0) * v[:,:,1:-4] + (5.0/6.0) * v[:,:,2:-3] + (2.0/6.0) * v[:,:,3:-2]) +
                w3up * ((2.0/6.0) * v[:,:,2:-3] + (5.0/6.0) * v[:,:,3:-2] + (-1.0/6.0) * v[:,:,4:-1])) /
               (w1up + w2up + w3up))
    # use WENO3 and centered difference for near-boundary points
    b1up_lower = (v[:, :, 1:2] - v[:, :, 2:3]) ** 2
    b2up_lower = (v[:, :, 2:3] - v[:, :, 3:4]) ** 2
    # # Original WENO
    # w1up = (1.0 / 3.0) / (weps + b1up_lower) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up_lower) ** 2
    # Improved smoothness indicators (Borges et al., 2008, JCP)
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_lower-b2up_lower)/(b1up_lower+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_lower-b2up_lower)/(b2up_lower+weps))**2)
    weno3_upward = (w1up * ((-1.0 / 2.0) * v[:, :, 1:2] + (3.0 / 2.0) * v[:, :, 2:3]) +
                    w2up * ((1.0 / 2.0) * v[:, :, 2:3] + (1.0 / 2.0) * v[:, :, 3:4])
                    ) / (w1up + w2up)
    weno5_upward = weno5_upward.at[:, :, 0:1].set(weno3_upward)
    v_l1 = 0.5 * (v[:, :, 1:2] + v[:, :, 2:3])
    b1up_upper = (v[:, :, -4:-3] - v[:, :, -3:-2]) ** 2
    b2up_upper = (v[:, :, -3:-2] - v[:, :, -2:-1]) ** 2
    # # Original WENO
    # w1up = (1.0 / 3.0) / (weps + b1up_upper) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up_upper) ** 2
    # Improved smoothness indicators (Borges et al., 2008, JCP)
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_upper-b2up_upper)/(b1up_upper+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_upper-b2up_upper)/(b2up_upper+weps))**2)
    v_u1 = (w1up * ((-1.0 / 2.0) * v[:, :, -4:-3] + (3.0 / 2.0) * v[:, :, -3:-2]) +
               w2up * ((1.0 / 2.0) * v[:, :, -3:-2] + (1.0 / 2.0) * v[:, :, -2:-1])
               ) / (w1up + w2up)
    zero4w = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+1+2*nl.ngy, 1))
    v_upward = jnp.concatenate((zero4w,
                                v_l1,
                                weno5_upward,
                                v_u1,
                                zero4w), axis=2)
    
    # If w < 0, downward
    b1dn = ((13.0 / 12.0) * (v[:,:,5:] - 2.0 * v[:,:,4:-1] + v[:,:,3:-2]) ** 2
           + 0.25 * (v[:,:,5:] - 4.0 * v[:,:,4:-1] + 3.0 * v[:,:,3:-2]) ** 2)
    b2dn = ((13.0 / 12.0) * (v[:,:,4:-1] - 2.0 * v[:,:,3:-2] + v[:,:,2:-3]) ** 2
           + 0.25 * (v[:,:,4:-1] - v[:,:,2:-3]) ** 2)
    b3dn = ((13.0 / 12.0) * (v[:,:,3:-2] - 2.0 * v[:,:,2:-3] + v[:,:,1:-4]) ** 2
           + 0.25 * (3.0 * v[:,:,3:-2] - 4.0 * v[:,:,2:-3] + v[:,:,1:-4]) ** 2)
    # w1dn = 0.1 / (weps + b1dn) ** 2
    # w2dn = 0.6 / (weps + b2dn) ** 2
    # w3dn = 0.3 / (weps + b3dn) ** 2
    w1dn = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b1dn+weps))**2)       
    w2dn = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b2dn+weps))**2)
    w3dn = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b3dn+weps))**2)
    weno5_downward = ((w1dn * (
            (2.0/6.0) * v[:,:,5:] + (-7.0/6.0) * v[:,:,4:-1] + (11.0/6.0)*v[:,:,3:-2]) +
                w2dn * ((-1.0/6.0) * v[:,:,4:-1] + (5.0/6.0) * v[:,:,3:-2] + (2.0/6.0) * v[:,:,2:-3]) +
                w3dn * ((2.0/6.0) * v[:,:,3:-2] + (5.0/6.0) * v[:,:,2:-3] + (-1.0/6.0) * v[:,:,1:-4])) /
               (w1dn + w2dn + w3dn))
    # use WENO3 and centered difference for near-boundary points
    b1dn_upper = (v[:, :, -2:-1] - v[:, :, -3:-2]) ** 2
    b2dn_upper = (v[:, :, -3:-2] - v[:, :, -4:-3]) ** 2
    # # Original WENO
    # w1dn = (1.0 / 3.0) / (weps + b1dn_upper) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn_upper) ** 2
    # Improved smoothness indicators
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_upper-b2dn_upper)/(b1dn_upper+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_upper-b2dn_upper)/(b2dn_upper+weps))**2)
    weno3_downward = (w1dn * ((-1.0 / 2.0) * v[:, :, -2:-1] + (3.0 / 2.0) * v[:, :, -3:-2]) +
                      w2dn * ((1.0 / 2.0) * v[:, :, -3:-2] + (1.0 / 2.0) * v[:, :, -4:-3])
                      ) / (w1dn + w2dn)
    weno5_downward = weno5_downward.at[:, :, -1:].set(weno3_downward)
    v_u1 = 0.5 * (v[:, :, -2:-1] + v[:, :, -3:-2])
    b1dn_lower = (v[:, :, 3:4] - v[:, :, 2:3]) ** 2
    b2dn_lower = (v[:, :, 2:3] - v[:, :, 1:2]) ** 2
    # # Original WENO
    # w1dn = (1.0 / 3.0) / (weps + b1dn_lower) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn_lower) ** 2
    # Improved smoothness indicators
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_lower-b2dn_lower)/(b1dn_lower+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_lower-b2dn_lower)/(b2dn_lower+weps))**2)
    v_l1 = (w1dn * ((-1.0 / 2.0) * v[:, :, 3:4] + (3.0 / 2.0) * v[:, :, 2:3]) +
                      w2dn * ((1.0 / 2.0) * v[:, :, 2:3] + (1.0 / 2.0) * v[:, :, 1:2])
                      ) / (w1dn + w2dn)
    v_downward = jnp.concatenate((zero4w,
                                  v_l1,
                                  weno5_downward,
                                  v_u1,
                                  zero4w), axis=2)

    rho0w = 0.5 * (rho0[:, :, 0:-1] + rho0[:, :, 1:]) * w[:, :, 1:-1]
    rho0w8v = 0.5 * (rho0w[:, 0:-1, :] + rho0w[:, 1:, :])
    # rho0*w above v point at the w level
    flux = jax.lax.select(rho0w8v >= 0.0, rho0w8v * v_upward[:, 1:-1, :],
                          rho0w8v * v_downward[:, 1:-1, :])
    # Having zero flux at bottom and top. It can be modified later by a boundary condition function
    vertical_flux = flux[nl.ngx:-nl.ngx, nl.ngy - 1:-(nl.ngy - 1), :]
    # nz+1 layers, ghost points discarded
    return vertical_flux


def horizontal_flux_v(weps, rho0, u, v):
    """ Horizontal v-momentum fluxes using 5th-order WENO

    Original WENO from Jiang adn Shu, 1996, JCP
    """
    # Compute x-direction flux first
    # If u >= 0, westerly
    b1w = ((13.0 / 12.0) * (v[0:-5, :, :] - 2.0 * v[1:-4, :, :] + v[2:-3, :, :]) ** 2
           + 0.25 * (v[0:-5, :, :] - 4.0 * v[1:-4, :, :] + 3.0 * v[2:-3, :, :]) ** 2)
    b2w = ((13.0 / 12.0) * (v[1:-4, :, :] - 2.0 * v[2:-3, :, :] + v[3:-2, :, :]) ** 2
           + 0.25 * (v[1:-4, :, :] - v[3:-2, :, :]) ** 2)
    b3w = ((13.0 / 12.0) * (v[2:-3, :, :] - 2.0 * v[3:-2, :, :] + v[4:-1, :, :]) ** 2
           + 0.25 * (3.0 * v[2:-3, :, :] - 4.0 * v[3:-2, :, :] + v[4:-1, :, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1w = 0.1 / (weps + b1w) ** 2
    # w2w = 0.6 / (weps + b2w) ** 2
    # w3w = 0.3 / (weps + b3w) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP)
    w1w = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b1w+weps))**2)       
    w2w = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b2w+weps))**2)
    w3w = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b3w+weps))**2)
    weno5_w = ((w1w * ((2.0 / 6.0) * v[0:-5, :, :] + (-7.0 / 6.0) * v[1:-4, :, :] + (11.0 / 6.0) * v[2:-3, :, :]) +
                w2w * ((-1.0 / 6.0) * v[1:-4, :, :] + (5.0 / 6.0) * v[2:-3, :, :] + (2.0 / 6.0) * v[3:-2, :, :]) +
                w3w * ((2.0 / 6.0) * v[2:-3, :, :] + (5.0 / 6.0) * v[3:-2, :, :] + (-1.0 / 6.0) * v[4:-1, :, :])) /
               (w1w + w2w + w3w))
    # If u < 0, easterly
    b1e = ((13.0 / 12.0) * (v[5:, :, :] - 2.0 * v[4:-1, :, :] + v[3:-2, :, :]) ** 2
           + 0.25 * (v[5:, :, :] - 4.0 * v[4:-1, :, :] + 3.0 * v[3:-2, :, :]) ** 2)
    b2e = ((13.0 / 12.0) * (v[4:-1, :, :] - 2.0 * v[3:-2, :, :] + v[2:-3, :, :]) ** 2
           + 0.25 * (v[4:-1, :, :] - v[2:-3, :, :]) ** 2)
    b3e = ((13.0 / 12.0) * (v[3:-2, :, :] - 2.0 * v[2:-3, :, :] + v[1:-4, :, :]) ** 2
           + 0.25 * (3.0 * v[3:-2, :, :] - 4.0 * v[2:-3, :, :] + v[1:-4, :, :]) ** 2)
    # w1e = 0.1 / (weps + b1e) ** 2
    # w2e = 0.6 / (weps + b2e) ** 2
    # w3e = 0.3 / (weps + b3e) ** 2
    w1e = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b1e+weps))**2)       
    w2e = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b2e+weps))**2)
    w3e = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b3e+weps))**2)
    weno5_e = ((w1e * ((2.0 / 6.0) * v[5:, :, :] + (-7.0 / 6.0) * v[4:-1, :, :] + (11.0 / 6.0) * v[3:-2, :, :]) +
                w2e * ((-1.0 / 6.0) * v[4:-1, :, :] + (5.0 / 6.0) * v[3:-2, :, :] + (2.0 / 6.0) * v[2:-3, :, :]) +
                w3e * ((2.0 / 6.0) * v[3:-2, :, :] + (5.0 / 6.0) * v[2:-3, :, :] + (-1.0 / 6.0) * v[1:-4, :, :])) /
               (w1e + w2e + w3e))

    rho0u = 0.5 * (rho0[2:-3, :, :] + rho0[3:-2, :, :]) * u[3:-3, :, :]  # rho0*u at u points
    rho0u8v = 0.5 * (rho0u[:, 0:-1, :] + rho0u[:, 1:, :])  # rho0*u at corner points, to the west of v points
    flux_x = jax.lax.select(rho0u8v >= 0.0, rho0u8v * weno5_w[:, 1:-1, :], rho0u8v * weno5_e[:, 1:-1, :])
    horizontal_flux_x = flux_x[:, nl.ngy - 1:-(nl.ngy - 1), nl.ngz:-nl.ngz]  # discard ghost points

    # Compute y-direction flux
    # If v >=0, southerly
    b1s = ((13.0 / 12.0) * (v[:, 0:-5, :] - 2.0 * v[:, 1:-4, :] + v[:, 2:-3, :]) ** 2
           + 0.25 * (v[:, 0:-5, :] - 4.0 * v[:, 1:-4, :] + 3.0 * v[:, 2:-3, :]) ** 2)
    b2s = ((13.0 / 12.0) * (v[:, 1:-4, :] - 2.0 * v[:, 2:-3, :] + v[:, 3:-2, :]) ** 2
           + 0.25 * (v[:, 1:-4, :] - v[:, 3:-2, :]) ** 2)
    b3s = ((13.0 / 12.0) * (v[:, 2:-3, :] - 2.0 * v[:, 3:-2, :] + v[:, 4:-1, :]) ** 2
           + 0.25 * (3.0 * v[:, 2:-3, :] - 4.0 * v[:, 3:-2, :] + v[:, 4:-1, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1s = 0.1 / (weps + b1s) ** 2
    # w2s = 0.6 / (weps + b2s) ** 2
    # w3s = 0.3 / (weps + b3s) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP) 
    w1s = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b1s+weps))**2)       
    w2s = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b2s+weps))**2)
    w3s = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b3s+weps))**2) 
    weno5_s = ((w1s * ((2.0 / 6.0) * v[:, 0:-5, :] + (-7.0 / 6.0) * v[:, 1:-4, :] + (11.0 / 6.0) * v[:, 2:-3, :]) +
                w2s * ((-1.0 / 6.0) * v[:, 1:-4, :] + (5.0 / 6.0) * v[:, 2:-3, :] + (2.0 / 6.0) * v[:, 3:-2, :]) +
                w3s * ((2.0 / 6.0) * v[:, 2:-3, :] + (5.0 / 6.0) * v[:, 3:-2, :] + (-1.0 / 6.0) * v[:, 4:-1, :])) /
               (w1s + w2s + w3s))
    # If v < 0, northerly
    b1n = ((13.0 / 12.0) * (v[:, 5:, :] - 2.0 * v[:, 4:-1, :] + v[:, 3:-2, :]) ** 2
           + 0.25 * (v[:, 5:, :] - 4.0 * v[:, 4:-1, :] + 3.0 * v[:, 3:-2, :]) ** 2)
    b2n = ((13.0 / 12.0) * (v[:, 4:-1, :] - 2.0 * v[:, 3:-2, :] + v[:, 2:-3, :]) ** 2
           + 0.25 * (v[:, 4:-1, :] - v[:, 2:-3, :]) ** 2)
    b3n = ((13.0 / 12.0) * (v[:, 3:-2, :] - 2.0 * v[:, 2:-3, :] + v[:, 1:-4, :]) ** 2
           + 0.25 * (3.0 * v[:, 3:-2, :] - 4.0 * v[:, 2:-3, :] + v[:, 1:-4, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1n = 0.1 / (weps + b1n) ** 2
    # w2n = 0.6 / (weps + b2n) ** 2
    # w3n = 0.3 / (weps + b3n) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP) 
    w1n = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b1n+weps))**2)       
    w2n = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b2n+weps))**2)
    w3n = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b3n+weps))**2)     
    weno5_n = ((w1n * ((2.0 / 6.0) * v[:, 5:, :] + (-7.0 / 6.0) * v[:, 4:-1, :] + (11.0 / 6.0) * v[:, 3:-2, :]) +
                w2n * ((-1.0 / 6.0) * v[:, 4:-1, :] + (5.0 / 6.0) * v[:, 3:-2, :] + (2.0 / 6.0) * v[:, 2:-3, :]) +
                w3n * ((2.0 / 6.0) * v[:, 3:-2, :] + (5.0 / 6.0) * v[:, 2:-3, :] + (-1.0 / 6.0) * v[:, 1:-4, :])) /
               (w1n + w2n + w3n))

    rho0v = 0.5 * (v[:, 2:-3, :] + v[:, 3:-2, :]) * rho0[:, 2:-2, :]  # rho0*v at scalar points
    flux_y = jax.lax.select(rho0v >= 0.0, rho0v * weno5_s, rho0v * weno5_n)
    horizontal_flux_y = flux_y[nl.ngx:-nl.ngx, :, nl.ngz:-nl.ngz]  # discard ghost points

    return horizontal_flux_x, horizontal_flux_y


def vertical_flux_w(weps, rho0, w):
    """ Vertical w-momentum flux using 3rd-order WENO

    Original WENO3 from Jiang adn Shu, 1996, JCP
    """
    # If w >= 0
    b1up = (w[:, :, 0:-3] - w[:, :, 1:-2]) ** 2
    b2up = (w[:, :, 1:-2] - w[:, :, 2:-1]) ** 2
    # w1up = (1.0 / 3.0) / (weps + b1up) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up) ** 2
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b2up)/(b1up+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b2up)/(b2up+weps))**2)
    weno3_upward = (w1up * ((-1.0 / 2.0) * w[:, :, 0:-3] + (3.0 / 2.0) * w[:, :, 1:-2]) +
                    w2up * ((1.0 / 2.0) * w[:, :, 1:-2] + (1.0 / 2.0) * w[:, :, 2:-1])
                    ) / (w1up + w2up)
    weno3_upward = weno3_upward.at[:, :, 0].set(0.5 * (w[:, :, 1] + w[:, :, 2]))  # reset level 1
    # If w < 0
    b1dn = (w[:, :, 3:] - w[:, :, 2:-1]) ** 2
    b2dn = (w[:, :, 2:-1] - w[:, :, 1:-2]) ** 2
    # w1dn = (1.0 / 3.0) / (weps + b1dn) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn) ** 2
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b2dn)/(b1dn+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b2dn)/(b2dn+weps))**2)
    weno3_downward = (w1dn * ((-1.0 / 2.0) * w[:, :, 3:] + (3.0 / 2.0) * w[:, :, 2:-1]) +
                      w2dn * ((1.0 / 2.0) * w[:, :, 2:-1] + (1.0 / 2.0) * w[:, :, 1:-2])
                      ) / (w1dn + w2dn)
    weno3_downward = weno3_downward.at[:, :, -1].set(0.5 * (w[:, :, -2] + w[:, :, -3]))
    # reset reconstruction at the penultimate level (w point), assuming ngz=1

    rho0w = 0.5 * (w[:, :, 1:-2] + w[:, :, 2:-1]) * rho0[:, :, 1:-1]
    flux = jax.lax.select(rho0w >= 0.0, rho0w * weno3_upward, rho0w * weno3_downward)
    # Concatenate zero flux at bottom and top. It can be modified later by a boundary condition function
    zero4w = jnp.zeros((nl.nx, nl.ny, 1))
    vertical_flux = jnp.concatenate((zero4w, flux[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, :], zero4w), axis=2)
    # nz+2 layers, lateral ghost points discarded
    # The first level is below ground and last level above model top. Let's keep them to make the code in
    # advection calculation simpler. Moreover, we need the advection tendency at grond (w level) for the
    # Poisson-like equation later. We create a placeholder level by have a flux below ground/above top.

    return vertical_flux


def vertical_flux_w2(weps, rho0, w):
    """ Vertical w-momentum flux using 5th-order WENO

    Original WENO5 from Jiang adn Shu, 1996, JCP
    """
    # If w >= 0, upward
    b1up = ((13.0 / 12.0) * (w[:,:,0:-5] - 2.0*w[:,:,1:-4] + w[:,:,2:-3]) ** 2
           + 0.25 * (w[:,:,0:-5] - 4.0*w[:,:,1:-4] + 3.0*w[:,:,2:-3]) ** 2)
    b2up = ((13.0 / 12.0) * (w[:,:,1:-4] - 2.0*w[:,:,2:-3] + w[:,:,3:-2]) ** 2
           + 0.25 * (w[:,:,1:-4] - w[:,:,3:-2]) ** 2)
    b3up = ((13.0 / 12.0) * (w[:,:,2:-3] - 2.0*w[:,:,3:-2] + w[:,:,4:-1]) ** 2
           + 0.25 * (3.0 * w[:,:,2:-3] - 4.0 * w[:,:,3:-2] + w[:,:,4:-1]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1up = 0.1 / (weps + b1up) ** 2
    # w2up = 0.6 / (weps + b2up) ** 2
    # w3up = 0.3 / (weps + b3up) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP)
    w1up = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b1up+weps))**2)       
    w2up = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b2up+weps))**2)
    w3up = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up-b3up)/(b3up+weps))**2)
    weno5_upward = ((w1up * (
            (2.0/6.0) * w[:,:,0:-5] + (-7.0/6.0)*w[:,:,1:-4] + (11.0/6.0) * w[:,:,2:-3]) +
                w2up * ((-1.0/6.0) * w[:,:,1:-4] + (5.0/6.0) * w[:,:,2:-3] + (2.0/6.0) * w[:,:,3:-2]) +
                w3up * ((2.0/6.0) * w[:,:,2:-3] + (5.0/6.0) * w[:,:,3:-2] + (-1.0/6.0) * w[:,:,4:-1])) /
               (w1up + w2up + w3up))
    # use WENO3 and centered difference for near-boundary points
    b1up_lower = (w[:, :, 1:2] - w[:, :, 2:3]) ** 2
    b2up_lower = (w[:, :, 2:3] - w[:, :, 3:4]) ** 2
    # # Original WENO
    # w1up = (1.0 / 3.0) / (weps + b1up_lower) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up_lower) ** 2
    # Improved smoothness indicators (Borges et al., 2008, JCP)
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_lower-b2up_lower)/(b1up_lower+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_lower-b2up_lower)/(b2up_lower+weps))**2)
    weno3_upward = (w1up * ((-1.0 / 2.0) * w[:, :, 1:2] + (3.0 / 2.0) * w[:, :, 2:3]) +
                    w2up * ((1.0 / 2.0) * w[:, :, 2:3] + (1.0 / 2.0) * w[:, :, 3:4])
                    ) / (w1up + w2up)
    weno5_upward = weno5_upward.at[:, :, 0:1].set(weno3_upward)
    w_l1 = 0.5 * (w[:, :, 1:2] + w[:, :, 2:3])
    b1up_upper = (w[:, :, -4:-3] - w[:, :, -3:-2]) ** 2
    b2up_upper = (w[:, :, -3:-2] - w[:, :, -2:-1]) ** 2
    # # Original WENO
    # w1up = (1.0 / 3.0) / (weps + b1up_upper) ** 2
    # w2up = (2.0 / 3.0) / (weps + b2up_upper) ** 2
    # Improved smoothness indicators (Borges et al., 2008, JCP)
    w1up = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_upper-b2up_upper)/(b1up_upper+weps))**2)
    w2up = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1up_upper-b2up_upper)/(b2up_upper+weps))**2)
    w_u1 = (w1up * ((-1.0 / 2.0) * w[:, :, -4:-3] + (3.0 / 2.0) * w[:, :, -3:-2]) +
               w2up * ((1.0 / 2.0) * w[:, :, -3:-2] + (1.0 / 2.0) * w[:, :, -2:-1])
               ) / (w1up + w2up)
    zero4w = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, 1))
    w_upward = jnp.concatenate((zero4w,
                                w_l1,
                                weno5_upward,
                                w_u1,
                                zero4w), axis=2)
    
    # If w < 0, downward
    b1dn = ((13.0 / 12.0) * (w[:,:,5:] - 2.0 * w[:,:,4:-1] + w[:,:,3:-2]) ** 2
           + 0.25 * (w[:,:,5:] - 4.0 * w[:,:,4:-1] + 3.0 * w[:,:,3:-2]) ** 2)
    b2dn = ((13.0 / 12.0) * (w[:,:,4:-1] - 2.0 * w[:,:,3:-2] + w[:,:,2:-3]) ** 2
           + 0.25 * (w[:,:,4:-1] - w[:,:,2:-3]) ** 2)
    b3dn = ((13.0 / 12.0) * (w[:,:,3:-2] - 2.0 * w[:,:,2:-3] + w[:,:,1:-4]) ** 2
           + 0.25 * (3.0 * w[:,:,3:-2] - 4.0 * w[:,:,2:-3] + w[:,:,1:-4]) ** 2)
    # w1dn = 0.1 / (weps + b1dn) ** 2
    # w2dn = 0.6 / (weps + b2dn) ** 2
    # w3dn = 0.3 / (weps + b3dn) ** 2
    w1dn = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b1dn+weps))**2)       
    w2dn = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b2dn+weps))**2)
    w3dn = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn-b3dn)/(b3dn+weps))**2)
    weno5_downward = ((w1dn * (
            (2.0/6.0) * w[:,:,5:] + (-7.0/6.0) * w[:,:,4:-1] + (11.0/6.0)*w[:,:,3:-2]) +
                w2dn * ((-1.0/6.0) * w[:,:,4:-1] + (5.0/6.0) * w[:,:,3:-2] + (2.0/6.0) * w[:,:,2:-3]) +
                w3dn * ((2.0/6.0) * w[:,:,3:-2] + (5.0/6.0) * w[:,:,2:-3] + (-1.0/6.0) * w[:,:,1:-4])) /
               (w1dn + w2dn + w3dn))
    # use WENO3 and centered difference for near-boundary points
    b1dn_upper = (w[:, :, -2:-1] - w[:, :, -3:-2]) ** 2
    b2dn_upper = (w[:, :, -3:-2] - w[:, :, -4:-3]) ** 2
    # # Original WENO
    # w1dn = (1.0 / 3.0) / (weps + b1dn_upper) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn_upper) ** 2
    # Improved smoothness indicators
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_upper-b2dn_upper)/(b1dn_upper+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_upper-b2dn_upper)/(b2dn_upper+weps))**2)
    weno3_downward = (w1dn * ((-1.0 / 2.0) * w[:, :, -2:-1] + (3.0 / 2.0) * w[:, :, -3:-2]) +
                      w2dn * ((1.0 / 2.0) * w[:, :, -3:-2] + (1.0 / 2.0) * w[:, :, -4:-3])
                      ) / (w1dn + w2dn)
    weno5_downward = weno5_downward.at[:, :, -1:].set(weno3_downward)
    w_u1 = 0.5 * (w[:, :, -2:-1] + w[:, :, -3:-2])
    b1dn_lower = (w[:, :, 3:4] - w[:, :, 2:3]) ** 2
    b2dn_lower = (w[:, :, 2:3] - w[:, :, 1:2]) ** 2
    # # Original WENO
    # w1dn = (1.0 / 3.0) / (weps + b1dn_lower) ** 2
    # w2dn = (2.0 / 3.0) / (weps + b2dn_lower) ** 2
    # Improved smoothness indicators
    w1dn = (1.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_lower-b2dn_lower)/(b1dn_lower+weps))**2)
    w2dn = (2.0/3.0) * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1dn_lower-b2dn_lower)/(b2dn_lower+weps))**2)
    w_l1 = (w1dn * ((-1.0 / 2.0) * w[:, :, 3:4] + (3.0 / 2.0) * w[:, :, 2:3]) +
                      w2dn * ((1.0 / 2.0) * w[:, :, 2:3] + (1.0 / 2.0) * w[:, :, 1:2])
                      ) / (w1dn + w2dn)
    w_downward = jnp.concatenate((zero4w,
                                  w_l1,
                                  weno5_downward,
                                  w_u1,
                                  zero4w), axis=2)

    rho0w = 0.5 * (w[:, :, 0:-1] + w[:, :, 1:]) * rho0[:, :, :]
    flux = jax.lax.select(rho0w >= 0.0, rho0w * w_upward, rho0w * w_downward)
    vertical_flux = flux[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, :]
    # nz+2 layers, lateral ghost points discarded
    # The first level is below ground and last level above model top. Let's keep them to make the code in
    # advection calculation simpler. Moreover, we need the advection tendency at grond (w level) for the
    # Poisson-like equation later. We create a placeholder level by have a flux below ground/above top.
    return vertical_flux


def horizontal_flux_w(weps, rho0, u, v, w):
    """ Horizontal w-momentum fluxes using 5th-order WENO

    Original WENO from Jiang adn Shu, 1996, JCP
    """
    # Compute x-direction flux first
    # If u >= 0, westerly
    b1w = ((13.0 / 12.0) * (w[0:-5, :, :] - 2.0 * w[1:-4, :, :] + w[2:-3, :, :]) ** 2
           + 0.25 * (w[0:-5, :, :] - 4.0 * w[1:-4, :, :] + 3.0 * w[2:-3, :, :]) ** 2)
    b2w = ((13.0 / 12.0) * (w[1:-4, :, :] - 2.0 * w[2:-3, :, :] + w[3:-2, :, :]) ** 2
           + 0.25 * (w[1:-4, :, :] - w[3:-2, :, :]) ** 2)
    b3w = ((13.0 / 12.0) * (w[2:-3, :, :] - 2.0 * w[3:-2, :, :] + w[4:-1, :, :]) ** 2
           + 0.25 * (3.0 * w[2:-3, :, :] - 4.0 * w[3:-2, :, :] + w[4:-1, :, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP) 
    # w1w = 0.1 / (weps + b1w) ** 2
    # w2w = 0.6 / (weps + b2w) ** 2
    # w3w = 0.3 / (weps + b3w) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP)    
    w1w = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b1w+weps))**2)       
    w2w = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b2w+weps))**2)
    w3w = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1w-b3w)/(b3w+weps))**2)
    weno5_w = ((w1w * ((2.0 / 6.0) * w[0:-5, :, :] + (-7.0 / 6.0) * w[1:-4, :, :] + (11.0 / 6.0) * w[2:-3, :, :]) +
                w2w * ((-1.0 / 6.0) * w[1:-4, :, :] + (5.0 / 6.0) * w[2:-3, :, :] + (2.0 / 6.0) * w[3:-2, :, :]) +
                w3w * ((2.0 / 6.0) * w[2:-3, :, :] + (5.0 / 6.0) * w[3:-2, :, :] + (-1.0 / 6.0) * w[4:-1, :, :])) /
               (w1w + w2w + w3w))
    # If u < 0, easterly
    b1e = ((13.0 / 12.0) * (w[5:, :, :] - 2.0 * w[4:-1, :, :] + w[3:-2, :, :]) ** 2
           + 0.25 * (w[5:, :, :] - 4.0 * w[4:-1, :, :] + 3.0 * w[3:-2, :, :]) ** 2)
    b2e = ((13.0 / 12.0) * (w[4:-1, :, :] - 2.0 * w[3:-2, :, :] + w[2:-3, :, :]) ** 2
           + 0.25 * (w[4:-1, :, :] - w[2:-3, :, :]) ** 2)
    b3e = ((13.0 / 12.0) * (w[3:-2, :, :] - 2.0 * w[2:-3, :, :] + w[1:-4, :, :]) ** 2
           + 0.25 * (3.0 * w[3:-2, :, :] - 4.0 * w[2:-3, :, :] + w[1:-4, :, :]) ** 2)
    # w1e = 0.1 / (weps + b1e) ** 2
    # w2e = 0.6 / (weps + b2e) ** 2
    # w3e = 0.3 / (weps + b3e) ** 2
    w1e = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b1e+weps))**2)       
    w2e = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b2e+weps))**2)
    w3e = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1e-b3e)/(b3e+weps))**2)
    weno5_e = ((w1e * ((2.0 / 6.0) * w[5:, :, :] + (-7.0 / 6.0) * w[4:-1, :, :] + (11.0 / 6.0) * w[3:-2, :, :]) +
                w2e * ((-1.0 / 6.0) * w[4:-1, :, :] + (5.0 / 6.0) * w[3:-2, :, :] + (2.0 / 6.0) * w[2:-3, :, :]) +
                w3e * ((2.0 / 6.0) * w[3:-2, :, :] + (5.0 / 6.0) * w[2:-3, :, :] + (-1.0 / 6.0) * w[1:-4, :, :])) /
               (w1e + w2e + w3e))

    rho0u = 0.5 * (rho0[2:-3, :, :] + rho0[3:-2, :, :]) * u[3:-3, :, :]  # rho0*u at u points
    rho0u8w = 0.5 * (rho0u[:, :, 0:-1] + rho0u[:, :, 1:])  # rho0*u at edge center points, to the west of w points
    flux_x = jax.lax.select(rho0u8w >= 0.0, rho0u8w * weno5_w[:, :, 1:-1], rho0u8w * weno5_e[:, :, 1:-1])
    horizontal_flux_x = flux_x[:, nl.ngy:-nl.ngy, :]
    # discard lateral ghost points, but still having the bottom and top w level

    # Compute y-direction flux
    # If v >=0, southerly
    b1s = ((13.0 / 12.0) * (w[:, 0:-5, :] - 2.0 * w[:, 1:-4, :] + w[:, 2:-3, :]) ** 2
           + 0.25 * (w[:, 0:-5, :] - 4.0 * w[:, 1:-4, :] + 3.0 * w[:, 2:-3, :]) ** 2)
    b2s = ((13.0 / 12.0) * (w[:, 1:-4, :] - 2.0 * w[:, 2:-3, :] + w[:, 3:-2, :]) ** 2
           + 0.25 * (w[:, 1:-4, :] - w[:, 3:-2, :]) ** 2)
    b3s = ((13.0 / 12.0) * (w[:, 2:-3, :] - 2.0 * w[:, 3:-2, :] + w[:, 4:-1, :]) ** 2
           + 0.25 * (3.0 * w[:, 2:-3, :] - 4.0 * w[:, 3:-2, :] + w[:, 4:-1, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1s = 0.1 / (weps + b1s) ** 2
    # w2s = 0.6 / (weps + b2s) ** 2
    # w3s = 0.3 / (weps + b3s) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP) 
    w1s = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b1s+weps))**2)       
    w2s = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b2s+weps))**2)
    w3s = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1s-b3s)/(b3s+weps))**2) 
    weno5_s = ((w1s * ((2.0 / 6.0) * w[:, 0:-5, :] + (-7.0 / 6.0) * w[:, 1:-4, :] + (11.0 / 6.0) * w[:, 2:-3, :]) +
                w2s * ((-1.0 / 6.0) * w[:, 1:-4, :] + (5.0 / 6.0) * w[:, 2:-3, :] + (2.0 / 6.0) * w[:, 3:-2, :]) +
                w3s * ((2.0 / 6.0) * w[:, 2:-3, :] + (5.0 / 6.0) * w[:, 3:-2, :] + (-1.0 / 6.0) * w[:, 4:-1, :])) /
               (w1s + w2s + w3s))
    # If v < 0, northerly
    b1n = ((13.0 / 12.0) * (w[:, 5:, :] - 2.0 * w[:, 4:-1, :] + w[:, 3:-2, :]) ** 2
           + 0.25 * (w[:, 5:, :] - 4.0 * w[:, 4:-1, :] + 3.0 * w[:, 3:-2, :]) ** 2)
    b2n = ((13.0 / 12.0) * (w[:, 4:-1, :] - 2.0 * w[:, 3:-2, :] + w[:, 2:-3, :]) ** 2
           + 0.25 * (w[:, 4:-1, :] - w[:, 2:-3, :]) ** 2)
    b3n = ((13.0 / 12.0) * (w[:, 3:-2, :] - 2.0 * w[:, 2:-3, :] + w[:, 1:-4, :]) ** 2
           + 0.25 * (3.0 * w[:, 3:-2, :] - 4.0 * w[:, 2:-3, :] + w[:, 1:-4, :]) ** 2)
    # # Original WENO (eg, Jiang and Shu, 1996, JCP)
    # w1n = 0.1 / (weps + b1n) ** 2
    # w2n = 0.6 / (weps + b2n) ** 2
    # w3n = 0.3 / (weps + b3n) ** 2
    # Improved smoothness indicators (Borges et al, 2008, JCP) 
    w1n = 0.1 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b1n+weps))**2)       
    w2n = 0.6 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b2n+weps))**2)
    w3n = 0.3 * (1.0 + jnp.fmin(1.0e30, jnp.abs(b1n-b3n)/(b3n+weps))**2)     
    weno5_n = ((w1n * ((2.0 / 6.0) * w[:, 5:, :] + (-7.0 / 6.0) * w[:, 4:-1, :] + (11.0 / 6.0) * w[:, 3:-2, :]) +
                w2n * ((-1.0 / 6.0) * w[:, 4:-1, :] + (5.0 / 6.0) * w[:, 3:-2, :] + (2.0 / 6.0) * w[:, 2:-3, :]) +
                w3n * ((2.0 / 6.0) * w[:, 3:-2, :] + (5.0 / 6.0) * w[:, 2:-3, :] + (-1.0 / 6.0) * w[:, 1:-4, :])) /
               (w1n + w2n + w3n))

    rho0v = 0.5 * (rho0[:, 2:-3, :] + rho0[:, 3:-2, :]) * v[:, 3:-3, :]  # rho0*v at v points
    rho0v8w = 0.5 * (rho0v[:, :, 0:-1] + rho0v[:, :, 1:])  # rho0*v at edge center points, to the south of w points
    flux_y = jax.lax.select(rho0v8w >= 0.0, rho0v8w * weno5_s[:, :, 1:-1], rho0v8w * weno5_n[:, :, 1:-1])
    horizontal_flux_y = flux_y[nl.ngx:-nl.ngx, :, :]
    # discard lateral ghost points, but still having the bottom and top w level

    return horizontal_flux_x, horizontal_flux_y
