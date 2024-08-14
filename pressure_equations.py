""" Functions for solving the Poisson-type pressure equation """

import jax.numpy as jnp
from advection import get_divergence, get_2d_divergence
import namelist_n_constants as nl


def laplace_of_pressure(x3d, x3d4u, y3d, y3d4v, z3d, z3d4w, rtt, pi):
    """ Compute the left-hand-side of the pressure equation

    rtt: rho_tilde * theta_tilde * theta
    pi: pi', the Exner function perturbation
    Assuming that rtt has ghost points, but pi doesn't, because the linear algebra solver requires a function with the
    same input and output vector/array shape. rtt's bottom and top ghost level contain the boundary condition for
    rho*theta_tilde*theta*dpi'/dz, which can be obtained by assuming w=0 at the bottom and top interface in the w
    equation.
    """
    pi_padded = padding_array(pi)
    dpi_dx = (pi_padded[1:, 1:-1, 1:-1] - pi_padded[0:-1, 1:-1, 1:-1]) / (
            x3d[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
            x3d[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    dpi_dy = (pi_padded[1:-1, 1:, 1:-1] - pi_padded[1:-1, 0:-1, 1:-1]) / (
            y3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz] -
            y3d[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz])
    dpi_dz = (pi_padded[1:-1, 1:-1, 1:] - pi_padded[1:-1, 1:-1, 0:-1]) / (
            z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
            z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz - 1:-nl.ngz])

    pi_x = 0.5 * (rtt[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] +
                  rtt[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) * dpi_dx
    pi_y = 0.5 * (rtt[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz] +
                  rtt[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz]) * dpi_dy
    pi_z = 0.5 * (rtt[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] +
                  rtt[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz - 1:-nl.ngz]) * dpi_dz
    pi_z = pi_z.at[:, :, 0].set(rtt[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, 0])
    pi_z = pi_z.at[:, :, -1].set(rtt[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, -1])

    pi_xx = (pi_x[1:, :, :] - pi_x[0:-1, :, :]) / (
            x3d4u[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
            x3d4u[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    pi_yy = (pi_y[:, 1:, :] - pi_y[:, 0:-1, :]) / (
            y3d4v[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz:-nl.ngz] -
            y3d4v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz:-nl.ngz])
    pi_zz = (pi_z[:, :, 1:] - pi_z[:, :, 0:-1]) / (
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz + 1:-nl.ngz] -
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz + 1)])

    return nl.Cp * (pi_xx + pi_yy + pi_zz)


def rhs_of_pressure_equation(rho0_theta0, pi0, rtt, u, v, w, adv4u, adv4v, adv4w, fu, fv, buoyancy,
                             x3d, x3d4u, y3d, y3d4v, z3d4w):
    """ Compute the right hand side of the pressure equation """
    rhs_adv = get_divergence(rho0_theta0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                          adv4u, adv4v, adv4w, x3d4u[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                          y3d4v[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                          z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    # Assuming that rho0_theta0 has all the ghost points, but advection tendencies have no ghost points

    rhs_cor = get_2d_divergence(rho0_theta0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                  fv, -fu, x3d4u[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
                                  y3d4v[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    # Assuming that fv and fu have been put onto u and v points, respectively, and have not ghost points

    rho0_theta0_b = interpolate_scalar2w(rho0_theta0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * buoyancy)
    # Assuming buoyancy B has no ghost points; interpolate from scalar to w points and include bottom/top w level values
    rhs_buoy = (rho0_theta0_b[:, :, 1:] - rho0_theta0_b[:, :, 0:-1]) / (
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz + 1:-nl.ngz] -
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz + 1)])

    rhs_pres = - horizontal_laplace_of_pressure_grad(x3d, x3d4u, y3d, y3d4v,
                                                    rtt, pi0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])

    rhs = rhs_adv + rhs_cor + rhs_buoy + rhs_pres
    return rhs, rhs_adv, rhs_cor, rhs_buoy, rhs_pres


def horizontal_laplace_of_pressure_grad(x3d, x3d4u, y3d, y3d4v, rtt, pi):
    """ Compute the left-hand-side of the pressure equation

    rtt: rho_tilde * theta_tilde * theta
    pi: pi_tilde, the base state Exner function perturbation
    Assuming that rtt has ghost points, but pi doesn't, because the linear algebra solver requires a function with the
    same input and output vector/array shape.
    """
    pi_padded = padding_array(pi)
    dpi_dx = (pi_padded[1:, 1:-1, 1:-1] - pi_padded[0:-1, 1:-1, 1:-1]) / (
            x3d[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
            x3d[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    dpi_dy = (pi_padded[1:-1, 1:, 1:-1] - pi_padded[1:-1, 0:-1, 1:-1]) / (
            y3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz] -
            y3d[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz])

    pi_x = 0.5 * (rtt[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] +
                  rtt[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]) * dpi_dx
    pi_y = 0.5 * (rtt[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz] +
                  rtt[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz]) * dpi_dy

    pi_xx = (pi_x[1:, :, :] - pi_x[0:-1, :, :]) / (
            x3d4u[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
            x3d4u[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    pi_yy = (pi_y[:, 1:, :] - pi_y[:, 0:-1, :]) / (
            y3d4v[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz:-nl.ngz] -
            y3d4v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz:-nl.ngz])

    return nl.Cp * (pi_xx + pi_yy)


def padding_array(arr):
    """ Padding an array with one ghost point on each side """
    arr_x = jnp.concatenate((arr[-1:, :, :], arr, arr[0:1, :, :]), axis=0)
    arr_xy = jnp.concatenate((arr_x[:, -1:, :], arr_x, arr_x[:, 0:1, :]), axis=1)
    x_size, y_size, _ = jnp.shape(arr_xy)
    bottom = jnp.reshape(arr_xy[:, :, 0], (x_size, y_size, 1))
    top = jnp.reshape(arr_xy[:, :, -1], (x_size, y_size, 1))
    arr_xyz = jnp.concatenate((bottom, arr_xy, top), axis=2)
    # The ghost points at the bottom and top are more like placeholders.
    return arr_xyz


def interpolate_scalar2w(scalar):
    """ Interpolate scalar array to w points, including bottom and top levels, but no ghost points. """
    scalar8w_part = 0.5 * (scalar[:, :, 0:-1] + scalar[:, :, 1:])
    bottom, top = extrapolate_bottom_top(scalar)
    scalar8w = jnp.concatenate((bottom, scalar8w_part, top), axis=2)
    return scalar8w


def extrapolate_bottom_top(scalar):
    """ Extrapolate a scalar array to bottom and top w levels

    We use second order Lagrange polynomial for the extrapolation, assuming the input scalar array has no ghost points.
    """
    # The extrapolation coefficients below assume there is no stretching near the bottom and top levels. They should
    # be adjusted if that is not true.
    cgs1 = 1.875
    cgs2 = -1.25
    cgs3 = 0.375
    cgt1 = 1.875
    cgt2 = -1.25
    cgt3 = 0.375
    bottom = cgs1 * scalar[:, :, 0] + cgs2 * scalar[:, :, 1] + cgs3 * scalar[:, :, 2]
    top = cgt1 * scalar[:, :, -1] + cgt2 * scalar[:, :, -2] + cgt3 * scalar[:, :, -3]
    x_size, y_size = top.shape
    bottom = jnp.reshape(bottom, (x_size, y_size, 1))
    top = jnp.reshape(top, (x_size, y_size, 1))
    return bottom, top


def calculate_rtt(rho0_theta0, theta, buoyancy):
    """ Calculate rho0*theta0*theta

    Include bottom and top ghost points to save boundary conditions for rho*theta_tilde*theta*dpi/dz,
    which can be obtained by assuming w=0 at the bottom and top interface in the w equation.
    Assuming buoyancy term has no ghost points
    """
    rtt_part = rho0_theta0[:, :, nl.ngz:-nl.ngz] * theta[:, :, nl.ngz:-nl.ngz]
    bottom, top = extrapolate_bottom_top(rho0_theta0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * buoyancy / nl.Cp)
    bottom_x = jnp.concatenate((bottom[-nl.ngx:, :, 0], bottom[:, :, 0], bottom[0:nl.ngx, :, 0]), axis=0)
    bottom_xy = jnp.concatenate((bottom_x[:, -nl.ngy:], bottom_x, bottom_x[:, 0:nl.ngy]), axis=1)
    bottom_xy = jnp.reshape(bottom_xy, (nl.nx + 2 * nl.ngx, nl.ny + 2 * nl.ngy, 1))
    top_x = jnp.concatenate((top[-nl.ngx:, :, 0], top[:, :, 0], top[0:nl.ngx, :, 0]), axis=0)
    top_xy = jnp.concatenate((top_x[:, -nl.ngy:], top_x, top_x[:, 0:nl.ngy]), axis=1)
    top_xy = jnp.reshape(top_xy, (nl.nx + 2 * nl.ngx, nl.ny + 2 * nl.ngy, 1))
    rtt = jnp.concatenate((bottom_xy, rtt_part, top_xy), axis=2)
    return rtt


def calculate_buoyancy(theta0, theta_p, qv):
    """ Calculate buoyancy term """
    ###### b = nl.g * (theta_p / theta0 + nl.repsm1 * qv)
    b = nl.g * (theta_p / theta0)    # ignore water vapor
    b8w = interpolate_scalar2w(b[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    return b[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz], b8w
