""" Functions for calculating the advection terms """
import jax
import jax.numpy as jnp
import namelist_n_constants as nl


def get_divergence(rho0, u, v, w, x3d4u, y3d4v, z3d4w):
    """ Compute the divergence of (rho0*u, rho0*v, rho0*w) """
    rho8u_part = 0.5 * (rho0[0:-1, :, :] + rho0[1:, :, :])
    rho8v_part = 0.5 * (rho0[:, 0:-1, :] + rho0[:, 1:, :])
    rho8w_part = 0.5 * (rho0[:, :, 0:-1] + rho0[:, :, 1:])
    rho8u = jnp.concatenate((rho8u_part[-1, :, :], rho8u_part, rho8u_part[0, :, :]), axis=0)  # periodic boundary
    rho8v = jnp.concatenate((rho8v_part[:, -1, :], rho8v_part, rho8v_part[:, 0, :]), axis=1)  # periodic boundary
    zero4w = jnp.zeros((nl.nx + 2 * nl.ngx, nl.ny + 2 * nl.ngy, 1))
    rho8w = jnp.concatenate((zero4w, rho8w_part, zero4w), axis=2)

    rho_u = rho8u * u
    rho_v = rho8v * v
    rho_w = rho8w * w

    div_x = (rho_u[1:, :, :] - rho_u[0:-1, :, :]) / (x3d4u[1:, :, :] - x3d4u[0:-1, :, :])
    div_y = (rho_v[:, 1:, :] - rho_v[:, 0:-1, :]) / (y3d4v[:, 1:, :] - y3d4v[:, 0:-1, :])
    div_z = (rho_w[:, :, 1:] - rho_w[:, :, 0:-1]) / (z3d4w[:, :, 1:] - z3d4w[:, :, 0:-1])
    div_rho_u = div_x + div_y + div_z
    return div_rho_u


def advection_scalar(rho0, scalar, u, v, w, x3d4u, y3d4v, z3d4w):
    """ Compute the advection term for a scalar """


def vertical_flux_scalar(weps, rho0, w, scalar):
    """ Vertical scalar flux using 3rd-order WENO

    Original WENO3 from Jiang adn Shu, 1996, JCP
    """
    # If w>=0
    b1up = (scalar[:, :, 0:-3] - scalar[:, :, 1:-2]) ** 2
    b2up = (scalar[:, :, 1:-2] - scalar[:, :, 2:-1]) ** 2
    w1up = (1.0 / 3.0) / (weps + b1up) ** 2
    w2up = (2.0 / 3.0) / (weps + b2up) ** 2
    weno3_upward = (w1up * ((-1.0 / 2.0) * scalar[:, :, 0:-3] + (3.0 / 2.0) * scalar[:, :, 1:-2]) +
                    w2up * ((1.0 / 2.0) * scalar[:, :, 1:-2] + (1.0 / 2.0) * scalar[:, :, 2:-1])
                    ) / (w1up + w2up)
    # If w<0
    b1dn = (scalar[:, :, 3:] - scalar[:, :, 2:-1]) ** 2
    b2dn = (scalar[:, :, 2:-1] - scalar[:, :, 1:-2]) ** 2
    w1dn = (1.0 / 3.0) / (weps + b1dn) ** 2
    w2dn = (2.0 / 3.0) / (weps + b2dn) ** 2
    weno3_downward = (w1dn * ((-1.0 / 2.0) * scalar[:, :, 3:] + (3.0 / 2.0) * scalar[:, :, 2:-1]) +
                      w2dn * ((1.0 / 2.0) * scalar[:, :, 2:-1] + (1.0 / 2.0) * scalar[:, :, 1:-2])
                      ) / (w1dn + w2dn)

    rho0_w = 0.5 * (rho0[:, :, 1:-2] + rho0[:, :, 2:-1]) * w[:, :, 2:-2]  # rho0*w at w points
    flux = jax.lax.select(rho0_w >= 0.0, rho0_w * weno3_upward, rho0_w * weno3_downward)
    # Concatenate zero flux at bottom and top. It can be modified later by a boundary condition function
    zero4w = jnp.zeros((nl.nx + 2 * nl.ngx, nl.ny + 2 * nl.ngy, 1))
    vertical_flux = jnp.concatenate((zero4w, flux, zero4w), axis=2)  # nz+1 layers

    return vertical_flux


def weno5(scalar):
    """ 5th order WENO reconstruction """
