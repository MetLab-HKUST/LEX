""" Functions for computing boundary conditions """
import jax
import jax.numpy as jnp
import numpy as np

import namelist_n_constants as nl


def atm_ocn_flux(z_bottom, u_bottom, v_bottom, theta_bottom, q_bottom, rho_bottom, surface_t):
    """ Surface fluxes at the atmosphere-ocean interface

    This scheme is the default scheme of CAM6 in CESM2.
    Conley, A. J., Garcia, R., Kinnison, D., Lamarque, J. F., Marsh, D., Mills, M., ... & Taylor, M. A. (2012).
    Description of the NCAR community atmosphere model (CAM 5.0). NCAR technical note, 3.
    """
    v_mag = jnp.maximum(jnp.sqrt(u_bottom ** 2 + v_bottom ** 2),
                        nl.atm_ocn_min_wind)  # assuming surface current is 0 m/s
    ssq = 0.98 * q_sat_sfc(surface_t) / rho_bottom
    # sea surface humidity (kg/kg); surface temperature should have the unit K
    delta_t = theta_bottom - surface_t  # potential temperature difference (K)
    delta_q = q_bottom - ssq  # specific humidity difference (kg/kg)
    alz = jnp.log(z_bottom / nl.sfc_z_ref)
    cp = nl.Cp * (1.0 + nl.Cpvir * ssq)

    # first estimate of Z/L and u_star, t_star, and q_star
    rdn = jnp.sqrt(cdn(v_mag))
    rhn = jax.lax.select(delta_t > 0.0, 0.018, 0.0327)
    ren = 0.0346
    u_star = rdn * v_mag
    t_star = rhn * delta_t
    q_star = ren * delta_q

    for i in range(nl.atm_ocn_max_iter):
        # To jit, we have to use fixed number of iteration
        hol = nl.Karman * nl.g * z_bottom * (
                t_star / theta_bottom + q_star / (1.0 / nl.repsm1 + q_bottom)) / u_star ** 2
        hol = jnp.where(hol > 10.0, 10.0, hol)
        hol = jnp.where(hol < -10.0, -10.0, hol)
        psi_mh, psi_xh = psi_m_s(hol)
        # shift wind speed using old coefficient
        rd = rdn / (1.0 + rdn / nl.Karman * (alz - psi_mh))
        u10n = v_mag * rd / rdn
        # update transfer coefficients at 10m and neutral stability
        rdn = jnp.sqrt(cdn(u10n))
        ren = 0.0346
        rhn = jax.lax.select(hol > 0.0, 0.018, 0.0327)
        # shift all coefficients to measurement height and stability
        rd = rdn / (1.0 + rdn / nl.Karman * (alz - psi_mh))
        rh = rhn / (1.0 + rhn / nl.Karman * (alz - psi_xh))
        re = ren / (1.0 + ren / nl.Karman * (alz - psi_xh))
        # update u_star, t_star, q_star using updated, shifted coefficients
        u_star = rd * v_mag
        t_star = rh * delta_t
        q_star = re * delta_q

    # compute fluxes
    tau = rho_bottom * u_star * u_star
    # momentum flux
    tau_x = tau * u_bottom / v_mag
    tau_y = tau * v_bottom / v_mag
    # heat flux
    sen = cp * tau * t_star / u_star
    lat = nl.lat_vap * tau * q_star / u_star
    evaporation = lat / nl.lat_vap

    # compute diagnostics: 2m ref T & Q, 10m wind speed squared
    hol2 = hol * nl.sfc_z_t_ref / z_bottom
    _, psi_x2 = psi_m_s(hol2)
    al2 = jnp.log(nl.sfc_z_ref / nl.sfc_z_t_ref)
    fac = (rh / nl.Karman) * (alz + al2 - psi_xh + psi_x2)
    t_ref = theta_bottom - delta_t * fac
    t_ref = t_ref - 0.01 * nl.sfc_z_t_ref  # potential temperature to temperature correction
    fac = (re / nl.Karman) * (alz + al2 - psi_xh + psi_x2)
    q_ref = q_bottom - delta_q * fac
    # the minus sign is multiplied below because the original algorithm define downward as positive
    return -tau_x, -tau_y, -sen, -evaporation, t_ref, q_ref, u10n


def q_sat_sfc(tk):
    """ Calculate saturation vapor density """
    q_sat = 640380.0 / jnp.exp(5107.4 / tk)
    return q_sat


def cdn(u_ref):
    """ Calculate C_D at neutral condition at 10 m """
    c_d = 0.0027 / u_ref + 0.000142 + 0.0000764 * u_ref
    return c_d


def psi_m_s(hol):
    """ Calculate the integrated flux profile for momentum and scalars """
    psi_ms_stable = -5.0 * hol

    xsq = jnp.maximum(jnp.sqrt(jnp.abs(1.0 - 16.0 * hol)), 1.0)
    xqq = jnp.sqrt(xsq)

    psi_m_unstable = jnp.log((1.0 + xqq * (2.0 + xqq)) * (1.0 + xqq * xqq) / 8.0) - 2.0 * jnp.arctan(xqq) + 1.571
    psi_s_unstable = 2.0 * jnp.log((1.0 + xqq * xqq) / 2.0)

    psi_m_select = jax.lax.select(hol > 0, psi_ms_stable, psi_m_unstable)
    psi_s_select = jax.lax.select(hol > 0, psi_ms_stable, psi_s_unstable)
    return psi_m_select, psi_s_select


def set_w_bc(w):
    """ Set w values at the bottom and top interface """
    w_zeros = np.zeros((nl.nx + 2 * nl.ngx, nl.ny + 2 * nl.ngy, 2))  # dummy array with 1s
    w = w.at[:, :, 0:2].set(w_zeros)
    w = w.at[:, :, -2:].set(w_zeros)
    return w


def set_rho0_bc(rho0):
    """ Set rho0 value below/above domain boundary. Mainly we want to avoid dividing by 0. """
    rho0 = rho0.at[:, :, 0].set(rho0[:, :, 1])
    rho0 = rho0.at[:, :, -1].set(rho0[:, :, -2])
    return rho0


def rayleigh_damping(tauh, tauf, u, v, w, theta):
    """ Upper-level Rayleigh damping """
    u_tendency = -nl.rd_alpha * (0.5 * (tauh[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] +
                                        tauh[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]
                                        ) *
                                 u[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]
                                 )
    # The damping here assume the base state has u0 = 0
    v_tendency = -nl.rd_alpha * (0.5 * (tauh[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz] +
                                        tauh[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz]
                                        ) *
                                 v[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]
                                 )
    # The damping here assume the base state has v0 = 0
    w_tendency = -nl.rd_alpha * (tauf[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] *
                                 w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]
                                 )
    # The damping here assume the base state has w0 = 0
    theta_avg = jnp.mean(theta, axis=(0, 1), keepdims=True)
    theta_tendency = -nl.rd_alpha * tauh[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * (
            theta[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] - theta_avg)
    # Damping towards the horizontal mean of theta
    return u_tendency, v_tendency, w_tendency, theta_tendency
