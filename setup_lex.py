""" Functions for setting up model grid and initial conditions"""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import namelist_n_constants as nl


@partial(jax.jit, static_argnames=['ic_option'])
def setup_grid_n_ic(ic_option):
    """ Set up the grid mesh and initial condition """
    # Coordinates for cell center variables
    x1d = np.linspace(0.0 - (nl.ngx - 0.5) * nl.dx, (nl.nx + nl.ngx - 0.5) * nl.dx, nl.nx + 2 * nl.ngx)
    y1d = np.linspace(0.0 - (nl.ngy - 0.5) * nl.dy, (nl.ny + nl.ngy - 0.5) * nl.dy, nl.ny + 2 * nl.ngy)
    z1d = np.linspace(0.0 - (nl.ngz - 0.5) * nl.dx, (nl.nz + nl.ngz - 0.5) * nl.dz, nl.nz + 2 * nl.ngz)
    x3d, y3d, z3d = np.meshgrid(x1d, y1d, z1d, indexing="ij")
    # Coordinates for u point variables
    x1d4u = np.linspace(0.0 - nl.ngx * nl.dx, (nl.nx + nl.ngx) * nl.dx, nl.nx + 1 + 2 * nl.ngx)
    x3d4u, _, _ = np.meshgrid(x1d4u, y1d, z1d, indexing="ij")
    # Coordinates for v point variables
    y1d4v = np.linspace(0.0 - nl.ngy * nl.dy, (nl.ny + nl.ngy) * nl.dy, nl.ny + 1 + 2 * nl.ngy)
    _, y3d4v, _ = np.meshgrid(x1d, y1d4v, z1d, indexing="ij")
    # Coordinates for w point variables
    z1d4w = np.linspace(0.0 - nl.ngz * nl.dz, (nl.nz + nl.ngz) * nl.dz, nl.nz + 1 + 2 * nl.ngy)
    _, _, z3d4w = np.meshgrid(x1d, y1d, z1d4w, indexing="ij")

    # Allocate all physics state variables for I.C.
    # "0" is used to denote the reference state; _p denotes perturbations
    rho0 = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    theta0 = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    rho0_theta0 = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    theta = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    pi0 = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    pip = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    u = jnp.zeros((nl.nx+1+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    v = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+1+2*nl.ngy, nl.nz+2*nl.ngz))
    w = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+1+2*nl.ngz))
    qv = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    surface_t = np.zeros((nl.nx, nl.ny))

    # Setup I.C.
    if ic_option == 1:
        rho0, theta0, rho0_theta0, pi0, pip, theta, qv, u, v, w, surface_t = setup_ic_option1(rho0, theta0, rho0_theta0,
                                                                                              pi0, pip, theta, qv,
                                                                                              u, v, w, surface_t,
                                                                                              x3d, y3d, z3d)
    # elif ic_option==2:
        # I.C. #2
    else:
        print("I.C.: undefined option!")
        exit()

    tauh, tauf = setup_damping_tau(z3d, z3d4w)

    rho0_theta0 = rho0 * theta0
    theta0_ic = theta0
    # physical state initial condition
    phys_ic = (rho0_theta0, rho0, theta0, theta, qv, u, v, w, pi0, pip)
    grid_ic = (theta0_ic, surface_t, x3d, y3d, z3d, x3d4u, y3d4v, z3d4w, tauh, tauf)
    return phys_ic, grid_ic


def setup_ic_option1(rho0, theta0, rho0_theta0, pi0, pip, theta, qv, u, v, w, surface_t, x3d, y3d, z3d):
    """ Set up the I.C. of option 1: a warm bubble """
    surface_t[:] = 300.0
    w = w.at[:].set(0.0)
    u = u.at[:].set(0.0)
    v = v.at[:].set(0.0)

    theta0 = theta0.at[:].set(300.0)
    d_pi0_dz = -nl.g / nl.Cp / theta0
    pi0_part = jnp.cumsum(d_pi0_dz[:, :, nl.ngz:-nl.ngz], axis=2) + 1.0
    pi0_bottom = jnp.ones((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, 1))
    pi08w = jnp.concatenate((pi0_bottom, pi0_part), axis=2)
    pi0 = pi0.at[:, :, nl.ngz:-nl.ngz].set(0.5 * (pi08w[:, :, 0:-1] + pi08w[:, :, 1:]))
    pi0 = pi0.at[:, :, (0, -1)].set(pi0[:, :, (nl.ngz, -nl.ngz)])

    rho0_theta0 = rho0_theta0.at[:].set(pi0**(nl.Cv/nl.Rd) * nl.p00 / nl.Rd)
    rho0 = rho0.at[:].set(rho0_theta0 / theta0)

    # initial center location of the warm bubble
    xc = 0.0
    yc = 0.0
    zc = 3.0
    # initial bubble radius
    xr = 2000.0
    yr = 2000.0
    zr = 2000.0
    r = jnp.sqrt(((x3d - xc) / xr)**2 + ((y3d - yc) / yr)**2 + ((z3d - zc) / zr)**2)
    theta_p = 1.0 * (jnp.cos(r * np.pi) + 1.0)
    theta_p = jnp.where(r > 1.0, 0.0, theta_p)
    theta = theta.at[:].set(theta0 + theta_p)

    pi = (rho0 * theta * nl.Rd / nl.p00)**(nl.Rd / nl.Cv)    # an estimate of total pi
    pip = pip.at[:].set(pi - pi0)

    pressure = pi**(nl.Cp / nl.Rd) * nl.p00
    t = theta * pi
    q_sat = rslf(pressure, t)
    qv = qv.at[:].set(q_sat * 0.3)    # constant RH = 30%

    return rho0, theta0, rho0_theta0, pi0, pip, theta, qv, u, v, w, surface_t


def setup_damping_tau(z3d, z3d4w):
    """ Set up the ramp-up factor for Rayleigh damping """
    tauh = 0.5 * (1.0 - np.cos(np.pi * (z3d - nl.z_damping) / (z3d4w[:, :, -1] - nl.z_damping)))
    tauf = 0.5 * (1.0 - np.cos(np.pi * (z3d4w - nl.z_damping) / (z3d4w[:, :, -1] - nl.z_damping)))
    tauh = np.where(z3d <= nl.z_damping, 0.0, tauh)
    tauf = np.where(z3d4w <= nl.z_damping, 0.0, tauf)
    return tauh, tauf


def rslf(p, t):
    """ Calculate the liquid saturation vapor mixing ratio based on temperature and pressure """
    # from Bolton (1980, MWR)
    esl = 611.2 * jnp.exp(17.67 * (t - 273.15) / (t - 29.65))
    # fix for very cold temps:
    esl = jnp.min(esl, p*0.5)
    r_sat = nl.eps * esl / (p - esl)

    return r_sat
