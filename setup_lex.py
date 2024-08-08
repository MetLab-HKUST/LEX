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
    theta = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    pi0 = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    pip = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    u = jnp.zeros((nl.nx+1+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))
    v = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+1+2*nl.ngy, nl.nz+2*nl.ngz))
    w = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+1+2*nl.ngz))
    qv = jnp.zeros((nl.nx+2*nl.ngx, nl.ny+2*nl.ngy, nl.nz+2*nl.ngz))

    # Setup I.C.
    if ic_option == 1:
        rho0, theta0, pi0, pip, theta, qv, u, v, w, surface_t = setup_ic_option1()
    # elif ic_option==2:
        # I.C. #2
    else:
        print("I.C.: undefined option!")
        exit()

    rho0_theta0 = rho0 * theta0
    # phys_state_ic: rho0, theta0, rho0*theta0, pi0, pip, theta_p, qv, u, v, w
    return rho0, theta0, rho0_theta0, pi0, pip, theta, qv, u, v, w, surface_t, x3d, y3d, z3d, x3d4u, y3d4v, z3d4w


def setup_ic_option1():
    """ Set up the I.C. of option #1 """
    # ...
    return rho0, theta0, pi0, pip, theta, qv, u, v, w, surface_t


def setup_damping_tau(z3d, z3d4w):
    """ Set up the ramp-up factor for Rayleigh damping """
    tauh = 0.5 * (1.0 - np.cos(np.pi * (z3d - nl.z_damping) / (z3d4w[:, :, -1] - nl.z_damping)))
    tauf = 0.5 * (1.0 - np.cos(np.pi * (z3d4w - nl.z_damping) / (z3d4w[:, :, -1] - nl.z_damping)))
    tauh = np.where(z3d <= nl.z_damping, 0.0, tauh)
    tauf = np.where(z3d4w <= nl.z_damping, 0.0, tauf)
    return tauh, tauf
