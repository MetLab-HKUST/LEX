""" Compute potential temperature tendency due to radiation """

import jax.numpy as jnp
import namelist_n_constants as nl


def newton_radiation(theta, theta0_ic):
    """ Hm / (Cp * pi0), Newtonian relaxation for now.

    This will be replaced by a real radiation scheme in the future.
    The default timescale is 12 hours.
    """
    theta_tend = -(theta - theta0_ic) / (12.0 * 3600.0)
    theta_tend = jnp.where(theta - theta0_ic > 1.0, -1.0 / (12.0 * 3600.0), theta_tend)
    theta_tend = jnp.where(theta - theta0_ic < -1.0, 1.0 / (12.0 * 3600.0), theta_tend)
    return theta_tend[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]
