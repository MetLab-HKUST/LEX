""" Large-Eddy simulation in JAX (LEX)

This is the LES in JAX (LEX) model. Its governing equations are based on the pseudo-incompressible equations.

DURRAN DR. A physically motivated approach for filtering acoustic waves from the equations governing compressible stratified flow. Journal of Fluid Mechanics. 2008;601:365-379. doi:10.1017/S0022112008000608

We use the Arakawa-C staggered grid for spatial discretization and the leapfrog time scheme. The Poisson-like equation for normalized pressure (Exner function) is solved using finite differencing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import netCDF4 as nc4
from . import namelist as nl
from . import setup_lex as setl



    
    


    
