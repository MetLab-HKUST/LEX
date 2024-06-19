""" Namelist variables and constants for LEX """

# Time integration
dt = 10.0    # integration time step

# Grid configuration
dx = 100.0   # x-direction grid spacing in meters
dy = 100.0   # y-direction grid spacing in meters
dz = 50.0    # z-direction grid spacing in meters
nx = 100     # number of grid cells in x-direction
ny = 100     # number of grid cells in y-direction
nz = 200     # number of grid cells in z-direction
ngx = 3      # number of ghost points on one side of the x-direction
ngy = 3      # number of ghost points on one side of the y-direction
ngz = 1      # number of ghost points on one side of the z-direction

# Environment
fCor = 0.5e-4    # Coriolis parameter

# Initial condition choice
icOption = 1

# Physical constants
Rd = 287.04    # Gas constant of dry air
Rv = 461.5     # Gas constant of water vapor
Cp = 1004.64   # Specific heat of air at constant pressure
Cv = Cp-Rd     # Specific heat of air at constant volume
eps = Rd/Rv
Cpwv = 1810.0    # specific heat of water vapor
Cpvir = Cpwv / Cp - 1.0
repsm1 = Rv/Rd - 1.0
lat_vap = 2.501e6    # latent heat of evaporation
g = 9.80616    # acceleration of gravity

# WENO scheme constant
epsilon = 1.0e-18   # epsilon in WENO advection

# Surface flux scheme
atm_ocn_max_iter = 8    # maximum interation for the surface flux scheme
atm_ocn_min_wind = 0.25    # minimum wind allowed in the surface flux scheme
sfc_z_ref = 10.0    # reference height for wind, 10 m
sfc_z_t_ref = 2.0    # reference height for scalar, 2 m
Karman = 0.4    # von Karman constant

# Rayleigh damping
ir_damp = True
z_damping = 3000.0    # height above which Rayleigh damping is applied
rd_alpha = 1.0/300.0    # Inverse e-folding time for upper - level Rayleigh damping layer
