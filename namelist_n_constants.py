""" Namelist variables and constants for LEX """

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
epsilon = 1.0e-18   # epsilon in WENO advection

