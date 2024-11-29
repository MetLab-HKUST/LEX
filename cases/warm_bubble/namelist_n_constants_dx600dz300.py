""" Namelist variables and constants for LEX """

# Time integration
dt = 30            # one-step integration time step
sprint_n = 1       # dt*sprint_n is the interval of data saving
relay_n = 60       # number of sprints, relay_n*sprint_n*dt is the total integration time
asselin_r = 0.25   # r factor in the Asselin filtering
integrate_opt = 1  # 1: SSPRK3, 2: Leapfrog

# Grid configuration
dx = 600.0   # x-direction grid spacing in meters
dy = 600.0   # y-direction grid spacing in meters
dz = 300.0   # z-direction grid spacing in meters
nx = 40      # number of grid cells in x-direction
ny = 40      # number of grid cells in y-direction
nz = 40      # number of grid cells in z-direction
ngx = 3      # number of ghost points on one side of the x-direction
ngy = 3      # number of ghost points on one side of the y-direction
ngz = 1      # number of ghost points on one side of the z-direction

# output file name
file_name_format = "experiments/lex_out_%0.4i.nc"
base_file_name = "experiments/lex_reference_state.nc"
save_num_levels = 1

# Initial condition choice
ic_option = 1
rand_opt = False

# Coriolis force
cor_opt = False
fCor = 0.0   # Coriolis parameter

# Surface flux scheme
sfc_opt = False
atm_ocn_max_iter = 8    # maximum interation for the surface flux scheme
atm_ocn_min_wind = 0.25    # minimum wind allowed in the surface flux scheme
sfc_z_ref = 10.0    # reference height for wind, 10 m
sfc_z_t_ref = 2.0    # reference height for scalar, 2 m
Karman = 0.4    # von Karman constant

# Turbulence model
turb_opt = 1    # 1: Smagorinsky

# Rayleigh damping
damp_opt = True
z_damping = 9600.0    # height above which Rayleigh damping is applied
rd_alpha = 1.0/300.0  # Inverse e-folding time for upper - level Rayleigh damping layer

# Diabatic heating
rad_opt = False      # turn on "radiation" or not

# pi' correction
pic_opt = False      # correction is needed if we need the actual density

# Physical constants
Rd = 287.04    # Gas constant of dry air
Rv = 461.5     # Gas constant of water vapor
Cp = 1004.64   # Specific heat of air at constant pressure
Cv = Cp-Rd     # Specific heat of air at constant volume
eps = Rd/Rv
Cpwv = 1810.0           # specific heat of water vapor
Cpvir = Cpwv / Cp - 1.0
repsm1 = Rv/Rd - 1.0
lat_vap = 2.501e6       # latent heat of evaporation
g = 9.80616             # acceleration of gravity
p00 = 100000.0          # reference surface pressure
