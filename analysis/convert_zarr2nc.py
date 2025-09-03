""" Convert a few variables in Zarr to netCDF """

import netCDF4 as nc4
import numpy as np
import zarr
import namelist_n_constants as nl


file_nums = [0, 10, 20, 30]

for file_num in file_nums:
    file_name_format = "../experiments/lex_out_%0.4i.zarr"     
    filename = file_name_format % file_num

    zarr_store = zarr.open_group(filename, mode="r")
    th = np.copy(zarr_store['theta'])
    qv = np.copy(zarr_store['qv'])
    u = np.copy(zarr_store['u'])
    v = np.copy(zarr_store['v'])
    w = np.copy(zarr_store['w'])
    ds_file = nc4.Dataset("../experiments/lex_reference_state.nc")
    pi0 = np.copy(ds_file.variables["pi0"])
    pip = np.copy(zarr_store['pip'])

    file_name_format = "lex_out_%0.4i.nc"     
    filename = file_name_format % file_num
    nc_file = nc4.Dataset(filename, mode="w", format="NETCDF4")
    nc_file.createDimension("time", None)
    nc_file.createDimension("x", nl.nx)
    nc_file.createDimension("y", nl.ny)
    nc_file.createDimension("z", nl.nz)
    nc_file.createDimension("x4u", nl.nx+1)
    nc_file.createDimension("y4v", nl.ny+1)
    nc_file.createDimension("z4w", nl.nz+1)

    itime = nc_file.createVariable("time", np.float64, ("time",))
    itime.long_name = "time (s)"
    x_nc = nc_file.createVariable("x", np.float64, ("x",))
    x_nc.long_name = "x (m)"
    y_nc = nc_file.createVariable("y", np.float64, ("y",))
    y_nc.long_name = "y (m)"
    z_nc = nc_file.createVariable("z", np.float64, ("z",))
    z_nc.long_name = "z (m)"
    x4u_nc = nc_file.createVariable("x4u", np.float64, ("x4u",))
    x4u_nc.long_name = "x (m) for u points"
    y4v_nc = nc_file.createVariable("y4v", np.float64, ("y4v",))
    y4v_nc.long_name = "y (m) for v points"
    z4w_nc = nc_file.createVariable("z4w", np.float64, ("z4w",))
    z4w_nc.long_name = "z (m) for w points"

    itime[:] = file_num * nl.sprint_n * nl.dt
    x = np.copy(ds_file.variables["x"])
    x_nc[:] = x
    y = np.copy(ds_file.variables["y"])
    y_nc[:] = y
    z = np.copy(ds_file.variables["z"])
    z_nc[:] = z
    x4u = np.linspace(0.0, nl.nx * nl.dx, nl.nx + 1)
    x4u_nc[:] = x4u
    y4v = np.linspace(0.0, nl.ny * nl.dy, nl.ny + 1)
    y4v_nc[:] = y4v
    z4w = np.linspace(0.0, nl.nz * nl.dz, nl.nz + 1)
    z4w_nc[:] = z4w

    x_size, y_size, z_size = nl.nx, nl.ny, nl.nz

    th_nc = nc_file.createVariable("th", np.float64, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    th_nc[:] = np.reshape(th, (1, x_size, y_size, z_size))
    qv_nc = nc_file.createVariable("qv", np.float64, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    qv_nc[:] = np.reshape(qv, (1, x_size, y_size, z_size))
    pip_nc = nc_file.createVariable("pip", np.float64, ("time", "x", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    pip_nc[:] = np.reshape(pip, (1, x_size, y_size, z_size))

    u_nc = nc_file.createVariable("u", np.float64, ("time", "x4u", "y", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    u_nc[:] = np.reshape(u, (1, x_size+1, y_size, z_size))
    v_nc = nc_file.createVariable("v", np.float64, ("time", "x", "y4v", "z"), fill_value=1.0e36, zlib=True, complevel=2)
    v_nc[:] = np.reshape(v, (1, x_size, y_size+1, z_size))
    w_nc = nc_file.createVariable("w", np.float64, ("time", "x", "y", "z4w"), fill_value=1.0e36, zlib=True, complevel=2)
    w_nc[:] = np.reshape(w, (1, x_size, y_size, z_size+1))

    nc_file.close()


