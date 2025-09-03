""" Save model physical states to Zarr stores """

import netCDF4 as nc4
import numpy as np
import namelist_n_constants as nl
import jax
import jax.numpy as jnp
import one_step_integration as one
import zarr
import asyncio


def save2zarr(phys_state, sfc_others, sprint_i, model_time):
    """ Write data to a Zarr file """
    filename = nl.file_name_format % sprint_i

    zarr_store = zarr.storage.LocalStore(filename)
    root = zarr.create_group(store=zarr_store)
    time = root.create_array(name="time", shape=(1,), dtype="float64")
    time[:] = model_time

    (theta_now, u_now, v_now, w_now, pip_now, qv_now) = phys_state
    (info, pip_const, tau_x, tau_y, sen, evap, t_ref, q_ref, u10n) = sfc_others

    x_size, y_size, z_size = nl.nx, nl.ny, nl.nz
    x4u_size = nl.nx + 1
    y4v_size = nl.ny + 1
    z4w_size = nl.nz + 1

    compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

    theta = root.create_array(name="theta", shape=(x_size, y_size, z_size), chunks=(x_size, y_size, z_size), dtype="float64", compressors=compressors)
    theta[:]= theta_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]

    pip = root.create_array(name="pip", shape=(x_size, y_size, z_size), chunks=(x_size, y_size, z_size), dtype="float64", compressors=compressors)
    pip[:]= pip_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]

    qv = root.create_array(name="qv", shape=(x_size, y_size, z_size), chunks=(x_size, y_size, z_size), dtype="float64", compressors=compressors)
    qv[:]= qv_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]

    u = root.create_array(name="u", shape=(x4u_size, y_size, z_size), chunks=(x4u_size, y_size, z_size), dtype="float64", compressors=compressors)
    u[:]= u_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]
    v = root.create_array(name="v", shape=(x_size, y4v_size, z_size), chunks=(x_size, y4v_size, z_size), dtype="float64", compressors=compressors)
    v[:]= v_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]
    w = root.create_array(name="w", shape=(x_size, y_size, z4w_size), chunks=(x_size, y_size, z4w_size), dtype="float64", compressors=compressors)
    w[:]= w_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]

    if nl.sfc_opt:
        tau_x_z = root.create_array(name="tau_x", shape=(x_size, y_size), chunks=(x_size, y_size), dtype="float64", compressors=compressors)
        tau_x_z[:]= tau_x
        tau_y_z = root.create_array(name="tau_y", shape=(x_size, y_size), chunks=(x_size, y_size), dtype="float64", compressors=compressors)
        tau_y_z[:]= tau_y

        sen_z = root.create_array(name="sen", shape=(x_size, y_size), chunks=(x_size, y_size), dtype="float64", compressors=compressors)
        sen_z[:]= sen
        evap_z = root.create_array(name="evap", shape=(x_size, y_size), chunks=(x_size, y_size), dtype="float64", compressors=compressors)
        evap_z[:]= evap

    return filename



