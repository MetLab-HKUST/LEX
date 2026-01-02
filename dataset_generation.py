""" Generate traing and testing datasets from the benchmark simulation data  """

import jax.numpy as jnp
import jax
import numpy as np
import zarr
import pickle
import namelist_n_constants as nl
import pressure_gradient_coriolis as pres_grad


def read_sim_data(filename, grids):
    """ Load benchmark data and generate a master numpy array with all samples

    This function produces a dataset with time steps larger than the benchmark simulation's
    if num_steps > 1
    """
    num_steps = nl.dl_step_ratio
    # number or original steps, e.g., original benchmark simulation may be run at 5s time step,
    # num_step3=3 means we use 15s for the coarse resolution run
    
    zarr_store = zarr.open_group(filename, mode="r")
    theta = np.copy(zarr_store['theta'])
    qv = np.copy(zarr_store['qv'])
    u = np.copy(zarr_store['u'])
    v = np.copy(zarr_store['v'])
    w = np.copy(zarr_store['w'])
    pip = np.copy(zarr_store['pi_pert'])

    # We only keep the data before the warm bubble completely impacts the ceiling.
    # - Case 0.5K: 1800s 
    # - Case 1.0K: 1800s 
    # - Case 1.5K: 1500s
    # - Case 2.0K: 1500s
    # - Case 2.5K: 1200s
    # - Case 3.0K: 1200s
    endpoints = [360, 360, 300, 300, 240, 240]
    # calculate how many sequences we have in total

    num_seq = 0
    for i in range(6):
        num_seq = num_seq + (endpoints[i] - nl.sprint_n * num_steps) + 1

    th_seq = np.empty((num_seq, nl.sprint_n+1, 40, 40, 40))
    qv_seq = np.empty((num_seq, nl.sprint_n+1, 40, 40, 40))
    pip_seq = np.empty((num_seq, nl.sprint_n+1, 40, 40, 40))
    u_seq = np.empty((num_seq, nl.sprint_n+1, 41, 40, 40))
    v_seq = np.empty((num_seq, nl.sprint_n+1, 40, 41, 40))
    w_seq = np.empty((num_seq, nl.sprint_n+1, 40, 40, 41))
    th_seq[:] = np.nan
    qv_seq[:] = np.nan
    pip_seq[:] = np.nan
    u_seq[:] = np.nan
    v_seq[:] = np.nan
    w_seq[:] = np.nan

    seq_id = 0
    for i in range(6):
        for j in range(endpoints[i] - nl.sprint_n*num_steps + 1):
            th_seq[seq_id, :, :, :, :] = theta[i, j:j+nl.sprint_n*num_steps+1:num_steps, :, :, :]
            qv_seq[seq_id, :, :, :, :] = qv[i, j:j+nl.sprint_n*num_steps+1:num_steps, :, :, :]
            pip_seq[seq_id, :, :, :, :] = pip[i, j:j+nl.sprint_n*num_steps+1:num_steps, :, :, :]
            u_seq[seq_id, :, :, :, :] = u[i, j:j+nl.sprint_n*num_steps+1:num_steps, :, :, :]
            v_seq[seq_id, :, :, :, :] = v[i, j:j+nl.sprint_n*num_steps+1:num_steps, :, :, :]
            w_seq[seq_id, :, :, :, :] = w[i, j:j+nl.sprint_n*num_steps+1:num_steps, :, :, :]
            seq_id = seq_id + 1

    del theta, qv, pip, u, v, w

    th_has_nan = np.isnan(th_seq).any()
    qv_has_nan = np.isnan(qv_seq).any()
    u_has_nan = np.isnan(u_seq).any()
    v_has_nan = np.isnan(v_seq).any()
    w_has_nan = np.isnan(w_seq).any()
    pip_has_nan = np.isnan(pip_seq).any()
    has_nan = np.any([th_has_nan, qv_has_nan, u_has_nan, v_has_nan, w_has_nan, pip_has_nan])
    if has_nan:
        print("!!! There is NaN in the benchmark dataset !!!")

    # Get scaling parameters
    theta_min = np.min(th_seq, axis=(0,1), keepdims=True)
    theta_max = np.max(th_seq, axis=(0,1), keepdims=True)
    u_min = np.min(u_seq, axis=(0,1), keepdims=True)
    u_max = np.max(u_seq, axis=(0,1), keepdims=True)
    v_min = np.min(v_seq, axis=(0,1), keepdims=True)
    v_max = np.max(v_seq, axis=(0,1), keepdims=True)
    w_min = np.min(w_seq, axis=(0,1), keepdims=True)
    w_max = np.max(w_seq, axis=(0,1), keepdims=True)
    qv_min = np.min(qv_seq, axis=(0,1), keepdims=True)
    qv_max = np.max(qv_seq, axis=(0,1), keepdims=True)

    x3d, y3d, z3d, x3d4u, y3d4v, z3d4w, cc1, cc2, tauh, tauf = grids
    theta_p = th_seq - np.min(theta_min.squeeze(), axis=(0,1))    # strictly speaking, we should subtract base state
    del_th = laplace_of_tensor(x3d, x3d4u, y3d, y3d4v, z3d, z3d4w, theta_p)
    qv_p = qv_seq - np.min(qv_min.squeeze(), axis=(0,1))
    del_qv = laplace_of_tensor(x3d, x3d4u, y3d, y3d4v, z3d, z3d4w, qv_p)
    th_la_min = np.min(del_th, axis=(0,1), keepdims=True)
    th_la_max = np.max(del_th, axis=(0,1), keepdims=True)
    qv_la_min = np.min(del_qv, axis=(0,1), keepdims=True)
    qv_la_max = np.max(del_qv, axis=(0,1), keepdims=True)

    # print(f"... del theta max: {np.max(th_la_max)}; min {np.min(th_la_min)}")
    # print(f"... del qv max: {np.max(qv_la_max)}; min {np.min(qv_la_min)}")   

    scaling_params = (theta_min, theta_max, u_min, u_max, v_min, v_max, w_min, w_max, qv_min, qv_max, th_la_min, th_la_max, qv_la_min, qv_la_max)

    dataset = (th_seq, u_seq, v_seq, w_seq, pip_seq, qv_seq, del_th, del_qv)

    with open(f"scaling_params_{nl.sprint_n}_lookahead_steps_{num_steps}_orig_steps.pkl", 'wb') as f:
        pickle.dump(scaling_params, f)
    
    return dataset, scaling_params


def generate_training_data(dataset, seed):
    """ Generate training and testing data from the master dataset """
    rng = np.random.default_rng(seed)
    num_seq = dataset[0].shape[0]
    heads = rng.choice(np.arange(num_seq), num_seq, replace=False)

    th_seq, u_seq, v_seq, w_seq, pip_seq, qv_seq, del_th, del_qv = dataset
    th_seq = th_seq[heads, :, :, :, :]
    u_seq = u_seq[heads, :, :, :, :]
    v_seq = v_seq[heads, :, :, :, :]
    w_seq = w_seq[heads, :, :, :, :]
    pip_seq = pip_seq[heads, :, :, :, :]
    qv_seq = qv_seq[heads, :, :, :, :]
    del_th = del_th[heads, :, :, :, :]
    del_qv = del_qv[heads, :, :, :, :]

    num_train = np.int32(num_seq * 0.75)

    th_train = th_seq[0:num_train, :, :, :, :]
    u_train = u_seq[0:num_train, :, :, :, :]
    v_train = v_seq[0:num_train, :, :, :, :]
    w_train = w_seq[0:num_train, :, :, :, :]
    pip_train = pip_seq[0:num_train, :, :, :, :]
    qv_train = qv_seq[0:num_train, :, :, :, :]
    del_th_train = del_th[0:num_train, :, :, :, :]
    del_qv_train = del_qv[0:num_train, :, :, :, :]

    th_test = th_seq[num_train:, :, :, :, :]
    u_test = u_seq[num_train:, :, :, :, :]
    v_test = v_seq[num_train:, :, :, :, :]
    w_test = w_seq[num_train:, :, :, :, :]
    pip_test = pip_seq[num_train:, :, :, :, :]
    qv_test = qv_seq[num_train:, :, :, :, :]
    del_th_test = del_th[num_train:, :, :, :, :]
    del_qv_test = del_qv[num_train:, :, :, :, :]

    del th_seq, u_seq, v_seq, w_seq, pip_seq, qv_seq, del_th, del_qv

    train_ic = (padding3_batch(th_train[:, 0, :, :, :]),
                padding3u_batch(u_train[:, 0, :, :, :]),
                padding3v_batch(v_train[:, 0, :, :, :]),
                padding3_batch(w_train[:, 0, :, :, :]),
                padding3_batch(pip_train[:, 0, :, :, :]),
                padding3_batch(qv_train[:, 0, :, :, :])
                )
    train_true = (th_train[:, 1:, :, :, :],
                  u_train[:, 1:, :, :, :],
                  v_train[:, 1:, :, :, :],
                  w_train[:, 1:, :, :, :],
                  pip_train[:, 1:, :, :, :],
                  qv_train[:, 1:, :, :, :],
                  del_th_train[:, 1:, :, :, :],
                  del_qv_train[:, 1:, :, :, :]
                  )

    del th_train, u_train, v_train, w_train, pip_train, qv_train

    test_ic = (padding3_batch(th_test[:, 0, :, :, :]),
               padding3u_batch(u_test[:, 0, :, :, :]),
               padding3v_batch(v_test[:, 0, :, :, :]),
               padding3_batch(w_test[:, 0, :, :, :]),
               padding3_batch(pip_test[:, 0, :, :, :]),
               padding3_batch(qv_test[:, 0, :, :, :])
                )
    test_true = (th_test[:, 1:, :, :, :],
                 u_test[:, 1:, :, :, :],
                 v_test[:, 1:, :, :, :],
                 w_test[:, 1:, :, :, :],
                 pip_test[:, 1:, :, :, :],
                 qv_test[:, 1:, :, :, :],
                 del_th_test[:, 1:, :, :, :],
                 del_qv_test[:, 1:, :, :, :]
                 )

    del th_test, u_test, v_test, w_test, pip_test, qv_test, del_th_test, del_qv_test

    epoch_size = train_ic[0].shape[0]
    test_size = test_ic[0].shape[0]

    return train_ic, train_true, test_ic, test_true, epoch_size, test_size


def padding3_tensor(tensor):
    arr_x = np.concatenate((tensor[:, :, -nl.ngx:, :, :], tensor, tensor[:, :, 0:nl.ngx, :, :]), axis=2)
    arr_xy = np.concatenate((arr_x[:, :, :, -nl.ngx:, :], arr_x, arr_x[:, :, :, 0:nl.ngx, :]), axis=3)
    seq_size, step_size, x_size, y_size, _ = np.shape(arr_xy)
    bottom = np.reshape(arr_xy[:, :, :, :, 0], (seq_size, step_size, x_size, y_size, 1))
    top = np.reshape(arr_xy[:, :, :, :, -1], (seq_size, step_size, x_size, y_size, 1))
    arr_xyz = np.concatenate((bottom, arr_xy, top), axis=4)
    # The ghost points at the bottom and top are more like placeholders, without practical use for now.                         
    return arr_xyz

    
def padding3u_tensor(tensor):
    """ Padding an array at U point with three ghost points on each side """
    # arr0 = arr.at[-(nl.ngx+1), :, :].set(arr[nl.ngx, :, :])  # make sure b.c. is same U
    arr_x = np.concatenate((tensor[:, :, -(nl.ngx+1):-1, :, :], tensor, tensor[:, :, 1:nl.ngx+1, :, :]), axis=2)
    arr_xy = np.concatenate((arr_x[:, :, :, -nl.ngy:, :], arr_x, arr_x[:, :, :, 0:nl.ngy, :]), axis=3)
    seq_size, step_size, x_size, y_size, _ = np.shape(arr_xy)
    bottom = np.reshape(arr_xy[:, :, :, :, 0], (seq_size, step_size, x_size, y_size, 1))
    top = np.reshape(arr_xy[:, :, :, :, -1], (seq_size, step_size, x_size, y_size, 1))
    arr_xyz = np.concatenate((bottom, arr_xy, top), axis=4)
    # The ghost points at the bottom and top are more like placeholders, without practical use for now.
    return arr_xyz


def padding3v_tensor(tensor):
    """ Padding an array at V point with three ghost points on each side """
    # arr0 = arr.at[:, -(nl.ngy+1), :].set(arr[:, nl.ngy, :])  # make sure b.c is same V
    arr_x = np.concatenate((tensor[:, :, -nl.ngx:, :, :], tensor, tensor[:, :, 0:nl.ngx, :, :]), axis=2)
    arr_xy = np.concatenate((arr_x[:, :, :, -(nl.ngy+1):-1, :], arr_x, arr_x[:, :, :, 1:nl.ngy+1, :]), axis=3)
    seq_size, step_size, x_size, y_size, _ = np.shape(arr_xy)
    bottom = np.reshape(arr_xy[:, :, :, :, 0], (seq_size, step_size, x_size, y_size, 1))
    top = np.reshape(arr_xy[:, :, :, :, -1], (seq_size, step_size, x_size, y_size, 1))
    arr_xyz = np.concatenate((bottom, arr_xy, top), axis=4)
    # The ghost points at the bottom and top are more like placeholders, without practical use for now.
    return arr_xyz


def padding3_batch(tensor):
    arr_x = np.concatenate((tensor[:, -nl.ngx:, :, :], tensor, tensor[:, 0:nl.ngx, :, :]), axis=1)
    arr_xy = np.concatenate((arr_x[:, :, -nl.ngx:, :], arr_x, arr_x[:, :, 0:nl.ngx, :]), axis=2)
    seq_size, x_size, y_size, _ = np.shape(arr_xy)
    bottom = np.reshape(arr_xy[:, :, :, 0], (seq_size, x_size, y_size, 1))
    top = np.reshape(arr_xy[:, :, :, -1], (seq_size, x_size, y_size, 1))
    arr_xyz = np.concatenate((bottom, arr_xy, top), axis=3)
    # The ghost points at the bottom and top are more like placeholders, without practical use for now.                         
    return arr_xyz

    
def padding3u_batch(tensor):
    """ Padding an array at U point with three ghost points on each side """
    # arr0 = arr.at[-(nl.ngx+1), :, :].set(arr[nl.ngx, :, :])  # make sure b.c. is same U
    arr_x = np.concatenate((tensor[:,-(nl.ngx+1):-1, :, :], tensor, tensor[:, 1:nl.ngx+1, :, :]), axis=1)
    arr_xy = np.concatenate((arr_x[:, :, -nl.ngy:, :], arr_x, arr_x[:, :, 0:nl.ngy, :]), axis=2)
    seq_size, x_size, y_size, _ = np.shape(arr_xy)
    bottom = np.reshape(arr_xy[:, :, :, 0], (seq_size, x_size, y_size, 1))
    top = np.reshape(arr_xy[:, :, :, -1], (seq_size, x_size, y_size, 1))
    arr_xyz = np.concatenate((bottom, arr_xy, top), axis=3)
    # The ghost points at the bottom and top are more like placeholders, without practical use for now.
    return arr_xyz


def padding3v_batch(tensor):
    """ Padding an array at V point with three ghost points on each side """
    # arr0 = arr.at[:, -(nl.ngy+1), :].set(arr[:, nl.ngy, :])  # make sure b.c is same V
    arr_x = np.concatenate((tensor[:, -nl.ngx:, :, :], tensor, tensor[:, 0:nl.ngx, :, :]), axis=1)
    arr_xy = np.concatenate((arr_x[:, :, -(nl.ngy+1):-1, :], arr_x, arr_x[:, :, 1:nl.ngy+1, :]), axis=2)
    seq_size, x_size, y_size, _ = np.shape(arr_xy)
    bottom = np.reshape(arr_xy[:, :, :, 0], (seq_size, x_size, y_size, 1))
    top = np.reshape(arr_xy[:, :, :, -1], (seq_size, x_size, y_size, 1))
    arr_xyz = np.concatenate((bottom, arr_xy, top), axis=3)
    # The ghost points at the bottom and top are more like placeholders, without practical use for now.
    return arr_xyz


def padding_tensor(tensor):
    arr_x = np.concatenate((tensor[:, :, -1:, :, :], tensor, tensor[:, :, 0:1, :, :]), axis=2)
    arr_xy = np.concatenate((arr_x[:, :, :, -1:, :], arr_x, arr_x[:, :, :, 0:1, :]), axis=3)
    seq_size, step_size, x_size, y_size, _ = np.shape(arr_xy)
    bottom = np.reshape(arr_xy[:, :, :, :, 0], (seq_size, step_size, x_size, y_size, 1))
    top = np.reshape(arr_xy[:, :, :, :, -1], (seq_size, step_size, x_size, y_size, 1))
    arr_xyz = np.concatenate((bottom, arr_xy, top), axis=4)
    return arr_xyz


def padding_tensor_jnp(tensor):
    arr_x = jnp.concatenate((tensor[:, :, -1:, :, :], tensor, tensor[:, :, 0:1, :, :]), axis=2)
    arr_xy = jnp.concatenate((arr_x[:, :, :, -1:, :], arr_x, arr_x[:, :, :, 0:1, :]), axis=3)
    seq_size, step_size, x_size, y_size, _ = jnp.shape(arr_xy)
    bottom = jnp.reshape(arr_xy[:, :, :, :, 0], (seq_size, step_size, x_size, y_size, 1))
    top = jnp.reshape(arr_xy[:, :, :, :, -1], (seq_size, step_size, x_size, y_size, 1))
    arr_xyz = jnp.concatenate((bottom, arr_xy, top), axis=4)
    return arr_xyz


def laplace_of_tensor(x3d, x3d4u, y3d, y3d4v, z3d, z3d4w, s):
    """ Compute the Laplacian of a scalar tensor """

    s_padded = padding_tensor(s)
    s_x = (s_padded[:, :, 1:, 1:-1, 1:-1] - s_padded[:, :, 0:-1, 1:-1, 1:-1]) / (
            x3d[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
            x3d[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    s_y = (s_padded[:, :, 1:-1, 1:, 1:-1] - s_padded[:, :, 1:-1, 0:-1, 1:-1]) / (
            y3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz] -
            y3d[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz])
    s_z = (s_padded[:, :, 1:-1, 1:-1, 1:] - s_padded[:, :, 1:-1, 1:-1, 0:-1]) / (
            z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
            z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz - 1:-nl.ngz])

    s_z[:, :, (0,-1)] = 0.0

    s_xx = (s_x[:, :, 1:, :, :] - s_x[:, :, 0:-1, :, :]) / (
            x3d4u[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
            x3d4u[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    s_yy = (s_y[:, :, :, 1:, :] - s_y[:, :, :, 0:-1, :]) / (
            y3d4v[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz:-nl.ngz] -
            y3d4v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz:-nl.ngz])
    s_zz = (s_z[:, :, :, :, 1:] - s_z[:, :, :, :, 0:-1]) / (
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz + 1:-nl.ngz] -
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz + 1)])

    return s_xx + s_yy + s_zz


def laplace_of_tensor_jnp(x3d, x3d4u, y3d, y3d4v, z3d, z3d4w, s):
    """ Compute the Laplacian of a scalar tensor """

    s_padded = jnp.copy(padding_tensor_jnp(s))
    s_x = (s_padded[:, :, 1:, 1:-1, 1:-1] - s_padded[:, :, 0:-1, 1:-1, 1:-1]) / (
            x3d[nl.ngx:-(nl.ngx - 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
            x3d[nl.ngx - 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    s_y = (s_padded[:, :, 1:-1, 1:, 1:-1] - s_padded[:, :, 1:-1, 0:-1, 1:-1]) / (
            y3d[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy - 1), nl.ngz:-nl.ngz] -
            y3d[nl.ngx:-nl.ngx, nl.ngy - 1:-nl.ngy, nl.ngz:-nl.ngz])
    s_z = (s_padded[:, :, 1:-1, 1:-1, 1:] - s_padded[:, :, 1:-1, 1:-1, 0:-1]) / (
            z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:] -
            z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz - 1:-nl.ngz])

    s_z = s_z.at[:, :, 0].set(0.0)
    s_z = s_z.at[:, :, -1].set(0.0)

    s_xx = (s_x[:, :, 1:, :, :] - s_x[:, :, 0:-1, :, :]) / (
            x3d4u[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
            x3d4u[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy, nl.ngz:-nl.ngz])
    s_yy = (s_y[:, :, :, 1:, :] - s_y[:, :, :, 0:-1, :]) / (
            y3d4v[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz:-nl.ngz] -
            y3d4v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz:-nl.ngz])
    s_zz = (s_z[:, :, :, :, 1:] - s_z[:, :, :, :, 0:-1]) / (
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz + 1:-nl.ngz] -
            z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-(nl.ngz + 1)])

    return s_xx + s_yy + s_zz
