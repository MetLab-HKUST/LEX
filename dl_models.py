""" Deep neural networks for parameterizing SGS processes

Check the network with the following commands
    x = jnp.ones((1,40,40,40,5))
    tabulate_fn = nn.tabulate(autoencoder(), jax.random.key(0), compute_flops=False, compute_vjp_flops=False)
    print(tabulate_fn(x))
We place the staggered points at grid centers to make it easier to handle ghost points.
"""

from flax import linen as nn
from typing import List
import jax
import jax.numpy as jnp
import namelist_n_constants as nl
import zarr
import pressure_gradient_coriolis as pres_grad


class simpleCNN(nn.Module):
    """ A very simple CNN to help test the training code """
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='SAME', use_bias=True)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='SAME', use_bias=True)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='SAME', use_bias=True)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=5, kernel_size=(3,3,3), strides=(1,1,1), padding='SAME', use_bias=True)(x)
        return x


class AutoEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3,3,3),strides=(1,1,1), padding='SAME', use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(2,2,2), padding='SAME', use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(2,2,2), padding='SAME', use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3,3,3),strides=(2,2,2), padding='SAME', use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3,3,3),strides=(2,2,2), padding='SAME', use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3,3,3),strides=(2,2,2), padding='SAME', use_bias=True)(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))   # flatten
        x = nn.Dense(features=512, use_bias=True)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0],1,1,1,512))

        x = nn.ConvTranspose(features=256, kernel_size=(3,3,3), strides=(2,2,2), padding='SAME', use_bias=True)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=128, kernel_size=(3,3,3), strides=(2,2,2), padding='VALID', use_bias=True)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(3,3,3), strides=(2,2,2), padding='SAME', use_bias=True)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(3,3,3), strides=(2,2,2), padding='SAME', use_bias=True)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(3,3,3), strides=(2,2,2), padding='SAME', use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3,3,3),strides=(1,1,1), padding='SAME', use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Conv(features=5, kernel_size=(3,3,3), strides=(1,1,1), padding='SAME', use_bias=True)(x)

        return x


class Zcnn(nn.Module):
    @nn.compact
    def __call__(self, x):
        """ Padding bottom and top with zerros to avoid periodic condition in the vertical direction """
        x_shape = x.shape
        bottom = jnp.zeros((x_shape[0], x_shape[1], x_shape[2], 1, x_shape[4]))
        top = jnp.zeros((x_shape[0], x_shape[1], x_shape[2], 1, x_shape[4]))
        x = jnp.concatenate((bottom, x, top), axis=3)

        x = nn.Conv(features=64, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)

        x = nn.Conv(features=64, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)

        x = nn.Conv(features=64, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)
        x = nn.Conv(features=32, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x.at[:, :, :, [0,-1], :].set(0.0)
        x = nn.gelu(x)
        
        x = nn.Conv(features=5, kernel_size=(3,3,3),strides=(1,1,1), padding='CIRCULAR', use_bias=True)(x)
        x = x[:, :, :, 1:-1, :]

        return x

    
def phys_state_tuple2array(phys_state, scaling_params):
    """ Convert physical state tuple to a stacked array for deep learning model """
    theta_min, theta_max, u_min, u_max, v_min, v_max, w_min, w_max, qv_min, qv_max, _, _, _, _ = scaling_params
    
    theta, u, v, w, pip_prev, qv = phys_state
    theta = (theta[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
             jnp.min(theta_min.squeeze(), axis=(0,1))) / (
                 jnp.max(theta_max.squeeze(), axis=(0,1)) -
                 jnp.min(theta_min.squeeze(), axis=(0,1)) + 0.01)
    u = (u[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
         jnp.min(u_min.squeeze(), axis=(0,1))) / (
             jnp.max(u_max.squeeze(), axis=(0,1)) -
             jnp.min(u_min.squeeze(), axis=(0,1)) + 0.01)
    v = (v[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
         jnp.min(v_min.squeeze(), axis=(0,1))) / (
             jnp.max(v_max.squeeze(), axis=(0,1)) -
             jnp.min(v_min.squeeze(), axis=(0,1)) + 0.01)
    w = (w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
         jnp.min(w_min.squeeze(), axis=(0,1))) / (
             jnp.max(w_max.squeeze(), axis=(0,1)) -
             jnp.min(w_min.squeeze(), axis=(0,1)) + 0.01)
    qv = (qv[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] -
          jnp.min(qv_min.squeeze(), axis=(0,1))) / (
              jnp.max(qv_max.squeeze(), axis=(0,1)) -
              jnp.min(qv_min.squeeze(), axis=(0,1)) + 1.0e-5)

    ui, vi, wi = unstagger_uvw(u, v, w)
    
    phys_state_stack = jnp.stack([theta, ui, vi, wi, qv], axis=3)
    phys_state_stack = jnp.reshape(phys_state_stack, (1, nl.nx, nl.ny, nl.nz, 5))
    return phys_state_stack


def correct_array2tuple(correct):
    """ Convert correction array to a tuple with correction for each variable """
    # theta_correct, ui_correct, vi_correct, wi_correct, pip_correct, qv_correct = jnp.unstack(correct, axis=3)
    theta_correct, ui_correct, vi_correct, wi_correct, qv_correct = jnp.unstack(correct, axis=3)
    u_correct, v_correct, w_correct = stagger_uvw(ui_correct, vi_correct, wi_correct)
    return theta_correct, u_correct, v_correct, w_correct, qv_correct


def unstagger_uvw(u, v, w):
    """ Interpolate u, v, w to center points """
    ui = 0.5 * (u[0:-1, :, :] + u[1:, :, :])
    vi = 0.5 * (v[:, 0:-1, :] + v[:, 1:, :])
    wi = 0.5 * (w[:, :, 0:-1] + w[:, :, 1:])
    return ui, vi, wi


def stagger_uvw(ui, vi, wi):
    """ Interpolate u, v, w to center points """
    uc = 0.5 * (ui[0:-1, :, :] + ui[1:, :, :])
    vc = 0.5 * (vi[:, 0:-1, :] + vi[:, 1:, :])
    wc = 0.5 * (wi[:, :, 0:-1] + wi[:, :, 1:])
    u = jnp.concatenate((uc[-1:,:,:], uc, uc[0:1,:,:]), axis=0)
    v = jnp.concatenate((vc[:,-1:,:], vc, vc[:,0:1,:]), axis=1)
    w = jnp.concatenate((jnp.zeros((nl.nx, nl.ny, 1)), wc, jnp.zeros((nl.nx, nl.ny, 1))), axis=2)
    return u, v, w
