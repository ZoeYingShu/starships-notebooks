# %%
import os
from multiprocessing import Pool
from pathlib import Path
from sys import path
path.append('/home/ldang05/Starships_prj/') # path to Starships

import matplotlib.pyplot as plt
import numpy as np
import starships.planet_obs as pl_obs
from astropy import constants as const
from astropy import units as u
from starships import correlation as corr
from starships.orbite import rv_theo_t
from starships.planet_obs import Observations

#%%

# seed the random number generator
rstate = np.random.default_rng(736109)

model_file = '/home/ldang05/projects/def-dlafre/ldang05/Data/WASP-77Ab/test_mod_WASP-77_mixed_H2O_CO_main_iso.npz'
model_file = np.load(model_file)
wv_high, model_high = model_file['wave_mod'], model_file['model_spec']

if not np.isfinite(model_high[100:-100]).all():
    print("NaN in high res model spectrum encountered")
    raise ValueError()


# %%

pl_name = 'WASP-77 A b'

# --- Data parameters
obs = Observations(name=pl_name)

planet = obs.planet

# pl_name = 'WASP-77 A b'





# pl_name = 'CoRoT-2 b'
# planet.ap = 0.0281*u.au # semi-major axis
# planet.incl = (88.08*u.deg).to(u.rad)
# planet.R_star = 0.902*u.R_sun # star radius
# planet.R_pl = 1.466*u.R_jup # planet radius
# planet.M_star = 0.97*const.M_sun
# # planet_obj.RV_sys = 23*u.Unit('km/s') # another option: u.km/u.s
# planet.Teff = 5529*u.K


# --- the planet gravity and H must be changed if you change Rp and/or Tp
planet.gp = const.G * planet.M_pl / planet.R_pl**2
planet.H = (const.k_B * planet.Tp / (planet.mu * planet.gp)).decompose()


Kp_scale = (planet.M_pl / planet.M_star).decompose().value


# --- Data path ---

# high_res_path = Path('/home/ldang05/projects/def-dlafre/ldang05/Data/CoRoT-2b/CustomMasking')
# high_res_file_stem = 'corot2b_HK_4-pc_mask_wings90'

high_res_path = Path('/home/ldang05/projects/def-dlafre/ldang05/Data/WASP-77Ab/CustomMasking')
high_res_file_stem = 'wasp77_HK_4-pc_mask_wings90'

kind_trans = 'emission'

# - Which sequences are taken
do_tr = [1]

# - Selecting bad exposures if wanted/needed
bad_indexs = None

## --- Additionnal global variables
inj_alpha = 'ones'
idx_orders = slice(None)
nolog = True

# Choose over which axis the logl is summed.
# -1 or 2 should always be present (sum over spectral axis)
# It is possible to sum over multiple axis, like orders and spectra
# with (-2, -1) or equivalently, (1, 2).
axis_sum = -1  # axis along which the logl is summed.

# -----------------------------------------------------------
## LOAD HIGHRES DATA
# log.debug(f'Hires files stem: {high_res_path / high_res_file_stem}')
# log.info('Loading Hires files.')
data_info, data_trs = pl_obs.load_sequences(high_res_file_stem, do_tr, path=high_res_path)

data_info['bad_indexs'] = bad_indexs
#%%

# Save useful quantities for logl computation in the data_trs dictionary to reduce computation load.
for data_tr in data_trs.values():
    # 'flux' is in fact pre-divided by the uncertainties. Un-do that to get the actual planet signal.
    # data_tr['pl_signal'] = data_tr['flux'] * data_tr['noise']
    
    # Pre-compute all values for the logl that are independent of the model, for all orders.
    data_tr['alpha'] = np.ones_like(data_tr['t_start'])
    data_tr['uncert_sum'] = np.sum(np.log(data_tr['noise'][:, idx_orders]), axis=axis_sum)
    data_tr['N'] = np.sum(~data_tr['flux'][:, idx_orders].mask, axis=axis_sum)
    data_tr['s2f'] = np.sum(data_tr['flux'][:, idx_orders]**2, axis=axis_sum)
    

# %%

def calc_logl_G_plain(model, alpha=1., beta=1., tr_key=None, N=None, s2f=None, uncert_sum=None, idx_ord=None, axis=None):
    """Compute log likelihood for Gibson2020 without the optimization of the noise."""
    
    # Get values from the data_trs dictionary if not given.
    if N is None:
        N = data_trs[tr_key]['N']
        
    if s2f is None:
        s2f = data_trs[tr_key]['s2f']
        
    if uncert_sum is None:
        uncert_sum = data_trs[tr_key]['uncert_sum']
        
    if idx_ord is None:
        idx_ord = idx_orders
        
    if axis is None:
        axis = axis_sum
    
    # Get flux / uncert
    flux = data_trs[tr_key]['flux'][:, idx_ord]
    
    # Divide model by the uncertainty
    model = (model / data_trs[tr_key]['noise'])[:, idx_ord]

    # Compute each term of the chi2
    R = np.ma.sum(model * flux, axis=axis) 
    s2g = np.ma.sum(model**2, axis=axis)
    
    # Compute the chi2 and the log likelihood.
    chi2 = (s2f - 2 * alpha * R + alpha**2 * s2g) / beta**2
    cst = -N / 2 * np.log(2. * np.pi) - N * np.log(beta) - uncert_sum
    logl = cst - chi2 / 2
    
    return logl


def calc_logl_G_nobeta(model, alpha=1., tr_key=None, N=None, s2f=None, uncert_sum=None, idx_ord=None, axis=None):
    """Compute log likelihood for Gibson2020 without the optimization of the noise."""
    
    # Get values from the data_trs dictionary if not given.
    if N is None:
        N = data_trs[tr_key]['N']
        
    if s2f is None:
        s2f = data_trs[tr_key]['s2f']
        
    if uncert_sum is None:
        uncert_sum = data_trs[tr_key]['uncert_sum']
        
    if idx_ord is None:
        idx_ord = idx_orders
        
    if axis is None:
        axis = axis_sum
    
    # Get flux / uncert
    flux = data_trs[tr_key]['flux'][:, idx_ord]
    
    # Divide model by the uncertainty
    model = (model / data_trs[tr_key]['noise'])[:, idx_ord]

    # Compute each term of the chi2
    R = np.ma.sum(model * flux, axis=axis) 
    s2g = np.ma.sum(model**2, axis=axis)
    
    # Compute the chi2 and the log likelihood.
    chi2 = (s2f - 2 * alpha * R + alpha**2 * s2g)
    # logl = - (N / 2) * np.ma.log(chi2 / N)
    
    return chi2

    
def log_like_detailed(theta):
    """Return the log likelihood for each exposure and each order."""
    v_sys, kp = theta

    logl_i = []
    # --- Computing the logL for all sequences
    for tr_i in data_trs.keys():
        vrp_orb = rv_theo_t(kp, data_trs[tr_i]['t_start'] * u.d, planet.mid_tr,
                            planet.period, plnt=True).value
        

        
        
        # args = (v_sys, data_trs[tr_i], planet, wv_high, model_high)
        # kwargs = dict(vrp_orb=vrp_orb, vr_orb=-vrp_orb * Kp_scale, nolog=nolog,
        #             alpha=np.ones_like(data_trs[tr_i]['t_start']), kind_trans=kind_trans)
        # logl_tr = corr.calc_log_likelihood_grid_retrieval(*args, **kwargs)
        
        # Get the model sequence.
        velocities = v_sys + vrp_orb - vrp_orb * Kp_scale + data_trs[tr_i]['RV_const']
        model_seq = corr.gen_model_sequence_noinj(velocities,
                                                data_tr=data_trs[tr_i],
                                                planet=planet,
                                                model_wave=wv_high[20:-20],
                                                model_spec=model_high[20:-20],
                                                kind_trans=kind_trans,
                                                alpha=data_trs[tr_i]['alpha']
                                                )

        # Calculate the log likelihood.
        logl_tr = calc_logl_G_plain(model_seq, tr_key=tr_i)

        if not np.isfinite(logl_tr).all():
            return -np.inf

        logl_i.append(logl_tr)
        
    # Concatenate the log likelihoods for all sequences.
    logl_i = np.concatenate(np.array(logl_i), axis=0)

    # total = corr.sum_logl(logl_i, data_info['trall_icorr'], idx_orders,
    #                     data_info['trall_N'], axis=0, del_idx=data_info['bad_indexs'], nolog=True,
    #                     alpha=data_info['trall_alpha_frac'])
    
    return logl_i


# %%

log_like_detailed(np.array([ 10, 150]))


# %%
# Compute the log likelihood for a grid of values of Kp and v_sys.
# Time to find the ideal number of process
from time import time

kp, vsys = np.meshgrid(np.linspace(140., 240., 200),
                     np.linspace(-60., 60., 180))


start = time()
print("Preparing the map with pool...")
n_process = 4
with Pool(n_process) as pool:
    logl_map = pool.map(log_like_detailed, np.array([np.ravel(vsys), np.ravel(kp)]).T)
print(f"Time elapsed with {n_process} processes: {time() - start:.2f} s")
data_shape = data_trs['0']['flux'].shape
logl_map = np.reshape(logl_map, (*kp.shape, *logl_map[0].shape))

# Save logl_map and kp, vsys for later use
out_path = Path('/home/ldang05/scratch/dynasty_maps')
# Create the output directory if it doesn't exist.
if not out_path.exists():
    out_path.mkdir()
np.savez(out_path / 'logl_map_detailed.npz', logl_map=logl_map, kp=kp, vsys=vsys)


# %%

lmap = np.load(Path('/home/ldang05/scratch/dynasty_maps/logl_map_detailed.npz'))
logl_map=lmap['logl_map']
kp=lmap['kp']
vsys=lmap['vsys']
plt.imshow(np.sum(logl_map, axis=(-1, -2)))
# %%
