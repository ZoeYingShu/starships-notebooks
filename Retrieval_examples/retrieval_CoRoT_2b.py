from sys import path
# path.append('/home/adb/PycharmProjects/')
path.append('/home/ldang05/Starships_prj/')

import os
import sys
# # force numpy to only use 1 core max
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_DYNAMIC'] = 'FALSE'
# os.environ['MKL_CBWR'] = 'COMPATIBLE'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'

from pathlib  import Path
import numpy as np
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig()

import emcee
import starships.spectrum as spectrum
from starships.orbite import rv_theo_t
from starships.mask_tools import interp1d_masked


# %%
interp1d_masked.iprint = False
import starships.correlation as corr
from starships.analysis import bands, resamp_model
import starships.planet_obs as pl_obs
from starships.planet_obs import Observations, Planet
import starships.petitradtrans_utils as prt

from starships import retrieval_utils as ru

from starships.instruments import load_instrum


import astropy.units as u
import astropy.constants as const
from astropy.table import Table



from scipy.interpolate import interp1d

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import gc

# from petitRADTRANS import nat_cst as nc
try:
    from petitRADTRANS.physics import guillot_global, guillot_modif
except ModuleNotFoundError:
    from petitRADTRANS.nat_cst import guillot_global, guillot_modif

# initialisation

##############################################################################
# os.environ['OMP_NUM_THREADS'] = '2'
# print_var = os.environ['OMP_NUM_THREADS']
# print(f'OMP_NUM_THREADS = {print_var}')

from multiprocessing import Pool, cpu_count

# Get kwargs from command line (if called from command line)
if __name__ == '__main__':
    kwargs_from_cmd_line = dict([arg.split('=') for arg in sys.argv[1:]])
    if kwargs_from_cmd_line:
        for key, value in kwargs_from_cmd_line.items():
            log.info(f"Keyword argument from command line: {key} = {value}")
else:
    kwargs_from_cmd_line = dict()

n_pcu = 20 # cpu
nwalkers = n_pcu * 6  # emcee will use a number of cores equal to half the number of walkers

log.info(f'Using {n_pcu} cpu and {nwalkers} walkers.')


####################################

# %%

base_dir = Path('/home/ldang05/scratch/retrieval/')
# try:
#     base_dir = os.environ['SCRATCH']
#     base_dir = Path(base_dir)
# except KeyError:
#     base_dir = Path.home()

pl_name = 'CoRoT-2 b'
# Name used for filenames
pl_name_fname = '_'.join(pl_name.split())

# --- Data path ---
# high_res_path = base_dir / Path('DataAnalysis/SPIRou/Reductions/WASP-33')
# high_res_file_stem = 'v07254_wasp33_nights1-2_pc2-4_mask_wings95-90'  # 'wasp127'

high_res_path = Path('/home/ldang05/projects/def-dlafre/ldang05/Data/CoRoT-2b/CustomMasking')
high_res_file_stem = 'corot2b_HK_4-pc_mask_wings90'

wfc3_path = None
# wfc3_path = Path('/home/adb/projects/def-dlafre/adb/Observations/HST/WFC3')
# # wfc3_file = Path('WASP-33_WFC3_subsample.ecsv')
# wfc3_file = Path('WASP-33_WFC3_full_res.ecsv')

spitzer_path = None
# spitzer_path = Path('/home/adb/projects/def-dlafre/adb/Observations/Spitzer')
# spitzer_file = Path('WASP-33_Spitzer.ecsv')

# - Type of retrieval : JR, HRR or LRR
retrieval_type = 'HRR'

# In HRR mode, use white light curve from WFC3 bandpass?
white_light = False

# - Dissociation in abundance profiles?
dissociation = False
# - Will you add Spitzer data points?
add_spitzer = False
# -- which kind of TP profile you want?
kind_temp = 'guillot'

# --- Save walker steps here :
walker_path = base_dir / Path(f'DataAnalysis/walker_steps/{pl_name_fname}')
# walker_file_out = Path('walker_steps_pc4_mask_wings90.h5')

# add burnin file path, save to scratch
walker_file_out = Path('walker_steps_pc4_mask_wings90_burnin.h5') # burnin & sampling file

try:
    idx_file = kwargs_from_cmd_line['idx_file']
    walker_file_out = walker_file_out.with_stem(f'{walker_file_out.stem}_{idx_file}')
except KeyError:
    log.debug('No `idx_file` found in command line arguments.')
log.info(f'walker_file_out = {walker_file_out}')

# Walker file used to init walkers (set to None if not used)
# either None, Path('file_name.h5') or walker_file_out
# walker_file_in = walker_file_out
walker_file_in = None # if burnin
init_mode = 'burin'  # Options: "burnin", "continue"

if walker_file_in is None:
    log.info('No walker_file_in given.')
else:
    log.info(f'walker_file_in = {walker_file_in}')
    log.info(f'init_mode = {init_mode}')

# - Types of models : in emission or transmission
kind_trans = 'emission'

# %%

# --- Data parameters
obs = Observations(name=pl_name)

p = obs.planet

# --- Update some params if needed
# ** To permanently change them, use a custom exofile

# p.ap = 0.0259*u.au
# p.incl = (86.63*u.deg).to(u.rad)
# # p.R_star = 1.444 * const.R_sun  # (p.ap/3.69).to(u.R_sun)
# # p.R_pl = 1.593 * const.R_jup  # (0.1143*p.R_star).to(u.R_jup)
# p.R_star = (p.ap/3.69).to(u.R_sun)
# p.R_pl = (0.1143*p.R_star).to(u.R_jup)
# p.M_star = 1.561*const.M_sun
# p.Teff = 7300*u.K

p.ap = 0.0281*u.au # semi-major axis
p.R_star = 0.902*u.R_sun # star radius
p.R_pl = 1.466*u.R_jup # planet radius
p.M_star = 0.97*const.M_sun

# --- the planet gravity and H must be changed if you change Rp and/or Tp
p.gp = const.G * p.M_pl / p.R_pl**2
p.H = (const.k_B * p.Tp / (p.mu * p.gp)).decompose()

planet = p

Kp_scale = (planet.M_pl / planet.M_star).decompose().value


# --- For emission spectra, you need a star spectrum
# spec_star = spectrum.PHOENIXspectrum(Teff=7400, logg=4.5) # change temperature and logg
# spec_star = np.load('/home/adb/Models/RotationSpectra/phoenix_teff_07400_logg_4.50_Z0.0_RotKer_vsini50.npz')
spec_star = None

# star_wv = spec_star['grid'] # star wavelength
# # star_wv = (spec_star['wave']).to(u.um).value
# star_flux = spec_star['flux']  # .to(u.erg / u.cm**2 /u.s /u.cm).value
# --------------


# - Selecting the wanted species:
list_mols = ['H2O', 'CO']
# list_mols = ['H2O', 'CO', 'CO2', 'OH']

# - Adding continuum opacities:
# continuum_opacities = ['H-']
continuum_opacities = []

nb_mols = len(list_mols) + len(continuum_opacities)

# data and priors

## --- HIGH RES DATA ---

# --- IF HRR OR JR, YOU NEED TO INCLUDE HIGH RES DATA -----
# - Data resolution and wavelength limit for the instrument
# instrum = load_instrum('spirou')
instrum = load_instrum('igrins')
high_res_wv_lim = instrum['high_res_wv_lim'] 

# - Which sequences are taken
do_tr = [1]

# - Selecting bad exposures if wanted/needed
bad_indexs = None

## --- Additionnal global variables
plot = False
nolog = True
inj_alpha = 'ones'
orders = np.arange(54)
opacity_sampling = 4  # downsampling of the petitradtrans R = 1e6, ex.: o_s = 4 -> R=250000


# -----------------------------------------------------------
## LOAD HIGHRES DATA
log.debug(f'Hires files stem: {high_res_path / high_res_file_stem}')
if retrieval_type == 'JR' or retrieval_type == 'HRR':
    log.info('Loading Hires files.')
    data_info, data_trs = pl_obs.load_sequences(high_res_file_stem, do_tr, path=high_res_path)
    data_info['bad_indexs'] = bad_indexs


## --- LOW RES DATA ---


#%%

# --- SELECT WHICH SPITZER (or other photometric data points) DATA YOU HAVE
if add_spitzer is True:
    # --- Reading transmission functions of the broadband points
    spit_trans_i1 = Table.read(spitzer_path / 'transmission_IRAC1.txt', format='ascii')  # 3.6 um
    spit_trans_i2 = Table.read(spitzer_path / 'transmission_IRAC2.txt', format='ascii')  # 4.5 um

    fct_i1 = interp1d(spit_trans_i1['col1'], spit_trans_i1['col2'] / spit_trans_i1['col2'].max())
    fct_i2 = interp1d(spit_trans_i2['col1'], spit_trans_i2['col2'] / spit_trans_i2['col2'].max())

    # - Select which spitzer/other broadband data points to add
    # *** in the same order as your data will be given
    wave_sp = [spit_trans_i1['col1'], spit_trans_i2['col1']]  # spit_trans_f2['col1'],
    fct_sp = [fct_i1, fct_i2]  # fct_f2,

    # --- READ YOUR SPITZER DATA
    #     spit_tab = Table.read(data_path + 'WASP-33/Spitzer.csv', format='ascii.ecsv')
    #     spit_wave, spit_data, spit_data_err = spit_tab['wave'].data, spit_tab['F_p/F_star'].data, spit_tab['err'].data
    data_spit = Table.read(spitzer_path / spitzer_file)
    spit_wave = data_spit['wave'].value
    spit_data = data_spit['F_p/F_star'].value
    spit_data_err = data_spit['err'].value

    spitzer = spit_wave, spit_data, spit_data_err, wave_sp, fct_sp
else:
    spitzer = None

# --- HST DATA ---

hst = {}
if wfc3_file is not None:
    data_HST = Table.read(wfc3_path / wfc3_file)

    HST_wave = data_HST['wave'].value
    HST_data = data_HST['F_p/F_star'].value / 100
    HST_data_err = data_HST['err'].value / 100

    ### - Must give : wave, data, data_err, instum_resolution ; to the hst dictionary, for each instrument used
    hst['WFC3'] = HST_wave, HST_data, HST_data_err, 75

if white_light:
    wl_wave = HST_wave
    mean_wl = np.mean(HST_data)
    mean_wl_err = np.sqrt(np.sum(HST_data_err ** 2)/HST_data_err.size)


# --- Will you add STIS data points?
add_stis = False

if add_stis:
    data_HST = Table.read(data_path + planet_path + 'HST_data_VIS.ecsv')
    HST_wave_VIS = data_HST['wavelength'].value
    HST_data_VIS = data_HST['data'].value
    HST_data_err_VIS = data_HST['err'].value

    hst['STIS'] = HST_wave_VIS, HST_data_VIS, HST_data_err_VIS, 50


#################
# Prior Functions
#################
# SQRT2 = math.sqrt(2.)
# import math as math
# from scipy.special import gamma,erfcinv

# # Stolen from https://github.com/JohannesBuchner/MultiNest/blob/master/src/priors.f90
# def log_prior(cube,lx1,lx2):
#     return 10**(lx1+cube*(lx2-lx1))

# def uniform_prior(cube,x1,x2):
#     return x1+cube*(x2-x1)

def gaussian_prior(cube, mu, sigma):
    #     SQRT2 = math.sqrt(2.)
    #     return mu + sigma*SQRT2*erfcinv(2.0*(1.0 - cube))
    return -(((cube - mu) / sigma) ** 2.) / 2.


# def log_gaussian_prior(cube,mu,sigma):
#     SQRT2 = math.sqrt(2.)
#     bracket = sigma*sigma + sigma*SQRT2*erfcinv(2.0*cube)
#     return mu*np.exp(bracket)

# def delta_prior(cube,x1,x2):
#     return x1

# # Sanity checks on parameter ranges
# def b_range(x, b):
#     if x > b:
#         return -np.inf
#     else:
#         return 0.

def a_b_range(x, a, b):
    if x < a:
        return -np.inf
    elif x > b:
        return -np.inf
    else:
        return 0.


# --- Basis of the TP profile
# -- Choosing which type you want
# '' = default guillot, 'modif' = modif guillot, 'iso' = isothermal
temp_params = dict()
temp_params['gravity'] = p.gp.cgs.value
temp_params['T_int'] = 500.
temp_params['T_eq'] = 1535.
temp_params['M_pl'] = p.M_pl
# --- For the basic guillot profile
temp_params['kappa_IR'] = 10 ** (-2)
temp_params['gamma'] = 10 ** (1)  # 0.01
# --- For the modified guillot profile
temp_params['delta'] = 10 ** (-7.0)
temp_params['ptrans'] = 10 ** (-3)
temp_params['alpha'] = 0.3

# - Generating the basis of the temperature profile
limP=(-10, 2)  # pressure log rannge
n_pts=50  # pressure n_points
temp_params['pressures'] = np.logspace(*limP, n_pts)
P0 = 10e-3  # -- reference pressure fixed

# --- Select prior types: range or gaussian (prior types can be added in the prior function)

params_prior = {}
params_prior['abund'] = ['range', -12.0, -0.5] # same abundance range for all molecules
params_prior['temp'] = ['range', 400, 3500] # T_equilibrium
# params_prior['cloud'] = ['range', -5.0, 2]
# params_prior['rpl'] = ['range', 1.0, 1.6]
params_prior['kp'] = ['range', 170-30, 170+30] # narrow down the limit +-30
params_prior['rv'] = ['range', -10, 50] # narrow down the limit
# params_prior['wind'] = ['range', 0, 20]
params_prior['tp_kappa'] = ['range', -3, 3]
# params_prior['tp_delta'] = ['range', -8, -3]
params_prior['tp_gamma'] = ['range', -2, 6]
# params_prior['tp_ptrans'] = ['range', -8, 3]
# params_prior['tp_alpha'] = ['range', -1.0, 1.0]
# params_prior['scat_gamma'] = ['range', -6, 0]
# params_prior['scat_factor'] = ['range', 0, 10]

# params_prior['e-'] = ['range', -12.0, -0.5]

# params_prior['log_f'] = ['range', 0, 0.8]

# --- which parameters are assigned to which index *AFTER* the chemical abundances
params_id = ru.gen_params_id_p(params_prior)

# Initialize walkers using the prior
prior_init_func = {'range': ru.init_uniform_prior}
# Special treatment for some paramters
special_treatment = {'kp': ['range', 165, 175], # where do walkers start
                     'rv': ['range', 18, 24]}
walker_init = ru.init_from_prior(nwalkers, prior_init_func, params_prior,
                                 n_mol=nb_mols, special_treatment=special_treatment)

# walker_init = [-12, -12, -12, -12, -12, -12, -12, -12, -12, -12, 2500, 223.3, -1.2, -3, -2, -12, -2] +\
#                ([10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  1500,   5,  5,  6,  6, 10, 3]
#                 * np.random.uniform(size=(nwalkers, 17)))


def log_prior_params(theta, nb_mols, params_id, params_prior):
    total = 0

    # --- Checking the prior for the abundances
    for i_mol in range(nb_mols):
        total += a_b_range(theta[i_mol], *params_prior['abund'][1:])
        if ~np.isfinite(total):
            return total

    # --- Checking the prior for the rest of the params
    for key in params_id.keys():
        if params_id[key] is not None:
            if params_prior[key][0] == "range":
                total += a_b_range(theta[nb_mols + params_id[key]], *params_prior[key][1:])
            if params_prior[key][0] == "gaussian":
                total += gaussian_prior(theta[nb_mols + params_id[key]], *params_prior[key][1:])
            if ~np.isfinite(total):
                return total

    return total


####################################################
# hig vs low res obj
####################################################

# Define low res wavelength range (might not be used)
lower_wave_lim = high_res_wv_lim[0]
upper_wave_lim = high_res_wv_lim[1]

if add_spitzer:
    upper_wave_lim = 5.

if add_stis:
    lower_wave_lim = 0.3

low_res_wv_lim = (lower_wave_lim, upper_wave_lim)

log.debug(f"Low res wavelength range: {low_res_wv_lim}")

def init_model_retrieval(mol_species=None, kind_res='high', lbl_opacity_sampling=None,
                         wl_range=None, continuum_species=None, pressures=None, **kwargs):
    """
    Initialize some objects needed for modelization: atmo, species, star_fct, pressures
    :param mol_species: list of species included (without continuum opacities)
    :param kind_res: str, 'high' or 'low'
    :param lbl_opacity_sampling: ?
    :param wl_range: wavelength range (2 elements tuple or list, or None)
    :param continuum_species: list of continuum opacities, H and He excluded
    :param pressures: pressure array. Default is `temp_params['pressures']`
    :param kwargs: other kwargs passed to `starships.petitradtrans_utils.select_mol_list()`
    :return: atmos, species, star_fct, pressure array
    """

    if mol_species is None:
        mol_species = list_mols

    if lbl_opacity_sampling is None:
        lbl_opacity_sampling = opacity_sampling

    if continuum_species is None:
        continuum_species = continuum_opacities

    if pressures is None:
        pressures = temp_params['pressures']

    species = prt.select_mol_list(mol_species, kind_res=kind_res, **kwargs)

    if kind_res == 'high':
        mode = 'lbl'
        Raf = instrum['resol']
        pix_per_elem = 2
        if wl_range is None:
            wl_range = high_res_wv_lim

    elif kind_res == 'low':
        mode = 'c-k'
        Raf = 1000
        pix_per_elem = 1
        if wl_range is None:
            wl_range = low_res_wv_lim
    else:
        raise ValueError(f'`kind_res` = {kind_res} not valid. Choose between high or low')


    atmo, _ = prt.gen_atm_all([*species.keys()], pressures, mode=mode,
                                      lbl_opacity_sampling=lbl_opacity_sampling, wl_range=wl_range,
                                      continuum_opacities=continuum_species)

    # --- downgrading the star spectrum to the wanted resolution
    if kind_trans == 'emission':
        resamp_star = np.ma.masked_invalid(
            resamp_model(star_wv[(star_wv >= wl_range[0] - 0.1) & (star_wv <= wl_range[1] + 0.1)],
                         star_flux[(star_wv >= wl_range[0] - 0.1) & (star_wv <= wl_range[1] + 0.1)], 500000, Raf=Raf,
                         pix_per_elem=pix_per_elem))
        fct_star = interp1d(star_wv[(star_wv >= wl_range[0] - 0.1) & (star_wv <= wl_range[1] + 0.1)],
                                     resamp_star)
    else:
        fct_star = None

    return atmo, species, fct_star

# Define global dictionaries where to put the model infos so they will be shared in functions. (save memory)
atmos_global = {'high': None, 'low': None}
species_global = {'high': None, 'low': None}
fct_star_global = {'high': None, 'low': None}


## logl

# lnprob

####################################################

def prepare_prt_inputs(theta):

    if params_id['cloud'] is not None:
        pcloud = 10 ** (theta[nb_mols + params_id['cloud']])  # in bars
    else:
        pcloud = None

    if params_id['rpl'] is not None:
        rpl = theta[nb_mols + params_id['rpl']]
        gravity = (const.G * planet.M_pl / (rpl * const.R_jup) ** 2).cgs.value
    else:
        gravity = temp_params['gravity']
        rpl = planet.R_pl.to(u.R_jup).value

    if params_id['tp_kappa'] is not None:
        kappa = 10 ** theta[nb_mols + params_id['tp_kappa']]
    else:
        kappa = temp_params['kappa_IR']

    if params_id['tp_gamma'] is not None:
        gamma = 10 ** theta[nb_mols + params_id['tp_gamma']]
    else:
        gamma = temp_params['gamma']

    if params_id['tp_delta'] is not None:
        delta = 10 ** theta[nb_mols + params_id['tp_delta']]
    else:
        delta = temp_params['delta']

    if params_id['tp_ptrans'] is not None:
        ptrans = 10 ** theta[nb_mols + params_id['tp_ptrans']]
    else:
        ptrans = temp_params['ptrans']

    if params_id['tp_alpha'] is not None:
        alpha = theta[nb_mols + params_id['tp_alpha']]
    else:
        alpha = temp_params['alpha']

    if params_id['scat_gamma'] is not None:
        gamma_scat = theta[nb_mols + params_id['scat_gamma']]
    else:
        gamma_scat = None

    if params_id['scat_factor'] is not None:
        factor = theta[nb_mols + params_id['scat_factor']]  # * (5.31e-31*u.m**2/u.u).cgs.value
    else:
        factor = None

    params_id['wind'] = None
    if params_id['wind'] is not None:
        rot_kwargs = {'rot_params': [rpl * const.R_jup, planet.M_pl, theta[nb_mols + params_id['temp']] * u.K,
                                     [theta[nb_mols + params_id['wind']]]], 'gauss': True, 'x0': 0,
            'fwhm': theta[nb_mols + params_id['wind']] * 1e3, }
    else:
        rot_kwargs = {'rot_params': None}

    # --- Generating the temperature profile
    if kind_temp == "modif":
        temperatures = guillot_modif(temp_params['pressures'], delta, gamma, temp_params['T_int'],
                                     theta[nb_mols + params_id['temp']], ptrans, alpha)
    elif kind_temp == 'iso':
        temperatures = theta[nb_mols + params_id['temp']] * np.ones_like(temp_params['pressures'])
    else:
        temperatures = guillot_global(temp_params['pressures'], kappa, gamma, gravity,
                                      temp_params['T_int'], theta[nb_mols + params_id['temp']])

    error_kwargs = dict()
    try:
        p_id = params_id['log_f']
    except KeyError:
        error_kwargs['log_f'] = 0.
    finally:
        error_kwargs['log_f'] = theta[nb_mols + p_id]

    prt_args = temperatures, gravity, P0, pcloud, rpl * const.R_jup.cgs.value, planet.R_star.cgs.value
    prt_kwargs = dict(gamma_scat=gamma_scat, kappa_factor=factor)

    return prt_args, prt_kwargs, rot_kwargs, error_kwargs


def update_abundances(theta, mode=None, ref_species=None):

    if ref_species is None:
        if mode is None:
            ref_species = list_mols.copy()
        else:
            ref_species = list(species_global[mode].keys())

    species = {mol: 10 ** (theta[i_mol]) for i_mol, mol
               in enumerate(ref_species[:len(list_mols)])}

    if continuum_opacities is not None:
        if 'H-' in continuum_opacities:
            species['H-'] = 10 ** theta[nb_mols - 1]
            species['H'] = 10 ** (-99.)
            if 'e-' not in  params_prior.keys():
                species['e-'] = 10 ** (-6.0)

    if 'e-' in params_prior.keys():
        p_id = params_id['e-']
        species['e-'] = 10 ** (theta[nb_mols + p_id])

    return species

def prepare_model_high_or_low(theta, prt_args, prt_kwargs, mode, rot_kwargs=None,
                              atmos_obj=None, fct_star=None, species_dict=None):

    if atmos_obj is None:
        # Initiate if not done yet
        if atmos_global[mode] is None:
            log.info(f'Model not initialized for mode = {mode}. Starting initialization...')
            output = init_model_retrieval(kind_res=mode)
            log.info('Saving values in `species_global`.')
            atmos_global[mode], species_global[mode], fct_star_global[mode] = output

        atmos_obj = atmos_global[mode]

    if fct_star is None:
        fct_star = fct_star_global[mode]

    # --- Updating the abundances
    # Note that if species is None (not specified), `species_global[mode]` will be used inside `update_abundances`.
    species = update_abundances(theta, mode, species_dict)

    # --- Generating the model
    wv_out, model_out = prt.retrieval_model_plain(atmos_obj, species, planet,
                                                    temp_params['pressures'], *prt_args,
                                                    **prt_kwargs, kind_trans=kind_trans,
                                               dissociation=dissociation, fct_star=fct_star)

    if mode == 'high':
        if np.isfinite(model_out[100:-100]).all():
            # --- Downgrading and broadening the model (if broadening is included)
            wv_out, model_out = prt.prepare_model(wv_out, model_out, int(1e6 / opacity_sampling),
                                                              **rot_kwargs)

    return wv_out, model_out


def prepare_spitzer(wv_low, model_low):

    # print('Spitzer')
    spit_wave, _, _, wave_sp, fct_sp = spitzer

    spit_mod = []
    for wave_i, fct_i in zip(wave_sp, fct_sp):
        # --- Computing the model broadband point
        cond = (wv_low >= wave_i[0]) & (wv_low <= wave_i[-1])
        spit_mod.append(np.average(model_low[cond], weights=fct_i(wv_low[cond])))

    spit_mod = np.array(spit_mod)

    return spit_wave, spit_mod


def prepare_hst(wv_low, model_low, Rbf, R_sampling, instrument, wave_pad=None):

    hst_wave, _, _, hst_res = hst[instrument]
    log.debug('Prepare HST...')
    log.debug(f"hst_wave: {hst_wave}")
    log.debug(f"hst_res: {hst_res}")

    if wave_pad is None:
        d_wv_bin = np.diff(hst_wave)
        wave_pad = 10 * d_wv_bin[[0, -1]]   # This is a bit arbitrary

    cond = (wv_low >= hst_wave[0] - wave_pad[0]) & (wv_low <= hst_wave[-1] + wave_pad[-1])

    _, resamp_prt = spectrum.resampling(wv_low[cond], model_low[cond], Raf=hst_res, Rbf=Rbf, sample=wv_low[cond])
    binned_prt_hst = spectrum.box_binning(resamp_prt, R_sampling / hst_res)
    fct_prt = interp1d(wv_low[cond], binned_prt_hst)
    mod = fct_prt(hst_wave)

    return hst_wave, mod


def lnprob(theta, ):
    total = 0.

    total += log_prior_params(theta, nb_mols, params_id, params_prior)

    #     print('Prior ',total)

    if not np.isfinite(total):
        #         print('prior -inf')
        return -np.inf

    prt_args, prt_kwargs, rot_kwargs, err_kwargs = prepare_prt_inputs(theta)

    ####################
    # --- HIGH RES --- #
    ####################
    if (retrieval_type == 'JR') or (retrieval_type == 'HRR'):

        wv_high, model_high = prepare_model_high_or_low(theta, prt_args, prt_kwargs, 'high', rot_kwargs)

        if not np.isfinite(model_high[100:-100]).all():
            print("NaN in high res model spectrum encountered")
            return -np.inf

        logl_i = []
        # --- Computing the logL for all sequences
        for tr_i in data_trs.keys():
            vrp_orb = rv_theo_t(theta[nb_mols + params_id['kp']], data_trs[tr_i]['t_start'] * u.d, planet.mid_tr,
                                planet.period, plnt=True).value

            RV = theta[nb_mols + params_id['rv']]
            #             logl_tr = calc_log_likelihood_grid_retrieval_global(theta[nb_mols + params_id['rv']],
            #                                                                 tr_i,
            #                                                                 wv_high,
            #                                                                 model_high,
            #                                                                 vrp_orb,
            #                                                                 -vrp_orb * Kp_scale)
            args = (RV, data_trs[tr_i], planet, wv_high, model_high)
            kwargs = dict(vrp_orb=vrp_orb, vr_orb=-vrp_orb * Kp_scale, nolog=nolog,
                          alpha=np.ones_like(data_trs[tr_i]['t_start']), kind_trans=kind_trans)
            logl_tr = corr.calc_log_likelihood_grid_retrieval(*args, **kwargs)

            #             print(data_trs[tr_i]['flux'][0])
            if not np.isfinite(logl_tr).all():
                return -np.inf

            logl_i.append(logl_tr)
        #             print(logl_tr)
        # --- Computing the total logL
        #         spirou_logl = corr.sum_logl(np.concatenate(np.array(logl_i), axis=0),
        #                                data_info['trall_icorr'], orders, data_info['trall_N'],
        #                                axis=0, del_idx=data_info['bad_indexs'],
        #                                      nolog=True, alpha=data_info['trall_alpha_frac'])
        #         total += spirou_logl.copy()
        total += corr.sum_logl(np.concatenate(np.array(logl_i), axis=0), data_info['trall_icorr'], orders,
                               data_info['trall_N'], axis=0, del_idx=data_info['bad_indexs'], nolog=True,
                               alpha=data_info['trall_alpha_frac'])

        if (retrieval_type == "HRR") and (white_light is True):
            log.debug("Using White Light from WFC3.")
            # --- White light info ---
            Rbf = instrum['resol']
            R_sampling = int(1e6 / opacity_sampling)
            _, mod = prepare_hst(wv_high, model_high, Rbf, R_sampling, 'WFC3')
            mean_mod = np.mean(mod)
            log.debug(f"White Light value: {mean_mod}")

            total += -1 / 2 * corr.calc_chi2(mean_wl, mean_wl_err, mean_mod)

    ###################
    # --- LOW RES --- #
    ###################
    if ((retrieval_type == 'JR') and (spitzer is not None)) or (retrieval_type == 'LRR') or (atmos_global['low'] is not None):
        #         print('Low res')

        wv_low, model_low = prepare_model_high_or_low(theta, prt_args, prt_kwargs, 'low')

        if np.sum(np.isnan(model_low)) > 0:
            print("NaN in low res model spectrum encountered")
            return -np.inf

        if spitzer is not None:
            _, spit_mod = prepare_spitzer(wv_low, model_low)

            # --- Computing the logL
            _, spit_data, spit_data_err, _, _ = spitzer
            total += -1 / 2 * corr.calc_chi2(spit_data, spit_data_err, spit_mod)

    #             print('Spitzer', spitzer_logl)

    if (retrieval_type == 'JR') or (retrieval_type == 'LRR'):
        #         print('HST')

        if (retrieval_type == 'JR') and (spitzer is None):
            # --- If no Spitzer or STIS data is included, only the high res model is generated
            # and this is the model that will be downgraded for WFC3
            wv_low, model_low = wv_high, model_high
            Rbf = instrum['resol']
            R_sampling = int(1e6 / opacity_sampling)
        else:
            Rbf = 1000
            R_sampling = 1000
        #         print(Rbf)
        for instrument in hst.keys():

            _, mod = prepare_hst(wv_low, model_low, Rbf, R_sampling, instrument)

            # --- Computing the logL
            _, hst_data, hst_data_err, _ = hst[instrument]
            total += corr.calc_logl_chi2_scaled(hst_data, hst_data_err, mod, err_kwargs['log_f'])
            # total += -1 / 2 * corr.calc_chi2(hst_data, hst_data_err, mod)

        del wv_low, model_low

    if retrieval_type != 'LRR':
        del wv_high, model_high

    gc.collect()

    return total


# Testing bloc

# # --- Testing block ---

# theta = pos[-1]
# start = hm.tic()

# logl = lnprob(theta,)

# hm.toc(start)
# print(logl)

# %%
# #######################

if __name__ == '__main__':

    # Start retrieval!

    import warnings

    warnings.simplefilter("ignore", FutureWarning)
    # warnings.simplefilter("ignore", RuntimeWarning)

    # --- Walkers initialisation ---
    # -- Either a random uniform initialisation (for every parameters)
    if walker_file_in is None:
        pos = walker_init
    elif init_mode == 'burnin':
        pos, _ = ru.init_from_burnin(nwalkers, wlkr_file=walker_file_in, wlkr_path=walker_path, n_best_min=10000)
    elif init_mode == 'continue': # same number of walkers
        # Last step of the chain
        pos = ru.read_walkers_file(walker_path / walker_file_in, discard=0)[-1]
    else:
        raise ValueError(f"{init_mode} not valid.")

    nwalkers, ndim = pos.shape

    log.info(f"(Number of walker, Number of parameters) = {pos.shape}")

    # Pre-run the log liklyhood function
    log.info("Checking if log likelyhood function is working.")
    for i_walker in range(nwalkers):
        logl = lnprob(pos[i_walker])
        if np.isfinite(logl):
            log.info("Success!")
            break
    else:
        log.warning("test not successful")

    # Make sure file does not already exist
    if init_mode != 'continue':
        file_stem = walker_file_out.stem
        for idx_file in range(100):
            if (walker_path / walker_file_out).is_file():
                walker_file_out = walker_file_out.with_stem(f'{file_stem}_{idx_file}')
            else:
                break
        else:
            raise ValueError('Walker File already exists.')

    # --- backend to track evolution ---
    # Create output directory if it does not exist
    walker_path.mkdir(parents=True, exist_ok=True)
    backend = emcee.backends.HDFBackend(walker_path / walker_file_out)

    # Run it!
    with Pool(n_pcu) as pool:
        print('Initialize sampler...')
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        pool=pool,
                                        backend=backend, a=2)  ### step size -- > à changer
        print('Starting the retrieval!')
        sampler.run_mcmc(pos, 500, progress=False)  # , skip_initial_state_check=True)
        # 500 for burnin
        # 5000-10000 for sampling

    log.info('End of retrieval. It seems to be a success!')
