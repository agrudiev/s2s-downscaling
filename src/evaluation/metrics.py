import numpy as np
import properscoring as ps
from scipy.stats import wasserstein_distance as wass_dist

from utils import get_cdf, get_pdf
from engineering.spectrum import get_1dpsd


# Metric functions in alphabetical order:

def cdf_distance(obs, sim, n_quantiles=200, distance="l2"):
    """
    Calculate the CDF distance between two arrays.

    ## Parameters:
    distance (string): Distance metric to use. Options are "l1", "l2", and "max". 
    obs (array): Observed values.
    sim (array): Simulated values.
    The latest corresponds to the Kolmogorov-Smirnov distance.
    

    ## Returns:
    cdf_distance (float): CDF distance.
    """
    # Define bins for the histogram
    global_max = max(np.nanmax(obs), np.nanmax(sim))
    global_min = min(np.nanmin(obs), np.nanmin(sim))
    wide = abs(global_max - global_min) / n_quantiles
    bins = np.arange(global_min, global_max + wide, wide)
    
    # Get CDF
    cdf_obs = get_cdf(obs, bins)
    cdf_sim = get_cdf(sim, bins)
    
    match distance:
        case "l1":
            cdf_distance = np.mean(np.abs(cdf_obs - cdf_sim))
        case "l2":
            cdf_distance = np.sqrt(np.mean((cdf_obs - cdf_sim) ** 2))
        case "max":
            cdf_distance = np.max(np.abs(cdf_obs - cdf_sim))
        case _:
            raise ValueError(f"Invalid distance metric: {distance}")
            
    return cdf_distance


def crps(obs, sim):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) between two arrays.
    Currently using the properscoring library.
    
    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.
    
    ## Returns:
    crps (float): Continuous Ranked Probability Score.
    """
    return np.mean(ps.crps_ensemble(obs, sim))


def mean_absolute_error(obs, sim, axis=(0,1,2)):
    """
    Calculate the Mean Absolute Error (MAE) between two arrays.

    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.
    axis (tuple): Axes over which to compute the MAE. Default is (0,1,2).

    ## Returns:
    mae (float): Mean Absolute Error.
    """
    return np.mean(np.abs(obs - sim))


def perkins_skill_score(obs, sim, n_quantiles=2000):
    """
    Calculate the Perkins Skill Score (PSS) between two arrays.

    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.

    ## Returns:
    pss (float): Perkins Skill Score.
    """
    # Find global min and max values across observed and historical simulated data
    global_max = np.maximum(np.nanmax(obs), np.nanmax(sim))
    global_min = np.minimum(np.nanmin(obs), np.nanmin(sim))
    
    # Define bins for the histogram
    wide = np.abs(global_max - global_min) / n_quantiles
    if np.isnan(wide):
        return np.NAN
    bins = np.arange(global_min, global_max + wide, wide)
    
    pdf_obs = get_pdf(obs, bins)
    pdf_sim = get_pdf(sim, bins)
    
    pss = np.sum(np.minimum(pdf_obs, pdf_sim))
    
    return pss


def psd_distance(
    distance,
    obs,
    obs_x_length,
    obs_y_length,
    sim,
    sim_x_length=None,
    sim_y_length=None,
    num=100,
):
    """
    Calculate the Potential Spectral Density (PSD) distance between two arrays.
    If the physical lengths of the x and y axes are not provided for the simulated data,
    it is assumed that they are the same as the observed data (this will save some computation time).

    ## Parameters:
    distance (string): Distance metric to use. Options are "l1", "l2", and "max".
    obs (array): Observed values.
    obs_x_length (float): Physical length of the x-axis.
    obs_y_length (float): Physical length of the y-axis.
    sim (array): Simulated values.
    sim_x_length (optional, float): Physical length of the x-axis for the simulated data. Default is None.
    sim_y_length (optional, float): Physical length of the y-axis for the simulated data. Default is None.
    num (int): Number of points to interpolate the PSD. Standard value is 100.

    ## Returns:
    psd-distance (float): PSD distance.
    """
    different_lengths = True
    if (sim_x_length is None) & (sim_y_length is None):
        sim_x_length = obs_x_length
        sim_y_length = obs_y_length
        different_lengths = False
        
    # Get wavelengths and PSDs
    obs_wavelengths, obs_psd = get_1dpsd(obs, obs_x_length, obs_y_length)
    sim_wavelengths, sim_psd = get_1dpsd(sim, sim_x_length, sim_y_length)
    
    if different_lengths:
        # Interpolate to the commons wavelengths
        min_wavelength = np.max([obs_wavelengths[0], sim_wavelengths[0]])
        max_wavelength = np.min([obs_wavelengths[-1], sim_wavelengths[-1]])
        wavelengths = np.linspace(min_wavelength, max_wavelength, num=num)
        obs_psd = np.interp(wavelengths, obs_wavelengths, obs_psd)
        sim_psd = np.interp(wavelengths, sim_wavelengths, sim_psd)
    
    else:
        # Use the observed wavelengths (as simulated will be the same)
        wavelengths = obs_wavelengths
    
    match distance:
        case "l1":
            psd_distance = np.mean(np.abs(obs_psd - sim_psd))
        case "l2":
            psd_distance = np.sqrt(np.mean((obs_psd - sim_psd)**2))
        case "max":
            psd_distance = np.max(np.abs(obs_psd - sim_psd))
        case _:
            raise ValueError(f"Invalid distance metric: {distance}")
            
    return psd_distance


def root_mean_squared_error(obs, sim):
    """
    Calculate the Mean Squared Error (MSE) between two arrays.

    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.

    ## Returns:
    mse (float): Mean Squared Error.
    """
    return np.sqrt(np.mean((obs - sim)**2))


def wasserstein_distance(obs, sim):
    """
    Calculate the Wasserstein distance between two arrays.
    Currently using the properscoring library.
    
    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.
    
    ## Returns:
    wasserstein_distance (float): Wasserstein distance.
    """
    return wass_dist(obs, sim)