import numpy as np
from utils import normalize_data


def _get_2d_spectrum_info(data, x_length, y_length):
    # Get spatial dimensions
    _, Ny, Nx = data.shape

    # Get radial wavenumber coordinate (from cartesian to polar coordinates)
    kx = np.fft.fftfreq(Nx, d=x_length/Nx)
    ky = np.fft.fftfreq(Ny, d=y_length/Ny)
    
    # Compute 2D Fourier Transform
    fft_data = np.fft.fft2(data, axes=(1,2))
    
    return (Ny, Nx), (kx, ky), fft_data


def get_1dpsd(data, x_length, y_length, data_std=1, rotation_angle=0):
    # Normalize data
    data = normalize_data(data, norm_std=data_std)[0]
    
    # Get 2D spectrum information
    (Ny, Nx), (kx, ky), fft_data = _get_2d_spectrum_info(data, x_length, y_length)
    Nt = data.shape[0]

    # Get radial wavenumber coordinate (from cartesian to polar coordinates)
    ky_grid, kx_grid = np.meshgrid(kx, ky) # yes it's reversed
    
    # Apply rotation to the grid
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle),  np.cos(rotation_angle)]
    ])
    kx_grid_rotated = rotation_matrix[0, 0] * kx_grid + rotation_matrix[0, 1] * ky_grid
    ky_grid_rotated = rotation_matrix[1, 0] * kx_grid + rotation_matrix[1, 1] * ky_grid

    # Calculate the new wavenumber grid after rotation
    k_grid = np.sqrt(kx_grid_rotated**2 + ky_grid_rotated**2)
    
    # Define 1D wavenumber bins
    Nbins = int(np.sqrt(Nx*Ny)/2)
    eps = 1e-5
    k_bins = np.linspace(0, np.max(k_grid)+eps, Nbins)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    # Compute 1D PSD for each time step t
    ps_t = np.zeros([len(k_centers), Nt])
    for t in range(Nt):
        ps_t[:, t] = np.histogram(k_grid, bins=k_bins, weights=np.abs(fft_data[t, :, :])**2)[0]
    
    psd_t = ps_t / (Nx*Ny)**2
    hist = np.histogram(k_grid, bins=k_bins)[0]
    mask = hist != 0
    psd_t = psd_t[mask,:] / hist[mask,None]
    psd = np.mean(psd_t, axis=1)
    return k_centers[::-1], psd[::-1]

    
def get_2dpsd(data, x_length, y_length, norm_std=1):
    # Normalize data
    data = normalize_data(data, norm_std=norm_std)[0]
    
    # Get 2D spectrum information
    (Ny, Nx), (kx, ky), fft_data = _get_2d_spectrum_info(data, x_length, y_length)
    
    # Get 2D PSD for each time step t
    psd_2d_t = np.abs(fft_data)**2 / (Nx*Ny)**2
    
    # Average over time
    psd_2d = np.mean(psd_2d_t, axis=0)
    
    return (kx, ky), psd_2d


def radial_low_pass_filter(data, cutoff_k, x_length, y_length):
    # Get 2D spectrum information
    (_, _), (kx, ky), fft_data = _get_2d_spectrum_info(data, x_length, y_length)

    # Tranformation to polar coordinates
    ky_2d, kx_2d = np.meshgrid(ky, kx, indexing='ij')
    k_2d =  np.sqrt(ky_2d**2 + kx_2d**2)

    # Apply the low-pass filter
    filter_mask = k_2d < cutoff_k
    fft_data_filtered = fft_data * filter_mask

    # Perform an inverse 2D FFT to get the filtered data
    return np.fft.ifft2(fft_data_filtered).real


def rectangular_low_pass_filter(data, cutoff_k, x_length, y_length):
    # Get 2D spectrum information
    (_, _), (kx, ky), fft_data = _get_2d_spectrum_info(data, x_length, y_length)
    
    # Put kx and ky in a 2D grid
    ky_2d, kx_2d = np.meshgrid(ky, kx, indexing='ij')
    
    # Rectangular low-pass filter
    mask = (np.abs(kx_2d) <= cutoff_k[0]) & (np.abs(ky_2d) <= cutoff_k[1])
    fft_data_filtered = fft_data * mask[None, :, :]

    # Perform an inverse 2D FFT to get the filtered data
    return np.fft.ifft2(fft_data_filtered).real


def get_k_star(wavelengths, era5_psd, cpc_psd, threshold=1e-4):
    # TODO: write exact method
    # Find indices where PSD values are below the threshold
    mask_era5 = era5_psd < threshold
    mask_cpc = cpc_psd < threshold

    # Select the PSD values and wavelengths that meet the condition
    era5_psd_below_threshold = era5_psd[mask_era5]
    cpc_psd_below_threshold = cpc_psd[mask_cpc]
    wavelengths_below_threshold = wavelengths[mask_era5 & mask_cpc]

    # Find the index where the two PSD curves intersect
    idx = np.argmin(np.abs(era5_psd_below_threshold - cpc_psd_below_threshold))

    # Return the wave number associated with the intersection point
    k_star = 2*np.pi / wavelengths_below_threshold[idx]
    return k_star


def get_std_star(data, psd_star):
    _, Ny, Nx = data.shape
    return np.sqrt(Nx*Ny * psd_star)
