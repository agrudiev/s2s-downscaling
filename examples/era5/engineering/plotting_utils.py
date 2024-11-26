import os
import matplotlib.pyplot as plt
import numpy as np
from utils import normalize_data

def plot_timeseries(times, era5_data, cpc_data, mean_title, var_title, figs_dir, mean_filename, var_filename, qm_data=None):
    # Mean
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    era5_ts = np.mean(era5_data, axis=(1,2))
    cpc_ts = np.mean(cpc_data, axis=(1,2))
    plt.plot(times, era5_ts, label='ERA5', color='red')
    plt.plot(times, cpc_ts, label='CombiPrecip', color='blue')
    
    if qm_data is not None:
        qm_ts = np.mean(qm_data, axis=(1,2))
        plt.plot(times, qm_ts, label='QM', color='green')
    
    ax.set_title(mean_title)
    ax.set_ylabel('Precipitation (mm/h)')
    ax.set_xlabel('Time')
    plt.legend()
    plt.savefig(os.path.join(figs_dir, mean_filename))
    
    # Variance
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    era5_ts = np.var(era5_data, axis=(1,2))
    cpc_ts = np.var(cpc_data, axis=(1,2))
    plt.plot(times, era5_ts, label='ERA5', color='red')
    plt.plot(times, cpc_ts, label='CombiPrecip', color='blue')
    
    if qm_data is not None:
        qm_ts = np.var(qm_data, axis=(1,2))
        plt.plot(times, qm_ts, label='QM', color='green')
    
    ax.set_title(var_title)
    ax.set_ylabel('Precipitation (mm/h)²')
    ax.set_xlabel('Time')
    plt.legend()
    plt.savefig(os.path.join(figs_dir, var_filename))


# Here we redefine plot_psds to include the lambda_star and psd_star values.
# Moreover, as we already calculated the PSDs, we can directly plot them!
def plot_psds(era5_wavelengths, era5_psd, cpc_wavelengths, cpc_psd, figs_dir, filename, threshold=1e3, lambda_star=None, psd_star=None):
    plt.figure(figsize=(8, 6))
    
    mask = era5_psd < threshold
    plt.loglog(era5_wavelengths[mask], era5_psd[mask], label="ERA5", color='red')
    mask = cpc_psd < threshold
    plt.loglog(cpc_wavelengths[mask], cpc_psd[mask], label="CombiPrecip", color='blue')
    
    if lambda_star is not None and psd_star is not None:
        plt.axvline(x=lambda_star, linestyle='--', color='grey')
        plt.text(lambda_star*1.01, psd_star*1.3, r'$\lambda^\star$', fontsize=14, color='grey')
    
    plt.xlabel(r"Wavelength $(km$)")
    plt.ylabel("PSD")
    plt.legend(fontsize='large')
    plt.savefig(os.path.join(figs_dir, filename))
    
    
def plot_psds_complete(
    raw_era5_wavelengths,
    raw_era5_psd,
    era5_x25_wavelengths,
    era5_x25_psd,
    lowpass_era5_wavelengths,
    lowpass_era5_psd,
    cpc_wavelengths,
    cpc_psd,
    figs_dir,
    filename,
    cutoff,
    threshold=100,
):
    plt.figure(figsize=(8, 6))
    mask = raw_era5_psd < threshold
    plt.loglog(raw_era5_wavelengths[mask], raw_era5_psd[mask], label="ERA5 (raw)", color='red', linestyle=':')
    mask = era5_x25_psd < threshold
    plt.loglog(era5_x25_wavelengths[mask], era5_x25_psd[mask], label="ERA5 (x25)", color='red', linestyle='--')
    mask = lowpass_era5_psd < threshold
    plt.loglog(lowpass_era5_wavelengths[mask], lowpass_era5_psd[mask], label="ERA5 (low-pass)", color='red', linestyle='-.')
    mask = cpc_psd < threshold
    plt.loglog(cpc_wavelengths[mask], cpc_psd[mask], label="CombiPrecip", color='blue')
    plt.axvline(x=2*np.pi/cutoff, linestyle='--', color='grey')
    plt.text(2*np.pi/cutoff*1.01, 10e-3, 'cutoff', fontsize=14, color='grey')
    plt.xlabel(r"Wavelength $(km$)")
    plt.ylabel("PSD")
    plt.legend(fontsize='large')
    plt.savefig(os.path.join(figs_dir, filename))
    plt.show()
    
    
def plot_parseval(data, data_ps_t, data_ps_2d_t, figs_dir, filename):
    norm_data = normalize_data(data, std_data=0.5)[0]
    norm_data_var_t = np.mean((norm_data - np.mean(norm_data))**2, axis=(1,2))
    ps_var_t = np.sum(data_ps_t, axis=0)
    ps_2d_var_t = np.sum(data_ps_2d_t, axis=(1,2))
    
    print("Variance from normal space: {:.3f}".format(np.mean(norm_data_var_t)))
    print("Variance from wavenumber: {:.3f}".format(np.mean(ps_var_t)))
    print("Variance from wavenumber (2D): {:.3f}".format(np.mean(ps_2d_var_t)))
    
    plt.figure(figsize=(8, 6))
    plt.plot(norm_data_var_t, label="using data", color='blue')
    plt.plot(ps_var_t, label="using ps", color='red')
    plt.plot(ps_2d_var_t, label="using ps", color='green')
    plt.xlabel("Hours since t0")
    plt.ylabel("Variance")
    plt.legend(fontsize='large')
    plt.savefig(os.path.join(figs_dir, filename))
