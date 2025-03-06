import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.ticker as ticker
import scipy.constants as const
from scipy.constants import c, e, m_p, m_e, epsilon_0, k as kb, m_u
from math import pi, log10
import happi 
import pint  

# ------------------------------------------------------------------------------------------------------------------
# Define the simulation folder
ffolder = '.'                  # Simulation main folder

# Particles 
species = ['electron', 'ion']

# Define useful data
my_dpi = 100

# -------------------------------------------------------------------------------------------------------------------

# Directories
sim_dir = '.'
import os
home = f'{os.getcwd()}'


# FUNCTIONS

def plot_subplot(ax, x_data, y_data_list, labels=None, xlabel='', ylabel='', title=None):
    """
    Simplified function to plot a subplot from SMILEI output lists.
    Assumes both x_data and y_data_list are lists or numpy arrays.
    """
    # Ensure labels are provided, or create empty labels if none
    if labels is None:
        labels = [''] * len(y_data_list)
    
    # Plot each y_data series against x_data
    for y_data, label in zip(y_data_list, labels):
        ax.plot(x_data, y_data, label=label)
    
    # Set labels and title for the subplot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Display legend if any labels are provided
    if any(labels):
        ax.legend()

def plot(xdata, ydata, xlabel='', ylabel='', title=None, legend=None, save=False, plot_name='plot.png', home=home, show=True, grid=False, logx=False, logy=False):
    """
    Simplified function to plot multiple plots from SMILEI output lists.
    Assumes xdata is a list or numpy array and ydata is either a list of lists or a single list or numpy array.
    """
    # Ensure ydata is a list of lists
    if not isinstance(ydata[0], (list, tuple, np.ndarray)):
        ydata = [ydata]
    
    # Ensure legend is a list of the same length as ydata
    if legend is None:
        legend = [''] * len(ydata)
    
    fig, ax = plt.subplots()
    for ydata_series, label in zip(ydata, legend):
        ax.plot(xdata, ydata_series, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(grid)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if any(legend):
        ax.legend()
    if save:
        filename = os.path.join(home, plot_name)
        fig.savefig(filename)
    if show:
        plt.show(block=True)

def save_plot(fig, plot_name, dpi=300):
    """
    Function to quick save the plots in the correct directory.
    Default dpi=300
    """
    fig.tight_layout()
    saving_dir = os.path.join(home, plot_name)
    fig.savefig(saving_dir, dpi=dpi)

# -------------------------------------------------------------------------------------------------------------------

s = happi.Open(sim_dir, pint=True)

# Data 
input_data = s.namelist
scalar     = s.Scalar
field      = s.Field
particle   = s.ParticleBinning

print("\n--------------------------------")
print("Scalar values in the simulation") 
print("--------------------------------\n")
print(scalar)
print("--------------------------------")
print("Fields in the simulation") 
print("--------------------------------\n")
print(field)
print("--------------------------------\n")
print(particle)
print("--------------------------------\n")

# Define units
energy_unit = (lambda dim: "J/m**2" if dim == 1 else "J/m" if dim == 2 else "J")(input_data.dim)
part_dens_unit_scalar = (lambda dim: "1/m**2" if dim == 1 else "1/m" if dim == 2 else None)(input_data.dim)
time_unit = "ps"
temperature_unit = "eV"
particle_density_unit = "1/m**3"
space_unit = "um"
temp_keV = 'keV'
energy_density_unit = 'J/m**3'

# Get data

# Times
times = scalar('time', units=[time_unit]).getData()

# Total, Field, Kinetic, Expected and Balance energy -> Ubal = Utot - Uexp
energy_vars = ['Utot', 'Ukin', 'Uelm', 'Uexp', 'Ubal', 'Ubal_norm', 'Uelm_bnd']

energy_dict = {}
for var in energy_vars:
    energy_dict[var] = np.array(scalar(var, units=[time_unit, energy_unit]).getData())

# Kinetic Energy
Ukin_dict = {}
for sp in species:
    Ukin_dict[sp] = np.array(scalar(f'Ukin_{sp}', units=[time_unit, energy_unit]).getData())

# Particle density
Dens_dict = {}
for sp in species:
    Dens_dict[sp] = np.array(scalar(f'Dens_{sp}', units=[time_unit, part_dens_unit_scalar]).getData())

# Temperatures 
temp_particles = [(2/3)*Ukin_dict[sp] / Dens_dict[sp] / e for sp in species]

# Absorbed Energy Fraction
Fraction_abs_laser_energy = scalar('Ukin/PoyXmin', units = [time_unit, energy_unit]).getData()

# Ion Average Charge
Z_dict = {}
for sp in species[1:]:
    Z_dict[sp] = np.array(scalar(f'Zavg_{sp}').getData())
    

# Scalar Plots
plot_name = "Scalars.png"

# 0,0
data_00 = [energy_dict['Utot'], energy_dict['Uelm'], energy_dict['Ukin']]
labels_00 = ['Utot', 'Uelm', 'Ukin']
y_label_00 = 'Energy [J/m]'
title_00 = 'Energy'
# 0,1
data_01 = [Ukin_dict[sp] for sp in species]
labels_01 = [f'Ukin_{sp}'  for sp in species]
y_label_01 = 'Kinetic Energy [J/m]'
title_01 = 'Particles Kinetic Energy'
# 0,2
data_02 = temp_particles
labels_02 = [f'T_{sp}' for sp in species]
y_label_02 = 'Temperature [eV]'
title_02 = 'Total temperature for each particle'
# 1,0
data_10 = [energy_dict['Uexp'], energy_dict['Utot']]
labels_10 = ['Uexp', 'Utot']
y_label_10 = 'Energy [J/m]'
title_10 = 'Expected and Total energy'
#1,1
data_11 = [Z_dict[sp] for sp in species[1:]]
labels_11 = [f'Zavg_{sp}' for sp in species[1:]]
y_label_11 = 'Average Charge [e]'
title_11 = 'Ions Average Charge in the sistem'
#1,2
data_12 = [Fraction_abs_laser_energy]
labels_12 = ['Ukin/PoyXmin']
y_label_12 = 'Fraction'
title_12 = 'Absorbed laser energy fraction'

# Create figure and axes
fig, ax = plt.subplots(2, 3, figsize=(12, 8), dpi=my_dpi, sharex=True)
#0,2
plot_subplot(ax[0][0], times, data_00, labels_00, 'Time [ps]', y_label_00, title_00)
plot_subplot(ax[0][1], times, data_01, labels_01, 'Time [ps]', y_label_01, title_01)
plot_subplot(ax[0][2], times, data_02, labels_02, 'Time [ps]', y_label_02, title_02)
plot_subplot(ax[1][0], times, data_10, labels_10, 'Time [ps]', y_label_10, title_10)
plot_subplot(ax[1][1], times, data_11, labels_11, 'Time [ps]', y_label_11, title_11)
plot_subplot(ax[1][2], times, data_12, labels_12, 'Time [ps]', y_label_12, title_12)
save_plot(fig, plot_name)
plt.show(block=True)










# Intensità funzione del tempo e spazio

# LASER WAVELENGTH in SI units [m]
wavelength_SI = 0.351 * micro  # wavelength of the laser pulse in m 
omega_r = 2 * pi * c / wavelength_SI  # reference frequency  


Ex = np.array(s.Field(0, 'Ex').getData()) * m_e * c * omega_r / e
times = scalar('time', units=["ps"]).getData()
space_extent = s.namelist.Main.grid_length[0] * 1e6 * c / omega_r  # Converte la lunghezza in micrometri
space_points = Ex.shape[1]
space = np.linspace(0, space_extent, space_points)  # Genera l'asse spaziale in μm

# Definisce una soglia inferiore modificabile
threshold = 1e-2

# Calcola il modulo quadro di Ex e lo converte in scala logaritmica con soglia inferiore
Ex_mod2 = np.abs(Ex) ** 2/ (6e15*1e4)
Ex_mod2[Ex_mod2 < threshold] = threshold  # Applica la soglia
Ex_mod2_log = np.log10(Ex_mod2)  # Converti in scala logaritmica

# Crea il grafico
plt.figure(figsize=(10, 6))
im = plt.imshow(Ex_mod2_log, aspect='auto', cmap='viridis', origin='lower', 
                extent=[space[0], space[-1], times[0], times[-1]])

# Formatta la colorbar per mostrare i valori come potenze di 10
cbar = plt.colorbar(im)
cbar.set_label(r'$|E_x|² [I_0]$')  # Etichetta della colorbar

def log_tick_formatter(val, pos):
    return r"$10^{%d}$" % val

cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))

# Etichette e titolo
plt.xlabel('Posizione [μm]')  # Posizione in micrometri
plt.ylabel('Tempo [ps]')
plt.title('Modulo quadro di $E_x$ in scala logaritmica')
plt.grid(True)

# Salva il grafico come file PNG
plot_name = "Ex_mod_squared_log.png"
plt.savefig(plot_name, dpi=300, bbox_inches='tight')









# Average Electron Temperature evolution --------------------------------------------------------------
energy_ele_ev_avgx = np.array(particle('#1', units = [energy_density_unit, space_unit, time_unit], average = {'x': 'all'}).getData())
dens_ele_avgx      = np.array(particle('#0', units = [particle_density_unit, space_unit, time_unit], average = {'x': 'all'}).getData())
temp_ele_avgx      = (2/3)*energy_ele_ev_avgx / dens_ele_avgx / e / const.kilo  # temperature in keV
plot(times, temp_ele_avgx, xlabel='Time [ps]', ylabel='Temperature [keV]', title='Average Electron Temperature evolution', save=True, plot_name='temp_ele_avgx.png', grid=True)

# Average Ion Temperature evolution
energy_ion_ev_avgx = np.array(particle('#3', units = [energy_density_unit, space_unit, time_unit], average = {'x': 'all'}).getData())
dens_ion_avgx      = np.array(particle('#2', units = [particle_density_unit, space_unit, time_unit], average = {'x': 'all'}).getData())
temp_ion_avgx      = (2/3)*energy_ion_ev_avgx / dens_ion_avgx / e / const.kilo  # temperature in keV
plot(times, temp_ion_avgx, xlabel='Time [ps]', ylabel='Temperature [keV]', title='Average Ion Temperature evolution', save=True, plot_name='temp_ion_avgx.png', grid=True)

# Temperature comparison 
legend = ['Electron Temperature', 'Ion Temperature']
plot(times, [temp_ele_avgx, temp_ion_avgx], xlabel='Time [ps]', ylabel='Temperature [keV]', title='Temperature comparison', legend = legend, save=True, plot_name='temp_comparison.png', grid=True)

# Relative spectral energy density -----------------------------------------------------------------------------------
t = field(1,'Ey').getTimesteps()
chunks = np.flatnonzero(np.diff(t)>1)

start = 0
fft_data_list = []
for chunk in chunks:
    data = np.array(field(1, 'Ey', timesteps = [t[start],t[chunk]]).getData())
    fft_data = np.fft.fftshift(np.fft.fft2(data))
    magnitude = np.abs(fft_data)
    fft_data_list.append(np.log(magnitude))  
    plt.imshow(np.log(magnitude), extent=[2549.75, 2550.25, 1014.25, 1013.75])
    plt.title(f"Data from {t[start]} to {t[chunk]}")
    plt.show(block=True)
    start = chunk+1

fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.imshow(fft_data_list[i],extent=[2549.75, 2550.25, 1014.25, 1013.75])
    ax.set_title(f"Segmento da {t[0]} a {t[-1]}")

ani = animation.FuncAnimation(fig, animate, frames=len(fft_data_list), interval=1000)
ani.save('spectral_energy_density.gif', writer='imagemagick')
plt.show()

# Spectral Energy SI-units ----------------------------------------------------------------------------------------------
start = 0
fft_data_list_SI = []
dt = input_data.dt
dx = input_data.dx
omega_laser = input_data.omega_r
for chunk in chunks:
    data = np.array(field(1, 'Ey', timesteps=[t[start], t[chunk]]).getData())
    
    # Get the time and space steps
    num_timesteps, num_spatial_points = data.shape
    Ndt = num_timesteps*dt
    space_length = num_spatial_points * dx
    
    # Transform the data
    fft_data = np.fft.fftshift(np.fft.fft2(data))
    magnitude = np.abs(fft_data)
    
    # Axes in SI units
    freqs_normalized = np.fft.fftshift(np.fft.fftfreq(num_timesteps, d=dt))*2*pi
    k_values = np.fft.fftshift(np.fft.fftfreq(num_spatial_points, d=dx))*2*pi
    
    # Plot 
    plt.imshow(np.log(magnitude), extent=[ k_values.min(), k_values.max(), freqs_normalized.min(), freqs_normalized.max()], aspect='auto', origin='lower')
    plt.colorbar(label='Log Magnitude')
    plt.ylim(-1.2,1.2)
    plt.xlim(-1.2,1.2)
    plt.ylabel(r'$\omega / \omega_0$')
    plt.xlabel(r'$k$ (1/m)')
    plt.title("2D spectrum of $E_y$ in function of $\omega / \omega_0$ and $k$ (1/m)")
    # Add lines to divide the plot
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  # Horizontal line at k = 0
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)  # Vertical line at omega = 0
    plt.show(block=True)
    
    fft_data_list_SI.append((np.log(magnitude), freqs_normalized, k_values))
    
    start = chunk + 1

if not fft_data_list_SI:
    raise ValueError("La lista fft_data_list_SI è vuota. Assicurati che i dati vengano popolati correttamente.")

fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    magnitude, freqs_normalized, k_values = fft_data_list_SI[i]
    im = ax.imshow(magnitude, extent=[k_values.min(), k_values.max(), freqs_normalized.min(), freqs_normalized.max()], aspect='auto', origin='lower')
    ax.set_ylim(-1.2,1.2)
    ax.set_xlim(-1.2,1.2)
    ax.set_ylabel(r'$\omega / \omega_0$')
    ax.set_xlabel(r'$k$ (1/m)')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.3)  # Horizontal line at k = 0
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.3)  # Vertical line at omega = 0
    #plt.colorbar(im, ax=ax, label='Log Magnitude')

ani = animation.FuncAnimation(fig, animate, frames=len(fft_data_list_SI), interval=1000)

# Save the animation using PillowWriter
writer = PillowWriter(fps=1)
ani.save('spectral_energy_density_SI.gif', writer=writer)
plt.show()

# 1D spectrum -----------------------------------------------------------------------------------------------------------
start = 0
dt = input_data.dt
dx = input_data.dx
omega_laser = input_data.omega_r
fft_data_list_1D = []

for chunk in chunks:
    data = np.array(field(1, 'Ey', timesteps=[t[start], t[chunk]]).getData())
    
    # Get the time and space steps
    num_timesteps, num_spatial_points = data.shape
    Ndt = num_timesteps*dt
    space_length = num_spatial_points * dx
    
    # Transform the data
    fft_data = np.fft.fftshift(np.fft.fft2(data))
    magnitude = np.abs(fft_data)
    
    # Axes in SI units
    freqs_normalized = np.fft.fftshift(np.fft.fftfreq(num_timesteps, d=dt))*2*pi
    #freqs_normalized = freqs/omega_laser
    
    magnitude_1D = np.sum(magnitude, axis=1)
    positive_freqs = freqs_normalized[freqs_normalized > 0]
    positive_magnitude = magnitude_1D[freqs_normalized > 0]  # Solo le frequenze positive
    
    plt.plot(positive_freqs, positive_magnitude, color='blue', label='1D Spectral Density')
    plt.xlim(0,1.1)
    plt.xlabel(r'$\omega / \omega_0$')
    plt.ylabel('Spectral Density')
    plt.title(f'1D Spectrum for Segment {start} to {chunk}')
    plt.yscale('log')  # Scala logaritmica per la densità spettrale
    plt.legend()
    plt.show(block=True)
    
    fft_data_list_1D.append((positive_freqs, positive_magnitude))
    
    start = chunk + 1
    
    
fig, ax = plt.subplots() # Crea l'animazione per lo spettro 1D

def animate_1D(i):
    ax.clear()
    positive_freqs, positive_magnitude = fft_data_list_1D[i]
    ax.plot(positive_freqs, positive_magnitude, color='blue', label='1D Spectral Density')
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$\omega / \omega_0$')
    ax.set_ylabel('Spectral Density')
    ax.set_title(f'1D Spectrum for Segment {i}')
    ax.set_yscale('log')  # Scala logaritmica per la densità spettrale
    ax.legend()

ani_1D = animation.FuncAnimation(fig, animate_1D, frames=len(fft_data_list_1D), interval=1000)

# Salva l'animazione utilizzando PillowWriter
writer_1D = PillowWriter(fps=1)
ani_1D.save('spectral_energy_density_1D.gif', writer=writer_1D)
plt.show()

# Electrons oscillations ------------------------------------------------------------------------------------------------
start = 0
fft_data_list_ele = []
for chunk in chunks:
    data = np.array(field(1, 'Rho_electron', timesteps=[t[start], t[chunk]]).getData())
    
    # data transformation
    fft_data = np.fft.fftshift(np.fft.fft2(data))
    magnitude = np.abs(fft_data)
    
    # data normalization
    freqs_normalized = np.fft.fftshift(np.fft.fftfreq(num_timesteps, d=dt))*2*pi
    k_values = np.fft.fftshift(np.fft.fftfreq(num_spatial_points, d=dx))*2*pi
    
    plt.imshow(np.log(magnitude), extent=[ k_values.min(), k_values.max(), freqs_normalized.min(), freqs_normalized.max()], aspect='auto', origin='lower')
    plt.colorbar(label='Log Magnitude')  
    #plt.ylim(-1.2,1.2)
    #plt.xlim(-1.2,1.2)
    plt.ylabel(r'$\omega / \omega_0$')
    plt.xlabel(r'$k$ (1/m)')
    plt.title("2D electrons density spectrum in function of $\omega / \omega_0$ and $k$ (1/m)")
    # Add lines to divide the plot
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  # Horizontal line at k = 0
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)  # Vertical line at omega = 0
    plt.show(block=True)
    
    fft_data_list_ele.append(np.log(magnitude))
 
    start = chunk + 1


fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    magnitude, freqs_normalized, k_values = fft_data_list_ele[i]
    im = ax.imshow(magnitude, extent=[k_values.min(), k_values.max(), freqs_normalized.min(), freqs_normalized.max()], aspect='auto', origin='lower')
    ax.set_ylabel(r'$\omega / \omega_0$')
    ax.set_xlabel(r'$k$ (1/m)')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  # Linea orizzontale a k = 0
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)  # Linea verticale a omega = 0
    plt.colorbar(im, ax=ax, label='Log Magnitude')

ani = animation.FuncAnimation(fig, animate, frames=len(fft_data_list_ele), interval=1000)

# Salva l'animazione utilizzando PillowWriter
writer = PillowWriter(fps=1)
ani.save('electrons_spectrum.gif', writer=writer)
plt.show()

# Ions oscillations ------------------------------------------------------------------------------------------------
start = 0
fft_data_list_ion = []
for chunk in chunks:
    data = np.array(field(1, 'Rho_ion', timesteps=[t[start], t[chunk]]).getData())
    
    # data transformation
    fft_data = np.fft.fftshift(np.fft.fft2(data))
    magnitude = np.abs(fft_data)
    
    # data normalization
    freqs_normalized = np.fft.fftshift(np.fft.fftfreq(num_timesteps, d=dt))*2*pi
    k_values = np.fft.fftshift(np.fft.fftfreq(num_spatial_points, d=dx))*2*pi
    
    plt.imshow(np.log(magnitude), extent=[ k_values.min(), k_values.max(), freqs_normalized.min(), freqs_normalized.max()], aspect='auto', origin='lower')
    plt.colorbar(label='Log Magnitude')  
    #plt.ylim(-1.2,1.2)
    #plt.xlim(-1.2,1.2)
    plt.ylabel(r'$\omega / \omega_0$')
    plt.xlabel(r'$k$ (1/m)')
    plt.title("2D ions density spectrum in function of $\omega / \omega_0$ and $k$ (1/m)")
    # Add lines to divide the plot
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  # Horizontal line at k = 0
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)  # Vertical line at omega = 0
    plt.show(block=True)
    
    fft_data_list_ion.append(np.log(magnitude))
 
    start = chunk + 1


fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    magnitude, freqs_normalized, k_values = fft_data_list_ion[i]
    im = ax.imshow(magnitude, extent=[k_values.min(), k_values.max(), freqs_normalized.min(), freqs_normalized.max()], aspect='auto', origin='lower')
    ax.set_ylabel(r'$\omega / \omega_0$')
    ax.set_xlabel(r'$k$ (1/m)')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  # Linea orizzontale a k = 0
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)  # Linea verticale a omega = 0
    plt.colorbar(im, ax=ax, label='Log Magnitude')

ani = animation.FuncAnimation(fig, animate, frames=len(fft_data_list_ion), interval=1000)

# Salva l'animazione utilizzando PillowWriter
writer = PillowWriter(fps=1)
ani.save('ions_spectrum.gif', writer=writer)
plt.show()
# --------------------------------------------------------------------------------------------------------------------------
