import happi
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.constants import micro, kilo, pico
from scipy.constants import c, e, m_e, epsilon_0, k as kb, m_u

def save_plot(fig, plot_name, dpi=300):
    """
    Function to quick save the plots in the correct directory.
    Default dpi=300
    """
    fig.tight_layout()
    saving_dir = os.path.join(home, plot_name)
    fig.savefig(saving_dir, dpi=dpi)

    

sim = happi.Open(".")
scalar     = sim.Scalar
home = f'{os.getcwd()}'

mass_ion = np.double(sim.namelist.Species["ion"].mass)

wavelength = 0.351*micro  # meters
time_unit = "ps"
times = scalar('time', units=[time_unit]).getData()

e_vx_mean = np.zeros(len(times))
i_vx_mean = np.zeros(len(times))
Ti = np.zeros(len(times))
Te = np.zeros(len(times))

electrons0 = sim.ParticleBinning(0, sum={"x": "all"}).get()
ions = sim.ParticleBinning(3, sum={"x": "all"}).get()

fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(1, 1, 1)

ev_k = e/kb  
for i, t in enumerate(times):
    vx_e = electrons0["vx"]*c
    A_e = electrons0["data"][i]
    e_vx_mean[i] = (A_e * vx_e).sum() / A_e.sum()
    Te[i] = (A_e * (vx_e - e_vx_mean[i]) ** 2).sum() / A_e.sum()*m_e/kb/ev_k/kilo

    vx_i = ions["vx"]*c
    A_i = ions["data"][i]
    i_vx_mean[i] = (A_i * vx_i).sum() / A_i.sum()
    Ti[i] = (A_i * (vx_i - i_vx_mean[i]) ** 2).sum() / A_i.sum() * mass_ion*m_e/kb/ev_k/kilo


ax.plot(times, Te, 'b')
ax.plot(times, Ti, 'r')
ax.set_xlabel("Time [ps]")
ax.set_ylabel("Temperature [keV]")
fig.canvas.draw()
ax.legend(['electrons', 'ions'], loc='upper left')
plot_name = "Temperature.png"
save_plot(fig, plot_name)
plt.show()




A_i = ions["data"][1]
vx_i = ions["vx"]*c
i_vx_mean[1] = (A_i * vx_i).sum() / A_i.sum()

A_e = electrons0["data"][1]
vx_e = electrons0["vx"]*c
e_vx_mean[1] = (A_e * vx_e).sum() / A_e.sum()

T_ele = (A_e * (e_vx_mean[1]) ** 2).sum() / A_e.sum() * m_e/kb/ev_k
T_ion = (A_i * (i_vx_mean[1]) ** 2).sum() / A_i.sum() * mass_ion*m_e/kb/ev_k
print(T_ele, T_ion)

# Il rapporto e_vx_mean[1]/ i_vx_mean[1] è circa uno che ha senso perchè la densità è la stessa

# Second part --------------------------------------------------------------------------------------------
# Plot scalari in cui togliamo il valore cinetico del centro di massa all'istante 0

A_i = ions["data"][1]
vx_i = ions["vx"]*c
i_vx_mean[1] = (A_i * vx_i).sum() / A_i.sum()

A_e = electrons0["data"][1]
vx_e = electrons0["vx"]*c
e_vx_mean[1] = (A_e * vx_e).sum() / A_e.sum()

T_ele = (A_e * (e_vx_mean[1]) ** 2).sum() / A_e.sum() * m_e/kb/ev_k
T_ion = (A_i * (i_vx_mean[1]) ** 2).sum() / A_i.sum() * mass_ion*m_e/kb/ev_k
print(T_ele, T_ion)

print((A_e * (e_vx_mean[1]) ** 2).sum() / A_e.sum())
print((A_i * (i_vx_mean[1]) ** 2).sum() / A_i.sum())


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.ticker as ticker
from scipy.constants import micro, kilo, pico
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
# Assuming T_ele and T_ion are already defined in the console
# They should be available here as variables

# Temperatures corrected for T_ele and T_ion
temp_particles = [
    ((2/3) * Ukin_dict[sp] / Dens_dict[sp] / e) - (T_ele if sp == 'electron' else T_ion) 
    for sp in species
]

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



