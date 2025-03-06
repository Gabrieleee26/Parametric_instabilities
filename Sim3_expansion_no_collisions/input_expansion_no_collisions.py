# SRS-SBS instabilities in a 1D plasma slab with collissions

from math import pi, sqrt
from scipy.constants import c, e, m_e, epsilon_0, k as kb, m_u
from scipy.constants import micro, kilo, pico

# LASER WAVELENGTH in SI units [m]
wavelength_SI = 0.351*micro      # wavelength of the laser pulse in m 
omega_r = 2*pi*c/wavelength_SI   # reference frequency  

# CONSTANTS
ev_k = e/kb                                             # eV to Kelvin conversion factor
mass_unit = m_u/m_e                                     # proton-electron mass ratio
n_cr = epsilon_0*m_e*(2*pi*c/wavelength_SI)**2/e**2     # critical density
eV_mc2 = 1/(m_e*c**2/e)                                 # eV to mc^2 conversion factor

# SIMULATION PARAMETERS in NORMAL UNITS 
simulation_time = 8*pico*omega_r    # reference time

dim = 1              # 1D system
length_x = 314*2*pi  # normalised simulation box length

Te = 3.4*kilo   # eV
Ti = 1.0*kilo   # eV

ne_max = 28  # penso sia inutile

# RESOLUTION
n_patches = 256
n_cpp = 8

CFL = 0.95
dx = length_x/n_patches/n_cpp
dt = CFL*dx

# PARTICLES PER CELL
ppc = 1500 #1024 2048

# PARTICLE DENSITY DEFINITION
x_boundary = 14*2*pi             # vacuum boundary (both sides sx dx)
ln = 286*2*pi                    # normalised plasma length

# PARTICLE VELOCITY DEFINITION  
lu = 300*2*pi                    # normalised reference length
us = 0.75*(micro/pico)/c

def n_profile(x):
    if (x > x_boundary) and (x < ln + x_boundary):
        return 0.05+0.23*(x/ln)  # normalised density profile in terms of critical density
    else:
        return 0.0
    
def u_profile(x):
    return us*((x/lu)-1)         # normalised velocity profile

# SIMULATION SET-UP -----------------------------------------------------------------------------

Main(
    geometry = "1Dcartesian",
    interpolation_order = 2,
    
    reference_angular_frequency_SI = omega_r,
    
    timestep = dt,
    simulation_time = simulation_time,
    
    print_every = 500, 
    
    cell_length = [dx],
    grid_length  = [length_x],
    
    number_of_patches = [n_patches],
    
    EM_boundary_conditions = [ ['silver-muller' , 'silver-muller'] ],
)

# SPECIES -----------------------------------------------------------------------------------------

# ELECTRON SPECIES
Species(
    name = 'electron',
    position_initialization = 'random',
    momentum_initialization = 'mj',
    particles_per_cell = ppc,
    mass = 1.0, 
    charge = -1.0,
    number_density = n_profile,
    mean_velocity = [u_profile,0,0],                        # define mean velocity of particles
    temperature = [Te*eV_mc2],                              # homogeneous temperature in units of m_e*c^2
    boundary_conditions = [ ['thermalize', 'thermalize'] ],
    thermal_boundary_temperature = [Te*eV_mc2, Te*eV_mc2],
)

# ION SPECIES (Mixture of Carbon and Hydrogen A_eff=8.73 Z_eff=4.54)
A_eff = 6.5
Z_eff = 3.5

Species(
    name = 'ion',
    position_initialization = 'random',
    momentum_initialization = 'mj',
    particles_per_cell = ppc,
    atomic_number = Z_eff,
    mass = A_eff*mass_unit,
    charge = Z_eff,
    number_density = lambda x: n_profile(x)/Z_eff,              # to maintain charge neutrality divide for Z1
    mean_velocity = [u_profile,0,0],                            # define mean velocity of particles
    temperature = [Ti*eV_mc2],                                  # homogeneous temperature in units of m_e*c^2
    boundary_conditions = [ ['thermalize', 'thermalize'] ],
    thermal_boundary_temperature = [Ti*eV_mc2, Ti*eV_mc2]
)

# LASER WAVE ---------------------------------------------------------------------------------------
t0 = 8*pico*10**(-4)*omega_r             # 8ps = 10^4 laser period expressed in reference units
l0 = 2*pi                                # normalized laser wavelength 
I0 = 6e15*1e4                            # 1e14 W/cm^2 => 1e18 W/m^2
E0 = sqrt(2*I0/(epsilon_0*c))            # V/m
a = e*E0/(m_e*omega_r*c)                 # Normalized vector potential

LaserPlanar1D(
    box_side = 'xmin',
    omega = 1,               # Omega_laser_SI/Omega_r
    a0 = a,
    time_envelope = ttrapezoidal(start=0., plateau = simulation_time-10*t0, slope1=10*t0, slope2=0.0),
)

# CHECKPOINT ----------------------------------------------------------------------------------------
Checkpoints(
    # restart_dir = "dump1",
    # dump_step = 10000,
    dump_minutes = 1380,  # 23h to interrupt the simulation if too long  
    #exit_after_dump = True,
    # keep_n_dumps = 2,
)

# -------------------------------------------------------------------------------------------
#                                       POST-PROCESSING
# -------------------------------------------------------------------------------------------

# SCALAR DIAGNOSTICS ------------------------------------------------------------------------
# https://smileipic.github.io/Smilei/Use/namelist.html#scalar-diagnostics

diag_interval = int(100*t0/dt) # diagnostic each 100 steps

DiagScalar(
    every = diag_interval,
)

# FIELD DIAGNOSTICS -------------------------------------------------------------------------
# https://smileipic.github.io/Smilei/Use/namelist.html#fields-diagnostics

DiagFields(
    every = diag_interval,
    fields = ['Ex', 'Ey', 'Ez','Bx', 'By', 'Bz', 'Rho_ion', 'Rho_electron'],
)

DiagFields(
    every = [ 0,  1e10,  int(simulation_time/dt/20),  2048],
    fields = ['Ey', 'Rho_ion', 'Rho_electron'],
)

# PARTICLE DIAGNOSTICS ----------------------------------------------------------------------
# https://smileipic.github.io/Smilei/Use/namelist.html#particlebinning-diagnostics

for species in ["electron", "ion"]:

    DiagParticleBinning(
        deposited_quantity = "weight",
        every = diag_interval,
        species = [species],
        axes = [
            ["x", 0, Main.grid_length[0], Main.number_of_cells[0]]
        ]
    )

    DiagParticleBinning(                         # divide for weight (Particle Binning 0) to get the kinetic energy
        deposited_quantity = "weight_ekin",
        every = diag_interval,
        species = [species],
        axes = [
            ["x", 0, Main.grid_length[0], Main.number_of_cells[0]]
        ]
    )