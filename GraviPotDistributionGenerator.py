'''
Code whritten and published by Leonid O. Shabaev in January 2025
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import G, parsec

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Font parameters
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 10

# Physical constants
M_Sun = 1.9891e30           # Mass of the Sun (kg)
kpc = 1e3 * parsec          # Kiloparsec

# Pixelation parameters
pixel_axis_size = 25                        # Number of pixels along the axes from zero
upscale_factor = 2 * pixel_axis_size * 1    # Map grid size (change last multiplier to integer)
scale = kpc                                 # Scale
N_star_init = 2e11                          # Initial number of stars

# Density factors: solar mass multiplied by initial number of stars
initial_density_factor = M_Sun * N_star_init

# Bulge parameters
rhob_0 = 1              # Initial bulge density
Rrhob_e = 1 * scale     # Effective bulge radius of the density profile

# Disk parameters
rhod_0 = 0.1            # Initial disk density
Rrhod_e = 5 * scale     # Effective radius of the density profile disk

# Sersic Index and nu parameter
n = 2.2
nu = 1.9992 * n - 0.3271

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Density distribution of bulge profile of the substance
def generation_bulge_density_profile(r):
    bulge_profile = np.exp(-nu * ((r / Rrhob_e) ** (1 / n) - 1))    # Sersic profile in essence
    normalized_bulge_profile = bulge_profile / np.max(bulge_profile)
    output_bulge_profile = rhob_0 * normalized_bulge_profile
    return output_bulge_profile

# Density distribution of disk profile of the substance
def generation_disk_density_profile(r):
    disk_profile = np.exp(-(r / Rrhod_e))
    normalized_disk_profile = disk_profile / np.max(disk_profile)
    output_disk_profile = rhod_0 * normalized_disk_profile
    return output_disk_profile

# Density distribution of total profile of the substance
def generation_total_density_profile(r):
    total_profile = generation_bulge_density_profile(r) + generation_disk_density_profile(r)
    return total_profile

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Generation points on a grid
def generation_points_on_grid(density_profile_func: callable):
    step_size = 2 * pixel_axis_size / upscale_factor
    x = np.arange(-pixel_axis_size + step_size / 2, pixel_axis_size, step_size)
    y = np.arange(-pixel_axis_size + step_size / 2, pixel_axis_size, step_size)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T * scale

    point_profile = density_profile_func(np.linalg.norm(points, axis=1))
    point_mass = initial_density_factor * point_profile / upscale_factor
    
    return points, point_mass, X

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Gravitational potentional
def gravitational_potential(masses, points, target_points):
    potential = np.zeros_like(target_points[:, 0])
    for i in range(len(target_points)):
        r = np.linalg.norm(points - target_points[i], axis=1)
        r[r == 0] = np.inf
        potential[i] = np.sum(-G * masses / r)
    return potential

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Making density distribution profile
def making_fig1():
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1 = plt.gca()

    r = np.linspace(0.01 * kpc, pixel_axis_size * kpc, 100)

    # Calculating profiles
    Bulge_Profile = generation_bulge_density_profile(r)
    Disk_Profile = generation_disk_density_profile(r)
    Total_Profile = generation_total_density_profile(r)

    # Normalization by maximum value (do not change the order of the arrangement for the sake of correct graph operation)
    Bulge_Profile /= np.max(Total_Profile)
    Disk_Profile /=np.max(Total_Profile)
    Total_Profile /= np.max(Total_Profile)

    # Setting a limit for a profile
    disk_intensity_profile_maxlimit = np.max(Total_Profile) * 1.25
    disk_intensity_profile_minlimit = np.min(Total_Profile) / 1.25

    # Design of figure
    ax1.plot(r / kpc, Total_Profile, label='Total', c='black', zorder=3)
    ax1.plot(r / kpc, 
             Bulge_Profile,
             linestyle='--',
             label=f'Bulge n = {n}',
             c='blue',
             zorder=2)
    ax1.plot(r / kpc,
             Disk_Profile,
             linestyle='--',
             label='Disk',
             c='red',
             zorder=1)
    ax1.set_xlabel('Radius (kpc)')
    ax1.set_ylabel(r'Density ($\rho/\rho_0$)')
    ax1.set_title(f'Density Distribution Profile')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_ylim(disk_intensity_profile_minlimit, disk_intensity_profile_maxlimit)
    ax1.grid(True)

    plt.show()
    return fig1, ax1

making_fig1()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Making density distribution on grid
def making_fig2():
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2 = plt.gca()

    # Generate points, masses, and grid
    points, masses, X = generation_points_on_grid(generation_total_density_profile)

    # Compute gravitational potential on the grid
    potential = gravitational_potential(masses, points, points)

    # Reshape potential to match grid dimensions
    potential_grid = potential.reshape(X.shape)

    im = ax2.imshow(potential_grid, extent=(-pixel_axis_size, pixel_axis_size, -pixel_axis_size, pixel_axis_size),
                    origin='upper', cmap='Blues')

    # Add colorbar
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)

    # Add labels and title
    ax2.set_title('Gravitational Potential Distribution Map')
    ax2.set_xlabel('X (kpc)')
    ax2.set_ylabel('Y (kpc)')

    plt.show()
    
    return fig2, ax2

making_fig2()
