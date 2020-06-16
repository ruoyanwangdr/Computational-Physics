__author__ = "Robin Mentel, Ruoyan Wang, Christopher Seay"

"""

simulates the molecular dynamics of argon atoms in a pseudo-infinite 3D box.
handles liquid, solid, and gaseous states. simulation is unitless and mass is
constant, so it is left out of equations.

"""

###NOTES: replace long variables in functions with semi-shorthand and then explain them
########  write force function, next_force function, 
########  x - position, v - velocity, h - dt, f - force, L - box_size, etc 
########  force: maybe, it's just the distance between a particle and its closest neighbors


### to work on: redoing initial velocity, getting rid of distances and other verbose variables,
### reimplement mic before calculating forces, calculating/storing total energies of system rather
### than per particle.
### plotting, correlation function

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

def scalar_distance(r, axis=0):
    """ magnitude of input vector r."""

    return np.sqrt(np.sum(r**2, axis)) 
    
def rand_norm_unit_vector():
    """ returns a normalized random unit vector."""

    vec = np.random.uniform(-1.0, 1.0, (3))
    return vec / scalar_distance(vec)

def lennard_jones_potential(r):
    """ unitless lennard-jones potential describing 
    potential of a given system."""

    return 4*(np.power(r, -12) - np.power(r, -6))

def LJP_dUdr(r):
    """ unitless derivaive of lennard-jones potential
    with respect to input vector r."""
    
    return -48 / np.power(r, 13) + 24 / np.power(r, 7)

def acceleration(r, norm_r):
    """ calculates acceleration (force) of particle from
    a distance r apart."""

    return -((LJP_dUdr(norm_r) * r / norm_r))

def current_velocity(x_new, x_prev, dt):
    """ current velocity of a particle from next
    position at timestep t+dt."""

    """
    parameters
    ----------
    x_new : float
        new x-position of particle
    x_prev : float
        previous x-position of particle
    dt : float
        current timestep t + dt

    returns
    ----------
    velocity at current timestep

    """
    
    return (x_new - x_prev) / 2*dt

def next_position(x, v, force, dt, L):
    """ compute next position of particle in space
    described by box size L."""

    """
    parameters
    ----------
    x : array
        current position vector of particle
    v : array
        current velocity vector of particle
    force : float
        current force vector felt by the particle
    dt : float
        current timestep
    L : float
        box size

    returns
    ----------
    next position vector for
    particle in a given direction 

    """

    x_next = x + dt*v + ((dt**2)/2 * force)
    
    return minimal_image_convention(x_next, L, dt)


def next_velocity(v, force, next_force, dt):
    """ calculates next velocity component for a particle."""

    """
    parameters
    ----------
    v : float
        velocity component of particle
    force : float
        current force felt by particle
    next_force : float
        force felt by particle at future timestep
    dt : float
        current timestep

    returns
    ----------
    next velocity component for particle

    """

    return v + dt/2.0 * (force + next_force)

def find_neighbors(dt, state, particle):
    
    """ finds neighbors of a given particle."""

    """
    parameters
    ----------
    dt : float
        timestep of simulation
    state : array
        current physical characteristics of simulation
    particle : int (array index) 
        atom index from state array

    returns
    ----------
    particle_state of neighbors for
    a given particle

    """
    
    xyz = state[dt, particle, :3]
    atom_index_list = np.arange(0, state.shape[1])
    atom_index_list = np.delete(atom_index_list, particle)
    neighbors = state[dt, atom_index_list, :3]

    return neighbors


def particle_force(dt, state, L, particle):

    """ finds the acceleration on a single body by LJ-Potential from others."""

    """
    parameters
    ----------
    dt : float
        timestep of simulation
    state : array
        current physical characteristics of simulation
    L : float
        box size
    particle : int (array index) 
        atom index from state array

    returns
    ----------
    specific force (essentially acceleration) felt
    by a particle. also returns potential and normalized
    distance between a particle and its nearest neighbor

    """
    
    xyz = state[dt, particle, :3]
    neighbors = find_neighbors(dt, state, particle)
    # don't understand this yet
    r = (xyz - neighbors + L / 2) % L - L / 2
    norm_r = scalar_distance(r,axis=1)
    new_r_norm = np.reshape(np.repeat(norm_r, 3), (len(norm_r), 3))
    if dt == 0:
        print(r)
        print(norm_r)
        print(new_r_norm)

    force = np.sum(acceleration(r, new_r_norm))
    potential = np.sum(lennard_jones_potential(norm_r))

    return force, potential, norm_r


def minimal_image_convention(dt, state, L):

    """ minimal image convention (applies boundary condition) of designing 
        a pseudo-infinite box."""

    """
    parameters
    ----------
    dt : float
        current timestep
    state : array 
        current physical state of particles
    L : float
        box size (volume)

    returns
    ----------
    new position for atom if beyond boundary

    """

    for i in range(3):
        if state > L / 2:
            state = state - L
        elif state < -L / 2:
            state = state + L
            
    
    return state

def lambda_scale_factor(dt, T, num_atoms, state):
    """calculates lambda scale factor."""

    """
    parameters
    ----------
    dt : float
        timestep
    T : float
        temperature
    num_atoms : float
        total number of particles
    state : array
        data on all particles

    returns
    ----------
    lambda scale factor for current state of simulation
    """

    return np.sqrt(((num_atoms - 1) * 3 * T) / 
            (np.sum(scalar_distance(state[dt, :, 3:], axis=1) ** 2)))

def main():

    ### NOTE: parameters of simulation, including initial physical conditions
    
    # kinetic energy lambda function
    KE = lambda v: 0.5*v**2

    # set phase from user input and guarantee reproducible results with seed
    np.random.seed(0)
    phase = input("phase state of argon to be simulated (gas, liquid, solid):")
    if phase != "solid" or phase != "liquid" or phase != "gas":
        phase = "gas"
    # set temperature and density
    if phase == "gas":
        temperature = 3.0
        density = 0.3
    elif phase == "solid":
        temperature = 1.0
        density = 0.8
    else:
        temperature = 0.5
        density = 1.2
    
    ### initial parameters start ###
    num_unit_cell_per_dim = 3
    num_unit_cells = 27         # 3**3 unit cells per dimension (3D)
    num_atom_per_unit_cell = 4  # number of atoms per unit cell
    num_atoms = num_unit_cells * num_atom_per_unit_cell # number of atoms
    num_dimensions = 3          # number of dimensions
    dt = 1e-4                   # timestep
    t_end = 0.4                 # end simulation time
    num_steps = int(t_end/dt)   # number of time steps
    box_size = np.power((num_atoms / density),1/3) # size of simulation box
    unit_cell_size = box_size / 3 # size of a unit cell per dimension

    # initialize arrays to store position, velocity for each particle;
    # energies kinetic, potential, total;
    # lambda factors; forces; distances
    # change to total energies again?
    particle_state = np.zeros((num_steps, num_atoms, 2*num_dimensions))
    kinetic_energy = np.zeros((num_steps, num_atoms))
    potential_energy = np.zeros((num_steps, num_atoms))
    total_energy = np.zeros((num_steps, num_atoms))
    lambdas = np.zeros(num_steps)
    # forces = np.zeros((1, 3))
    # distances = np.zeros(num_atoms - 1)
    # distances = np.zeros((num_steps, num_atoms, num_atoms - 1))

    # velocities given by a gaussian distribution
    # rescaling parameters for lambda factors also defined
    mu = 1
    sigma = np.sqrt(temperature)
    rescaling_interval = 10
    rescaling_time = int(rescaling_interval / dt)
    rescaling_end = int(2 * rescaling_interval / dt)
    velocity_distribution = np.random.normal(mu, sigma, num_atoms)
    ### initial parameters end ###

    ### initial conditions start ###
    # positions of first unit cell
    particle_state[0, 0, :3] = np.array([0, 0, 0])
    particle_state[0, 1, :3] = np.array([0,
                                        unit_cell_size / 2, 
                                        unit_cell_size / 2])
    particle_state[0, 2, :3] = np.array([unit_cell_size / 2,
                                        unit_cell_size / 2,
                                        0])
    particle_state[0, 3, :3] = np.array([unit_cell_size / 2,
                                        0,
                                        unit_cell_size / 2])


    # setting initial coordinates and velocities ona an fcc lattice
    # keep track of particle index
    # line 163 from robin's code for initial velocities/get_initial_velocities
    particle = 0
    for x in range(num_unit_cell_per_dim):
        for y in range(num_unit_cell_per_dim):
            for z in range(num_unit_cell_per_dim):
                for i in range(num_atom_per_unit_cell):
                    particle_state[0, particle, :3] = particle_state[particle % 4, 0, :3] \
                            + np.array([x, y, z]) * unit_cell_size
                    particle_state[0, particle, 3:] = velocity_distribution[particle] * \
                            rand_norm_unit_vector()
                    particle += 1

    # inital lambda and energy values
    lambdas[0] = lambda_scale_factor(0, temperature, num_atoms, particle_state)
    for i in range(num_atoms):
        F, P, norm_r = particle_force(0, particle_state, box_size, i)
        kinetic_energy[0, i] = scalar_distance(KE(particle_state[0, i, 3:]))
        potential_energy[0, i] = 0.5 * P
        total_energy[0, i] = kinetic_energy[0, i] + potential_energy[0, i]

    ### initial conditions end ###
    
    # note: 'd' before a variable denotes timestep (ie, small) change,
    # so, dPE, dKE are small changes in potential_energy and kinetic_energy
   
    ### run simulation
    for t in range(1, num_steps):
        # rescale when needed
        if t >= rescaling_time and t < rescaling_end \
                and (t - rescaling_time) % rescaling_interval == 0:

            lambda_factor = np.average(lambdas[t-1 - rescaling_interval:t-1])
            particle_state[t-1, :, 3:] = \
                    lambda_factor * particle_state[t-1, :, 3:]

        for i in range(num_atoms):
            # find F, positions, velocities, energies
            # wanna do mic before force calc, so do temp_particle_state stuff here from robin's code
            forces, P, r = particle_force(t-1, particle_state, box_size, i)
            particle_state[t, i, :3] = next_position(particle_state[t-1, i, :3],
                                                    particle_state[t-1, i, 3:],
                                                    forces,
                                                    dt,
                                                    box_size)
            F_next, P_next, distances[t, i, :] = particle_force(t,
                                                                particle_state,
                                                                box_size,
                                                                i)
            next_v = next_velocity(particle_state[t-1, i, 3:],
                                    forces,
                                    F_next,
                                    dt)
            particle_state[t, i, 3:] = next_v
            kinetic_energy[t, i] = scalar_distance(KE(next_v))
            potential_energy[t, i] = 0.5 * P_next
            total_energy[t, i] = kinetic_energy[t, i] + potential_energy[t, i]
            lambdas[t] = lambda_scale_factor(t,
                                            temperature,
                                            num_atoms,
                                            particle_state)
                                                                

if __name__ == '__main__':
    sys.exit(main())


