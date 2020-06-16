__author__ = "Robin Mentel, Ruoyan Wang, Christopher Seay"

"""

simulates the molecular dynamics of argon atoms in a pseudo-infinite 3D box.
handles liquid, solid, and gaseous states. simulation is unitless and mass is
constant, so it is left out of equations.

"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


def Distance_Two_Bodies(particle_1, particle_2):
    """ returns distance between two bides. """

    pos_1 = particle_1[:3]
    pos_2 = particle_2[:3]
    distance = np.sqrt(np.sum((pos_1 - Pos2)**2))
    
    return distance

def Distances_Many_Bodies(body1, others):
    """ returns the distances between one body and a set of other bodies. """

    diff = body1[:3] - others[:, :3]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    return dists

def rand_norm_unit_vector(num_atoms):
    """ returns random unit vectors for all particles to get direction randomised velocites. """

    vec = np.random.uniform(-1.0, 1.0, (num_atoms, 3))
    norm_vec = np.sqrt(np.sum(vec**2))

    return vec / norm_vec

def lennard_jones_potential(r):
    """ returns value of lennard-jones potential at a given distance r. """
    
    LJPot =  4.*(np.power(r, -12) - np.power(r, -6))

    return LJPot

def current_velocity(x_new, x_prev, h):
    """ returns current velocity of a particle from next
    position at timestep. """

    """
    parameters
    ----------
    x_new : array
        new x-position of particle
    x_prev : array
        previous x-position of particle
    h : float
        simulation timestep

    """
    vel = (x_new - x_prev) / 2*h
    
    return vel

def next_position(pos, vel, acc, h):
    """ returns next position of particle."""

    """
    parameters
    ----------
    pos : array
        current position vector of particle
    vel : array
        current velocity vector of particle
    acc : float
        current acceleration acting on the particle
    h : float
        simulation timestep

    returns
    ----------
    position at simulation timestep

    """

    x_next = pos + vel*h + acc*(0.5*h**2)
    
    return x_next


def next_velocity(vel, acc, next_acc, h):
    """ returns velocity at next time stamp for a particle. """

    """
    parameters
    ----------
    vel : array
        velocity component of particle
    acc : array
        current force felt by particle
    next_acc : array
        accelaration acting on particle at next time stamp
    h : float
        simulation timestep

    returns
    ----------
    velocity at current timestep

    """

    return vel + 0.5*h*(acc + next_acc)

def lambda_scale_factor(state, T, num_atoms):
    """ returns lambda scale factor for a given particle state. """

    """
    parameters
    ----------
    state : array
        current state of the simulation
    T : float
        temperature
    num_atoms : int
        total number of particles

    returns
    ----------
    lambda scale factor

    """
    velocities = state[:, 3:]
    lambda_factor = np.sqrt( (num_atoms - 1) * 3 * T / np.sum(velocities**2) )
    
    return lambda_factor

def pressure_calc(dists, dUdr, T, rho, num_atoms):
    """ returns the pressure as given by Verlet et al. 1961.
    
    parameters
    ----------
    dists : array (108, 107)
        distances between all particle
    dUdr : array (108, 107)
        value of the derivative of the LJ-potential between all particles
    T : float
        temperature
    rho : float
        density
    num_atoms : int
        total number of particles

    returns
    ----------
    pressure at the final state of particle simulation

    """
    # remove double counts
    dists = np.triu(dists, 0)
    
    pressure = T*rho * (1 - 1/(3*num_atoms*T)*0.5*np.sum(dists*dUdr) )
    
    return pressure
    
def particle_force(state, particle):

    """ returns the acceleration acting on a single body due to LJ-Potential from the other particles,
    and its potential energy."""

    """
    parameters
    ----------
    state : array
        state of simulation at current time stamp
    particle : int (array index) 
        state array index for particle

    returns
    ----------
    force acting on a particle. effectively
    the acceleration of a particle (mass unity
    in simulation)

    """
    
    pos_1 = state[particle, :3]
    
    for i, other_body in enumerate(state):
        if i != particle:
            
            # calc distances and else with new 
            pos_2 = other_body[:3]
            Dpos = pos_2 - pos_1
            r = Distance_Two_Bodies(pos_1, pos_2)
            acc = 0
            sum_Epot = 0
            
            # try gravity interaction to get it work, then switch to LJ
            #scalar_acc = np.power(r, -3.)
            # split up vec(acc) in a scalar acceleration = abs(acc)/r, and a vector vec(e_acc) = vec(r)
            scalar_acc = LJP_dUdr(r) / r
            
            sum_acc = np.add(acc, scalar_acc*Dpos)
            
            # calculate potential energy in that potential
            dU = lennard_jones_potential(r)
            sum_Epot += dU
            
    return acc, sum_Epot

def LJP_dUdr(r):
    """ unitless derivaive of lennard-jones potential
    with respect to input vector r."""
    
    dUdr = -48 / np.power(r, 13) + 24 / np.power(r, 7)
    
    return dUdr

def plotting_state(state, box_size):
    margin = 0.3
    plt.figure(figsize=(12, 8))
    for p in np.rollaxis(state, 1):
        plt.scatter(p[0, 0], p[0, 1], s=100, c='k', zorder=-5, marker='x')
        plt.scatter(p[:, 0], p[:, 1], s=1, c='r')
        plt.scatter(p[-1, 0], p[-1, 1], s=25, c='b', zorder=5)

    plt.xlim(-margin, box_size+margin)
    plt.ylim(-margin, box_size+margin)
    plt.xlabel("X Coordinate", fontsize=16)
    plt.ylabel("Y Coordinate", fontsize=16)
    plt.show()
    plt.close()
    
def plotting_total_energies(energies):
    plt.figure(figsize=(12, 8))
    plt.scatter(energies[1:, 0], energies[1:, 1], s=10, c='k')
    #plt.ylim(-2*abs(np.median(energies[:100,1])), 2*abs(np.median(energies[:100,1])))
    plt.xlabel("Time t", fontsize=16)
    plt.ylabel("Total Energy", fontsize=16)
    plt.show()
    plt.close()

def plotting_kin_energies(energies):
    plt.figure(figsize=(12, 8))
    plt.scatter(energies[1:, 0], energies[1:, 1], s=10, c='k')
    #plt.ylim(-2*abs(np.median(energies[:100,1])), 2*abs(np.median(energies[:100,1])))
    plt.xlabel("Time t", fontsize=16)
    plt.ylabel("Kinetic Energy", fontsize=16)
    plt.show()
    plt.close()

def plotting_lambdas(lambdas):
    plt.figure(figsize=(12, 8))
    plt.scatter(lambdas[:, 0], lambdas[:, 1], s=10, c='k')
    plt.xlabel("Time t", fontsize=16)
    plt.ylabel("Lambda Scale Sactor", fontsize=16)
    plt.show()
    plt.close()
    

def main():

    
    start_time = time.time()
    fs = 16 # set fontsize
    
    ### NOTE: parameters of simulation, including initial physical conditions
    
    # kinetic energy lambda function with input velocity
    KE = lambda v: 0.5*v**2

    # set phase from user input and guarantee reproducible results with seed
    #np.random.seed(0)
    phase = input("Phase state of Argon to be simulated (gas, liquid, solid): ")
    # set temperature and density
    if phase == "gas":
        temperature = 3.0
        density = 0.3
        print ("phase:", phase)
    elif phase == "liquid":
        temperature = 1.0
        density = 0.8
        print ("phase:", phase)
    elif phase == "solid":
        temperature = 0.5
        density = 1.2
        print ("phase:", phase)
    else: # phase != "solid" or phase != "liquid" or phase != "gas":
        phase = "gas"
        temperature = 3.0
        density = 0.3
        print ("No correct input - phase set to gas")
    
    ### initial parameters start ###
    num_unit_cell_per_dim = 3
    num_unit_cells = 27         # 3**3 unit cells per dimension (3D)
    num_atom_per_unit_cell = 4  # number of atoms per unit cell
    num_atoms = num_unit_cells * num_atom_per_unit_cell # number of atoms
    h = 1e-3                 # timestep
    t_end = 1.          # end simulation time
    num_steps = int(t_end / h)   # number of time steps
    print ("Running for", num_steps, "steps with h:", h, "until", t_end)
    box_size = np.power((num_atoms / density),1/3) # size of simulation box
    print ("density:", density, "num atoms:", num_atoms)
    print ("hence box size:", np.round(box_size, 1))
    unit_cell_size = box_size / 3 # size of a unit cell per dimension
    ucs = unit_cell_size

    # initialize arrays to store position, velocity for each particle;
    # energies kinetic, potential, total;
    # lambda factors; forces; distances; dUdR-array
    particle_state = np.zeros((num_steps, num_atoms, 2*3))
    kinetic_energy = np.zeros((num_steps, 2))
    potential_energy = np.zeros((num_steps, 2))
    total_energy = np.zeros((num_steps, 2))
    DEkin, DEpot, DEtot = 0, 0, 0
    lambdas = np.zeros((num_steps, 2))
    distances = np.zeros((num_atoms, num_atoms-1))
    array_dUdr = np.zeros((num_atoms, num_atoms-1))
    
    # rescaling parameters for lambda factors also defined
    mu = 1
    sigma = np.sqrt(temperature)
    rescaling_interval = 20
    rescaling_time = rescaling_interval*h
    rescaling_end = 50 * rescaling_time
    ### initial parameters end ###

    ### initial conditions start ###
    # positions of first unit cell
    particle_state[0, 0, :3] = np.array([0, 0, 0])
    particle_state[0, 1, :3] = np.array([0, ucs/2., ucs/2.])
    particle_state[0, 2, :3] = np.array([ucs/2., ucs/2., 0])
    particle_state[0, 3, :3] = np.array([ucs/2., 0, ucs/2.])

    # setting initial coordinates on an fcc lattice
    p = 0 # counter for particle
    for x in range(num_unit_cell_per_dim):
        for y in range(num_unit_cell_per_dim):
            for z in range(num_unit_cell_per_dim):
                for i in range(num_atom_per_unit_cell):
                    
                    particle_state[0, p, :3] = particle_state[0, p % 4, :3] + ucs*np.array([x, y, z])
                    p += 1
    
    # set particles velocites
    velocity_distribution = np.power(2*np.pi*temperature, -1.) * np.random.normal(mu, sigma/np.sqrt(temperature), (num_atoms, 1)) * rand_norm_unit_vector(num_atoms)
    particle_state[0, :, 3:] = velocity_distribution
    
    # inital lambda and energy values
    lambdas[0, 1] = lambda_scale_factor(particle_state[0], temperature, num_atoms)
    
    for i in range(num_atoms):
        
        pos = particle_state[0, i, :3]
        vel = particle_state[0, i, 3:]
        
        # implement MIC here with a temporary array to store the new positions of size 108x3
        # do the actual MIC, and perform all calculations with the positions from "temp_state"
        temp_state = ( ( pos - particle_state[0, :, :3] + box_size/2.) % box_size ) - box_size/2.
        acc, Epot = particle_force(temp_state, i)
        DEkin += 0.5*np.sum(vel**2)
        DEpot += Epot
        DEtot += (0.5*np.sum(vel**2) + Epot)
        
    kinetic_energy[0] = 0, DEkin
    potential_energy[0] = 0, DEpot
    total_energy[0] = 0, DEtot
    
    ### initial conditions end ###
    
    # note: 'd' before a variable denotes timestep (ie, small) change,
    # so, dPE, dKE are small changes in potential_energy and kinetic_energy
    
    load_time = time.time()
    t_loading = start_time - load_time
    
    ### run simulation
    for s in range(1, num_steps):#num_steps):
        t = float(s*h)
        if s % 50 == 0:
            print ("step:", s)
        # rescale when needed
        if (t % rescaling_time == 0.) and (t < rescaling_end):
            min_s = s-1 - rescaling_interval
            abs_s = lambda s: (abs(s) + s)/2
            
            average_lambda = np.average(lambdas[int(abs_s(min_s)):s-1])
            particle_state[s-1, :, 3:] = average_lambda * particle_state[s-1, :, 3:]

        # integrate EOM for single particles
        for i in range(num_atoms):
            DEkin, DEpot, DEtot = 0, 0, 0
            
            pos_1 = particle_state[s-1, i, :3]
            vel_1 = particle_state[s-1, i, 3:]
            
            # implement MIC here with a temporary array to store the new positions of size 108x3
            # do the actual MIC, and perform all calculations with the positions from "temp_state"
            temp_state = ( ( pos_1 - particle_state[s-1, :, :3] + box_size/2.) % box_size ) - box_size/2.
            
            # integrate EqOM and obtain Ekin, Epot
            first_acc, first_Epot = particle_force(temp_state, i)
            particle_state[s, i, :3] = next_position(particle_state[s-1, i, :3], particle_state[s-1, i, 3:],
                                                    first_acc, h)
            

            next_acc, next_Epot = particle_force(particle_state[s], i)
            next_v = next_velocity(particle_state[s-1, i, 3:], first_acc, next_acc, h)
            
            particle_state[s, i, 3:] = next_v
            DEkin += 0.5*np.sum(next_v**2)
            DEpot += next_Epot
            DEtot += (0.5*np.sum(next_v**2) + next_Epot)
            ### Calculation of lambdas should work
            lambdas[s] = t, lambda_scale_factor(particle_state[s], temperature, num_atoms)
            
        # put particles back into the box
        particle_state[s, :, :3] = particle_state[s, :, :3] % box_size
            
        kinetic_energy[s] = t, DEkin
        potential_energy[s] = t, DEpot
        total_energy[s] = t, DEtot
    
# run PCF and pressure analysis
    for i in range(num_atoms):
        # mask current particle
        mask = np.ones(num_atoms, dtype=bool)
        mask[i] = False
        distances[i] = Distances_Many_Bodies(body1=particle_state[0, i, :3], others=(particle_state[0, :, :3])[mask] )
        
    array_dUdr = LJP_dUdr(distances)
    pair_correlation_function(distances, box_size, num_atoms)
    pressure = pressure_calc(dists=distances, dUdr=array_dUdr, T=temperature, rho=density, num_atoms=num_atoms)
    print ("Calculated Pressure: ", pressure)
    
    # Diagnostics
    print ("Median Total Energy:", np.round(np.median(total_energy), 4) )
    print ("Median lambda:", np.round(np.median(lambdas[:,1]), 1) )
    # plotting_state(particle_state, box_size)
    # plotting_total_energies(total_energy)
    # plotting_lambdas(lambdas)
                            
    end_time = time.time()
    t_total = end_time - start_time
    t_sim = end_time - load_time
    print ("total runtime in seconds:", t_total, "/ minutes:", t_total/60.)
        
    print ("done")
