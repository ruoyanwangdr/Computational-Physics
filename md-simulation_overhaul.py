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

def Distance_Between_Bodies(Body1, Body2):
    """ returns distance between Body1 and Body2. """
    Pos1 = Body1[:3]
    Pos2 = Body2[:3]
    R = np.sqrt( np.sum( (Pos1-Pos2)**2 ) )
    return R

def rand_norm_unit_vector(num_atoms):
    """ returns random unit vectors for all particles to get direction randomised velocites. """

    vec = np.random.uniform(-1.0, 1.0, (num_atoms, 3))
    norm_vec = np.sqrt(np.sum(vec**2))
    return vec / norm_vec

def lennard_jones_potential(r):
    """ returns value of lennard-jones potential at a given distance r. """
    
    LJPot =  4.*(np.power(r, -12) - np.power(r, -6))

    return LJPot

def LJP_dUdr(r):
    """ unitless derivaive of lennard-jones potential
    with respect to input vector r."""
    
    dUdr = -48 / np.power(r, 13) + 24 / np.power(r, 7)
    
    return dUdr

def current_velocity(x_new, x_prev, h):
    """ returns current velocity of a particle from next
    position at timestep. """

    """
    input parameters
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
    """ returns next position of particle. """

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

    """

    x_next = pos + vel*h + acc*(0.5*h**2)
    
    return x_next


def next_velocity(vel, acc, next_acc, h):
    """ returns velocity at next time step for a particle. """

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

    """

    return vel + 0.5*h*(acc + next_acc)

def lambda_scale_factor(state, T, num_atoms):
    """ returns lambda scale factor for a given particle state. """

    """
    parameters
    ----------
    state : array
        state of the simulation at current time stamp
    T : float
        temperature
    num_atoms : int
        total number of particles
    """
    
    velocities = state[:, 3:]
    lambda_factor = np.sqrt( (num_atoms - 1) * 3 * T / np.sum(velocities**2) )
    
    return lambda_factor

def particle_force(state, particle):

    """ returns the acceleration acting on a single body due to LJ-Potential from the other particles,
    and its potential energy. """

    """
    parameters
    ----------
    state : array
        state of simulation at current time stamp
    particle : int (array index) 
        state array index for particle

    """
    
    pos_1 = state[particle, :3]
    
    for i, other_body in enumerate(state):
        if i != particle:
            
            # calc distances and else with new 
            pos_2 = other_body[:3]
            Dpos = pos_2 - pos_1
            r = Distance_Between_Bodies(pos_1, pos_2)
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

def plotting_state(state, box_size):
    margin = 0.3
    plt.figure()
    for p in np.rollaxis(state, 1):
        plt.scatter(p[0, 0], p[0, 1], s=10, c='k', zorder=25)
        plt.scatter(p[:, 0], p[:, 1], s=1, c='r')

    plt.xlim(-margin, box_size+margin)
    plt.ylim(-margin, box_size+margin)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()
    plt.close()
    
def plotting_total_energies(energies):
    plt.figure()
    plt.scatter(energies[:, 0], energies[:, 1], s=10, c='k')
    #plt.ylim(-2*abs(np.median(energies[:100,1])), 2*abs(np.median(energies[:100,1])))
    plt.xlabel("Time t")
    plt.ylabel("Total Energy")
    plt.show()
    plt.close()

def plotting_kin_energies(energies):
    plt.figure()
    plt.scatter(energies[:, 0], energies[:, 1], s=10, c='k')
    #plt.ylim(-2*abs(np.median(energies[:100,1])), 2*abs(np.median(energies[:100,1])))
    plt.xlabel("Time t")
    plt.ylabel("Kinetic Energy")
    plt.show()
    plt.close()

def plotting_lambdas(lambdas):
    plt.figure()
    plt.scatter(lambdas[:100, 0], lambdas[:100, 1], s=10, c='k')
    plt.xlabel("Time t")
    plt.ylabel("Lambda Scale Factor")
    plt.show()
    plt.close()

def main():

    ### simulation parameters, including initial physical conditions
    
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
        print ("No correct input - phase set to gas")
    
    ### initial parameters start ###
    num_unit_cell_per_dim = 3
    num_unit_cells = 27         # 3**3 unit cells per dimension (3D)
    num_atom_per_unit_cell = 4  # number of atoms per unit cell
    num_atoms = num_unit_cells * num_atom_per_unit_cell # number of atoms
    h = 1e-3                 # timestep
    t_end = .1                # end simulation time
    num_steps = int(t_end / h)   # number of time steps
    box_size = np.power((num_atoms / density),1/3) # size of simulation box
    print ("density:", density, "num atoms:", num_atoms)
    print ("hence box size:", np.round(box_size, 1))
    unit_cell_size = box_size / 3 # size of a unit cell per dimension
    ucs = unit_cell_size

    # initialize arrays to store position, velocity for each particle;
    # energies kinetic, potential, total;
    # lambda factors; forces; distances
    particle_state = np.zeros((num_steps, num_atoms, 2*3))
    kinetic_energy = np.zeros((num_steps, 2))
    potential_energy = np.zeros((num_steps, 2))
    total_energy = np.zeros((num_steps, 2))
    DEkin, DEpot, DEtot = 0, 0, 0
    lambdas = np.zeros((num_steps, 2))

    # rescaling parameters for lambda factors also defined
    mu = 1
    sigma = np.sqrt(temperature)
    rescaling_interval = 5
    rescaling_time = rescaling_interval*h
    rescaling_end = 10 * rescaling_time
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
    velocity_distribution = np.random.normal(mu, sigma, (num_atoms, 1)) * rand_norm_unit_vector(num_atoms)
    particle_state[0, :, 3:] = velocity_distribution
    # for sanity check: set random positions
    #particle_state[0, :, :3] = np.random.uniform(0, box_size, (num_atoms, 3))
    # for sanity check
    #print (particle_state[0, 10])
    
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
   
    ### run simulation
    for s in range(1, num_steps):
        t = float(s*h)
        if s % 50 == 0:
            print ("step:", s)
        # rescale when needed
        ### NOT WORKING ATM
        #if (t % rescaling_time == 0.) and (t < rescaling_end):
            
            #average_lambda = np.average(lambdas[min(0, s-1 - rescaling_interval):s-1])
            #particle_state[s-1, :, 3:] = average_lambda * particle_state[s-1, :, 3:]

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
            next_vel = next_velocity(particle_state[s-1, i, 3:], first_acc, next_acc, h)
            
            particle_state[s, i, 3:] = next_vel
            DEkin += 0.5*np.sum(next_vel**2)
            DEpot += next_Epot
            DEtot += (0.5*np.sum(next_vel**2) + next_Epot)
            ### Calculation of lambdas should work
            lambdas[s] = t, lambda_scale_factor(particle_state[s], temperature, num_atoms)
            
        # put particles back into the box
        particle_state[s, :, :3] = particle_state[s, :, :3] % box_size
            
        kinetic_energy[s] = t, DEkin
        potential_energy[s] = t, DEpot
        total_energy[s] = t, DEtot
    
    # Diagnostics
    print ("Mean Total Energy:", np.round(np.mean(total_energy), 4) )
    print ("Mean lambda:", np.round(np.mean(lambdas[:,1]), 1) )
    plotting_state(particle_state, box_size)
    plotting_kin_energies(kinetic_energy)
    plotting_total_energies(total_energy)
    plotting_lambdas(lambdas)
    #print (total_energy)
    print (lambdas)
                                                                
    print ("done")
    
if __name__ == '__main__':
    sys.exit(main())
