__author__ = "Robin Mentel, Ruoyan Wang, Christopher Seay"

"""
simulates the molecular dynamics of argon atoms in a pseudo-infinite 3D box.
handles liquid, solid, and gaseous states. 
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

def Get_initial_velocities(nbodies, v_central, T):
    # kBT = 2sigma^2 -> sigma = sqrt(kBT/2)
    kB = 1. # change later
    sigma = np.power(kB*T*0.5, 0.5)
    velocities = np.random.normal(loc=v_central, scale=sigma, size=(nbodies, ndim))
    return velocities

def distance_between_bodies(Ndim, body1, body2):
    """calculates the distance between two argon atoms in a given number of dimensions."""
    
    """
    parameters
    ----------
    Ndim: : int
        number of dimensions
    body1 : array 
        placeholder
    body2 : array
        placeholder

    returns
    ----------
    """

    position_1 = body1[1:Ndim]
    position_2 = body2[1:Ndim]
    distance = np.sqrt(np.sum(((position_1 - position_2)**2)))

    return distance

#def single_body_acceleration(ndim, box_size, sigma, mass, S, bodies, body_index): # old
def single_body_acceleration(ndim, box_size, sigma, mass, temp_bodies, body_index): # new
    """finds the acceleration on a single body by LJ-Potential from others."""
    #print ("body_index:", body_index)
    epsilon = np.power(2., 1/6)*sigma
    body_index = int(body_index)
    #body = bodies[S, body_index] # old
    body = temp_bodies[body_index] # new
    position_1 = body[:ndim]
    acceleration = np.asarray([0, 0, 0])
    
    sum_potential_energy = 0
    
    # go through the rest of bodies to calc acceleration
    #for i, other_body in enumerate(bodies[S, :, :]): # old
    for i, other_body in enumerate(temp_bodies): # new
        if int(i) != body_index:
            # implement minimal image convention:
            # "other_body" is simply the particle data of the other particle
            #other_body = minimal_image_convention(S, ndim, body, other_body, box_size) # old one
            other_body = temp_bodies[i] # new
                       
            # calc distances and else with new 
            position_2 = other_body[:ndim]
            position_difference = position_2 - position_1
            distance = distance_between_bodies(ndim, body, other_body)
            
            # try gravity interaction to get it work, then switch to LJ
            #Scalar_acceleration = 1*M*other_body[0]/np.power(R, 3.)
            # split up vec(acc) in scalar acc = abs(a)/r, and vec(e_a) = vec(r)
            #Scalar_acceleration = 4*epsilon/mass*( -12.*np.power(sigma, 12.)*np.power(R, -14.) + 6.*np.power(sigma, 6.)*np.power(R, -8.) )
            #acceleration = np.add(acceleration, Scalar_acceleration*DPos)
            
            # calculate potential energy in that potential
            #dU = 4*epsilon*(np.power(sigma/R, 12) - np.power(sigma/R, 6.))
            
            
            # Updated by Ruoyan
            # Convert potential energy and scalar acceleration to dimensionless
            # With convertion, length will be in units of sigma, energy in units of epsilon, 
            # and time in units of (m*sigma**2/epsilon)**(1/2)
            scalar_acceleration = 4*epsilon/(mass*sigma**2)*(12*(distance)**(-14) - 6*(distance)**(-8))
            acceleration = np.add(acceleration, scalar_acceleration * position_difference)
            
            dPE = 4*epsilon*( distance**(-12) - distance**(-6) )
            sum_potential_energy += dPE
    
    return acceleration, sum_potential_energy

# return new pos of other body if too far away
# not needed anymore now
######
def minimal_image_convention(S, ndim, body, other_body, box_size):

    """ minimal image convention of designing a pseudo-infinite box."""

    """
    parameters
    ----------


    returns
    ----------

    """

    position_1 = body[1:1+ndim]
    position_2 = other_body[1:1+ndim]
    # changed variable names here:
    #change_in_position = position_2 - position_1
    distance_vector = position_2 - position_1
    new_other = np.zeros_like(other_body)
    new_other[:] = other_body[:]
    ### OLD
    # go through all three dimensions D and check if the other body is farther away than half a box:
    # if so, add or subtract box size from pos of other body:
    for i in range(3):
        if distance_vector[i] > box_size/2:
            new_other[1+i] = position_2[i] - box_size
        elif distance_vector[i] < -box_size/2:
            new_other[1+i] = position_2[i] + box_size    
    
    return new_other

# plotting functions
    
def plot_entire_simulation(particle_data, x0, x1):
    plt.figure()
    for p in np.rollaxis(particle_data, 1):
        plt.scatter(p[0, 1], p[0, 2], s=10, c='k', zorder=25)
        plt.scatter(p[:, 1], p[:, 2], s=1)

    plt.xlim(x0, x1)
    plt.ylim(x0, x1)

def main():
    ### params
    global nbodies, mass, ndim, dt, T
    nbodies = 4
    T = 100 # tempertaure in K
    # set particle mass
    mass = 1
    ndim = 3
    t_end = .5
    dt = 1e-4 # timestep
    nsteps = int(t_end/dt)
    particle_data = np.zeros((nsteps, nbodies, 2*ndim))
    # energies: t, kinetic_energy, potential_energy, total_energy
    energies = np.zeros((nsteps, 4))

    # initial conditions
    box_size = 3.
    # positions
    x0, x1 = 0., box_size
    particle_data[0, :, :3] = np.random.uniform(x0, x1, particle_data[0, :, :3].shape)
    #velocities
    #v0, v1 = -.5, .5 # old
    #particle_data[0, :, 3:] = np.random.uniform(v0, v1, particle_data[0, :, 3:].shape) # old
    v_central = 0.5
    particle_data[0, :, 3:] = Get_initial_velocities(nbodies, v_central, T) # new
    # sigma, epsilon for L-J-Potential
    # sigma = 3.405 # in angstrom
    sigma = .1
    epsilon = np.power(2., 1/6)*sigma
    
    ### note: 'd' before a variable denotes timestep (ie, small) change,
    ### so, dPE, dKE are small changes in potential_energy and kinetic_energy
    # run simulation:
    for s in range(nsteps - 1):
        # for every time step:
        potential_energy = 0.
        kinetic_energy = 0.
        t = s*dt
    
        for i, body1 in enumerate(particle_data[s]):
            # for every particle:
        
            # get current pos, vel
            pos1 = body1[:ndim]
            velocity_1 = body1[ndim:]
            
            # define temporary particle_data-array for simpler MIC
            temp_particle_data = np.zeros_like(particle_data[0, :, :ndim])
            
            # calc acceleration with current ('old') positions
            # implement minimum image convention
            temp_particle_data[:] =  ( (pos1 - particle_data[s, :, :ndim] + box_size/2) % box_size ) - box_size/2
            #old_acceleration, dPE = single_body_acceleration(ndim, box_size, sigma, mass s, particle_data, i) # OLD
            old_acceleration, dPE = single_body_acceleration(ndim, box_size, sigma, mass, temp_particle_data, i) # NEW 
            # calc new positions and copy to data array
            new_position = pos1 + dt*velocity_1 + 0.5*dt*dt/mass*old_acceleration
            particle_data[s+1, i, :ndim] = new_position
            
            # calc acceleration with new positions
            # implement minimum image convention
            temp_particle_data[:] =  ( (pos1 - particle_data[s+1, :, :ndim] + box_size/2) % box_size ) - box_size/2
            #new_acceleration = single_body_acceleration(ndim, box_size, sigma, mass, s+1, particle_data, i)[0] # OLD
            new_acceleration = single_body_acceleration(ndim, box_size, sigma, mass, temp_particle_data, i)[0] # NEW + TEMP
            # calc new velocity with old and new acceleration, and copy to data array
            new_velocity = velocity_1 + 0.5*dt*dt/mass*(new_acceleration + old_acceleration)
            #print(new_velocity)
            particle_data[s+1, i, ndim:2*ndim] = new_velocity
        
            # energies
            potential_energy += dPE
            kinetic_energy += 0.5*mass*np.sum(velocity_1*velocity_1)
    
        # put all particles back in the box:
        # it might be that this ruins all energy conservation
        particle_data[s+1, :, :ndim] = np.mod(particle_data[s+1, :, :ndim], box_size)
    
        # calculate the energies
        total_energy = potential_energy + kinetic_energy
        energies[s] = t, kinetic_energy, potential_energy, total_energy
    
    plot_entire_simulation(particle_data, x0, x1)

if __name__ == '__main__':
    sys.exit(main())
