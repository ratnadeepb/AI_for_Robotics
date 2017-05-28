# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:20:06 2017

@author: Ratnadeepb
"""

## Particle Filter Class

import numpy as np
import numpy.linalg as la
import scipy.stats as st
from robot_class import robot

class particle_filter:
    def __init__(self, no_of_particles, my_world, my_robot):
        if my_world.__class__.__name__ != 'world':
            raise ValueError("Invalid Object")
        if my_robot.__class__.__name__ != 'robot':
            raise ValueError("Invalid Object")
        self.particles = []
        self.my_robot = my_robot
        self.my_world = my_world
        for i in range(no_of_particles):
            self.particles.append(robot(world=self.my_world))
        self.particles = np.array(self.particles)
    
    def move_particles(self, turn, distance):
        self.my_robot.move(turn, distance)
        for i in range(self.particles.shape[0]):
            self.particles[i].move(turn, distance)
        
    def sense(self):
        self.my_robot.sense()
        for i in range(self.particles.shape[0]):
            self.particles[i].sense()
        
    def set_weight(self):
        self.weights = np.zeros(self.particles.shape[0])
        
        for i in range(self.weights.shape[0]):
            w = []
            z = st.norm.pdf(self.particles[i].Z)
            for j in range(self.my_world.landmarks.shape[0]):
                w.append(z[j])
            # The norm function is randomly chosen
            # No mathematical analysis behind it
            # Apart from a bit of intution from linear algebra
            self.weights[i] = la.norm(self.my_robot.Z - np.array(w))
        
    def select_particles(self):
        new_particles = []
        index = np.random.randint(0, self.weights.shape[0])
        m = 2 * self.weights.max()
        beta = 0
        for i in range(self.particles.shape[0]):
            beta += np.random.uniform(0, m)
            while beta > self.weights[index]:
                beta = beta - self.weights[index]
                if index == self.weights.shape[0] - 1:
                    index = 0
                else:
                    index += 1
            new_particles.append(self.particles[index])
        self.particles = np.array(new_particles)

if __name__ == "__main__":
    from robot_class import world
    landmarks = np.array([[20, 20],
                      [80, 80],
                      [20, 80],
                      [80, 20]])

    world_size = 100    
    my_world = world(world_size, landmarks)
    myrobot = robot(world=my_world)
    myparticles = particle_filter(1000, my_world, myrobot)
    for i in range(10):
        myrobot.move(-np.pi/2, 15)
        myparticles.move_particles(-np.pi/2, 15)
        myrobot.sense()
        myparticles.sense()
        myparticles.set_weight()
        myparticles.select_particles()