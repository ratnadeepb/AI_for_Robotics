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
    def __init__(self, no_of_particles, my_world):
        if my_world.__class__.__name__ != 'world':
            raise ValueError("Invalid Object")
        self.particles = []
        self.my_world = my_world
        for i in range(no_of_particles):
            self.particles.append(robot(world=self.my_world))
        self.particles = np.array(self.particles)
    
    def move_particles(self, turn, distance):
        for i in range(self.particles.shape[0]):
            self.particles[i].move(turn, distance)
        
    def sense(self):
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
            self.weights[i] = la.norm(np.array(w))