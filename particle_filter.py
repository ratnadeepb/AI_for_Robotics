# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:20:06 2017

@author: Ratnadeepb
"""

import numpy as np
from robot_class import world, robot
## Particle Filters

class particle_filter:
    def __init__(self, no_of_particles, my_world):
        if not isinstance(my_world, world):
            raise ValueError("Invalid Object")
        self.particles = np.zeros((no_of_particles, 3))
        self.my_world = my_world
        for i in range(no_of_particles):
            self.particles[i] = robot(self.my_world.size, 
                                      self.my_world.landmarks)
    
    def move_particles(self, turn, distance):        
        for i in range(len(self.particles)):
            self.particles[i].move(turn, distance)