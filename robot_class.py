# -*- coding: utf-8 -*-
"""
Created on Wed May 17 20:23:36 2017

@author: Ratnadeepb
"""

## Robot class

import numpy as np

landmarks = np.array([[20, 20],
                      [80, 80],
                      [20, 80],
                      [80, 20]])

world_size = 100  # A 100 m by 100 m world

class world:
    def __init__(self, size, landmarks):
        self.size = size
        self.landmarks = landmarks

class robot:
    def __init__(self, world_size):
        self.size = world_size
        self.x = np.random.rand() * self.size
        self.y = np.random.rand() * self.size
        self.orientation = np.random.rand() * 2 * np.pi
        self.v = np.array([self.x, self.y])
        self.forward_noise = np.random.randn()
        self.turn_noise = np.random.randn()
        self.sense_noise = np.random.randn()
        
    def set_pos(self, x, y, orientation):
        if x > self.x - self.size:
            raise ValueError("Invalid Input")
        if y > self.y - self.size:
            raise ValueError("Invalid Input")
            
        self.x = x
        self.y = y
        
        orientation %= (2 * np.pi)
        if orientation < 0:
            orientation += 2 * np.pi
        self.orientation += orientation
        self.orientation %= (2 * np.pi)
        
    def move(self, turn, forward):
        turn %= (2 * np.pi)
        if turn < 0:
            turn += 2 * np.pi
        
        self.orientation += turn
        self.orientation %= (2 * np.pi)
        
        turn += self.turn_noise
        R = np.array([[np.cos(turn), -np.sin(turn)],
                       [np.sin(turn), np.cos(turn)],
                       [0, 0]])
        
        forward += self.forward_noise
        t = np.array([forward, 0, 1]).reshape(3, 1)

        T = np.column_stack((R, t))
        
        v = np.array([self.x, self.y, 1]).reshape(3, 1)
        temp = T @ v
        
        if temp[0,0] > 100:
            x1 = 100
        elif temp[0,0] < 0:
            x1 = 0
        else:
            x1 = temp[0,0]
            
        if temp[1,0] > 100:
            y1 = 100
        elif temp[1,0] < 0:
            y1 = 0
        else:
            y1 = temp[1,0]
        
        self.x = x1
        self.y = y1
    
    def set_sense_noise(self, noise):
        self.sense_noise = noise
        
    def sense(self):
        self.z = np.zeros((2, 1))
        self.z[0, 0] = self.sense_noise * np.random.randn() + np.round(self.x)
        self.z[1, 0] = self.sense_noise * np.random.randn() + np.round(self.y)