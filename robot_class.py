# -*- coding: utf-8 -*-
"""
Created on Wed May 17 20:23:36 2017

@author: Ratnadeepb
"""

## Robot class

import numpy as np

class world:
    def __init__(self, size, landmarks):
        self.size = size
        try:
            self.landmarks = np.array(landmarks)
        except:
            raise ValueError("Invalid Input")
        
        # Landmarks should always have 2 rows (x, y)
        if self.landmarks.shape[1] != 2:
            raise ValueError("Incorrect number of coordinates!")
    
    def __repr__(self):
        return "[{}, {}]".format(self.size, self.landmarks)

class robot:
    def __init__(self, **kwargs):
        if len(kwargs) < 1:
            raise ValueError("At least 1 named argument required")
        if len(kwargs) > 2:
            raise ValueError("Only 2 named arguments supported")

        if len(kwargs) == 2:
            try:
                # Create my_world
                self.my_world = world(kwargs['world_size'], 
                                   kwargs['landmarks'])
            except:
                raise ValueError("Invalid Inputs")
        
        elif len(kwargs) == 1:
            try:
                # Use an existing world
                self.my_world = kwargs['world']
            except NameError:
                raise NameError("Invalid world name")
            except ValueError:
                raise ValueError("Invalid world value")
        
        self.x = np.random.rand() * self.my_world.size
        self.y = np.random.rand() * self.my_world.size
        self.Z = np.zeros((self.my_world.landmarks.shape[0], 1))            

        self.orientation = np.random.rand() * 2 * np.pi
        self.v = np.array([self.x, self.y])

        self.forward_noise = np.random.randint(1, 10) * np.random.randn()
        self.turn_noise = np.random.randint(1, 10) * np.random.randn()
        self.sense_noise = np.random.randint(1, 10) * np.random.randn() 
        
    def __repr__(self):
        return "{:.3f}\n{:.3f}\n{:.3f}".format(self.x, self.y, 
                                                self.orientation)
        
    def set_pos(self, x, y, orientation):
        if x > self.my_world.size or x < 0:
            raise ValueError("Invalid Input")
        if y > self.my_world.size or y < 0:
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
        R = np.array([[np.cos(turn), -np.sin(turn), 0],
                       [np.sin(turn), np.cos(turn), 0],
                       [0, 0, 1], [0, 0, 0]])
        
        forward += self.forward_noise
        t = np.array([forward, 0, 0, 1]).reshape(4, 1)

        T = np.column_stack((R, t))
        
        v = np.array([self.x, self.y, 0, 1]).reshape(4, 1)
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
    
    def set_noise(self, forward_noise, turn_noise, sense_noise):
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise
        
    def sense(self):
        for i in range(self.Z.shape[0]):
            z = np.abs((self.my_world.landmarks[i, 0] - self.x) ** 2) + \
                np.abs((self.my_world.landmarks[i, 1] - self.y) ** 2)
            z = np.round(np.sqrt(z)) + self.sense_noise * np.random.randn()
            self.Z[i] = z