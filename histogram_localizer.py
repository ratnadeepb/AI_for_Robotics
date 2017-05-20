# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 18:33:32 2017

@author: Ratnadeepb
"""

import numpy as np
import sys

# 2D localization
def moved(probs, move, p_move):
    """
    This function takes the probablity distribution along a 2D grid and 
    returns the probability distrubution after the move.
    
    @Inputs:
        probs       : a 2D grid distribution of current probabilities
        move        : the move instruction
        p_move      : the probability that the move was made
    
    @Output:
        probs       : the original distribution mutated by move
    """
    
    len_rows = len(probs)     # Vertical length of the grid
    len_cols = len(probs[0])  # Horizontal length of the grid
    
    temp = np.zeros(probs.shape)
    
    for r in range(len_rows):
        for c in range(len_cols):
            temp[r][c] += probs[(r - move[0]) % len_rows]\
                            [(c - move[1]) % len_cols] * p_move \
                            + probs[r][c] * (1 - p_move)
    
    return np.round(temp / temp.sum(), decimals=4)

def sensed(colors, probs, sense, p_sense):
    """
    This function takes the probablity distribution along a 2D grid and 
    returns the probability distrubution after the sense action.
    
    @Inputs:
        colors      : a 2D grid distribution of colors, same size as probs
                      This is an over simplified set of features
        probs       : a 2D grid distribution of current probabilities
        sense       : the sense value
        p_sense     : the probability that the sense value is correct
    
    @Output:
        probs       : the original distribution mutated by sensing
    """
    
    len_rows = len(probs)
    len_cols = len(probs[0])
    
    temp = np.zeros(probs.shape)
    
    for r in range(len_rows):
        for c in range(len_cols):
            hit = (sense == colors[r][c])
            temp[r][c] += probs[r][c] * (hit * p_sense + (1 - hit) * (1 - p_sense))
    
    return np.round(temp / temp.sum(), decimals=4)

def localise(probs, colors, p_move, p_sense, move, sense, action):
    """
    This function takes the probablity distribution along a 2D grid of over
    simplified features and returns the probability distrubution after both
    moving and sensing action.
    
    @Inputs:
        colors      : a 2D grid distribution of colors, same size as probs
                      This is an over simplified set of features
        probs       : a 2D grid distribution of current probabilities
        move        : the move instruction
        p_move      : the probability that the move was made
        sense       : the sense value
        p_sense     : the probability that the sense value is correct
        action      : this is a boolean value determining whether to move first
                      (True) or sense first (False)
    
    @Output:
        probs       : the original distribution mutated by move and sense
    """
    
    # Check that probs is a list or tuple or numpy array
    assert isinstance(probs, tuple) or isinstance(probs, list) or \
    isinstance(probs, np.ndarray), "a 2D grid distribution is required"
    # Convert probs into a numpy array if not already
    if isinstance(probs, tuple) or isinstance(probs, list):
        probs = np.array(probs)
    # Check that probs is a 2D grid distribution
    assert len(probs.shape) == 2, "a 2D grid is expected"
    # Check that colors is a list or tuple or numpy array
    assert isinstance(colors, tuple) or isinstance(colors, list) or \
    isinstance(colors, np.ndarray), "a 2D grid distribution is required"
    # Convert probs into a numpy array if not already
    if isinstance(colors, tuple) or isinstance(colors, list):
        colors = np.array(colors)
    # Check that probs is a 2D grid distribution
    assert len(colors.shape) == 2, "a 2D grid is expected"
    # Check that probs and colors have the same shape
    assert probs.shape == colors.shape, "colors and probs represent the same grid"
    # Check that move is a proper 2D instrution
    assert len(move) == 2, "we need 2D instructions for move"
    for m in move:
        assert np.issubdtype(m, np.int), "we need a single move instruction"
    # Check that sense has a proper value
    assert sense in colors, "sense value needs to be part of colors"
    
    
    # Check action
    if action:
        probs = moved(probs, move, p_move)
        probs = sensed(colors, probs, sense, p_sense)
    else:
        probs = sensed(colors, probs, sense, p_sense)
        probs = moved(probs, move, p_move)
    
    return probs

def setup(p_move, p_sense, action):
    """
    This function setsup all the variables.
    
    @Input:
        p_move      : the probability that the move was made
        p_sense     : the probability that the sense value is correct
        action      : this is a boolean value determining whether to move first
                      (True) or sense first (False)
                      
    @Output:
        colors      : a 2D grid distribution of colors, same size as probs
                      This is an over simplified set of features
        probs       : a 2D grid distribution of current probabilities
    """

    # We are going to use a random sized grid
    rows, cols = np.random.randint(4, 10, size=2)
    
    # We are going to use 'R' and 'G' as grid features by default
    colors = np.random.choice(['R', 'G'], size=(rows, cols))
    
    # Using the uniform distribution
    # State of maximum confusion during setup
    probs = np.round(np.ones((rows, cols)) / (rows * cols), decimals=4)
    
    return (colors, probs)

def run_sim(moves, senses, p_move, p_sense, action, rnd, **kwargs):
    """
    This function is the primary function that runs the histogram localisation.
    
    @Inputs:
        moves       : allows the user to provide a sequence of moves
        senses      : allows the user to provide a sequence of senses
        p_move      : the probability that the move was made
        p_sense     : the probability that the sense value is correct
        action      : this is a boolean value determining whether to move first
                      (True) or sense first (False)
        rnd         : a boolean value that let's the user control if they want
                      to run the simulation with (True) or without (False) the
                      setup function.
        **kwargs    : if rnd is True then this can be used by the user to 
                      provide the initial probs and the colors grids
                      
    @Output:
        probs       : a 2D grid distribution of current probabilities
        
    @Doctests:
        >>> run_sim([[0,0]], ['R'], 1.0, 1.0, False, True, 
        colors=[['G', 'G', 'G'],['G', 'R', 'G'],['G', 'G', 'G']], 
        probs = [[ 0.1111,  0.1111,  0.1111],[ 0.1111,  0.1111,  0.1111],
        [ 0.1111,  0.1111,  0.1111]])
    
        array([[0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0]])
    """
    
    # Check that moves is valid
    assert isinstance(moves, tuple) or isinstance(moves, list) or \
    isinstance(moves, np.ndarray), "a 2D grid distribution is required"
    # Convert moves into a numpy array if not already
    if isinstance(moves, tuple) or isinstance(moves, list):
        moves = np.array(moves)
    # Check that senses is valid
    assert isinstance(senses, tuple) or isinstance(senses, list) or \
    isinstance(senses, np.ndarray), "a 2D grid distribution is required"
    # Convert moves into a numpy array if not already
    if isinstance(senses, tuple) or isinstance(senses, list):
        senses = np.array(senses)
    # Check that p_move is a probability
    assert p_move <= 1 and p_move >= 0, "p_move is a probability"
    # Check that p_sense is a probability
    assert p_sense <= 1 and p_sense >= 0, "p_move is a probability"
    # Check that action is boolean
    assert isinstance(action, bool), "move - True, sense - False"
    # Check that rnd is boolean
    assert isinstance(rnd, bool), "use setup - True"
    # Check kwargs
    if not rnd:
        assert len(kwargs) == 2, "we are expecting colors and probs only"
    
    # Decide whether to use the setup function
    # colors, probs = [], []
    if rnd:
        colors, probs = setup(p_move, p_sense, action)
    else:
        # Check the names are corerct
        # This could have been done in the assert section as well
        try:
            colors = kwargs['colors']
            probs = kwargs['probs']
        except:
            print("use args 'colors=' and 'probs='")
            sys.exit(1)
    
    for move, sense in zip(moves, senses):
        probs = localise(probs, colors, p_move, p_sense, move, sense, action)
    
    return probs