#!/usr/bin/env python3
import pygame
from shapely.geometry import *
from shapely.affinity import *
from math import *
from shapely.ops import nearest_points
from shapely.geometry.polygon import LinearRing
import numpy as np
#import keras
import sys
from copy import deepcopy
from tqdm import tqdm
import json
import codecs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import time
import datetime
from collections import deque

successGates = []
numOfOutputs = 5
numOfInputs = 24
death_penalty = -1
LEARNING_RATE=0.005
visualization_scale_multiplier = 60

np.random.seed(123)
verbose = 0
print("Initializing track")
# trackball = [(0,0),(0,4),(0,8),(0,12),(4,12),(8,12),(12,12),(12,8),(12,4),(12,0),(8,0),(4,0),(0,0)]
trackball = [(0,0),(0, 4), (0, 8), (2, 12), (2, 16), (0, 20), (-2, 24), (-6, 28), (-2,32), (0, 32), (4, 32), (8, 32), (12, 32), (16,32), (20, 32), (20, 28), (20,24),(20,20),(20,16),(21, 17), (21,24),(21,28),(21,32),(24, 32), (28,32), (32,32), (36,32), (36,28),(36, 24),(36,20),(36, 16), (36,12),(36, 8), (36, 4), (36, 0), (32, 0), (28, 0), (24, 0), (20, 0), (16, 0), (12, 0), (8, 0), (4, 0), (0,0)]

ring = LinearRing(trackball)

# trackball2 = [(4,4),(4,8),(8,8),(8,4),(4,4)]
trackball2 = [(4, 4), (4, 8), (8, 12), (8, 16), (4, 20), (4, 24), (0, 28), (4, 28), (8, 28), (12, 24), (12, 20), (12, 16), (16, 12), (12, 8), (16, 8), (20, 8), (24, 8), (28, 8), (28, 12), (28, 16), (28, 20), (28, 24), (28, 28), (32,28), (32, 4), (4, 4)]

ring2 = LinearRing(trackball2)
#trackball = [(0,0),(0, 4), (0, 8), (2, 12), (2, 16), (0, 20), (-2, 24), (-6, 28), (-2,32), (0, 32), (4, 32), (8, 32), (12, 32), (16,32), (20, 32), (20, 28), (20,24),(20,20),(20,16),(21, 17), (21,24),(21,28),(21,32),(24, 32), (28,32), (32,32), (36,32), (36,28),(36, 24),(36,20),(36, 16), (36,12),(36, 8), (36, 4), (36, 0), (32, 0), (28, 0), (24, 0), (20, 0), (16, 0), (12, 0), (8, 0), (4, 0), (0,0)]
# ring = LinearRing(trackball)

# trackball2 = [(4,4),(4,8),(8,8),(8,4),(4,4)]

X_range = (min([min([a_tuple[0] for a_tuple in trackball]),min([a_tuple[0] for a_tuple in trackball2])]), max([max([a_tuple[0] for a_tuple in trackball]),max([a_tuple[0] for a_tuple in trackball2])]))
y_range = (min([min([a_tuple[1] for a_tuple in trackball]),min([a_tuple[1] for a_tuple in trackball2])]), max([max([a_tuple[1] for a_tuple in trackball]),max([a_tuple[1] for a_tuple in trackball2])]))
diag_distance_of_track = ((y_range[0] - y_range[1])**2 + (X_range[0] - X_range[1])**2)**(1/2)
masterGatesToDo = []
vision_array_preprocessed = []
directionArray = [["w"],["a"],["d"],["w","a"],["w","d"],["s"],["s","a"],["s","d"]]
schotastic_action_preprocessed_range = range(0,numOfOutputs)
# schotastic_action_preprocessed_probability_array = [exploitation]
# for i in range(0, numOfOutputs-1):
#     schotastic_action_preprocessed_probability_array.append(exploration/(numOfOutputs-1))


for i in range(0,numOfInputs):
    vision_array_preprocessed.append(i*(360/(numOfInputs)))

for i in trackball[:-1]:
    successGates.append(LineString([Point(i), Point(nearest_points(Point(i),ring2)[1])]))
successGates.append(successGates.pop(0))
successGates.append(successGates.pop(0))
tempGates = []

for i in range(0,12):
    tempGates.append(successGates[i])
for i in range(14,20):
    tempGates.append(successGates[i])
tempGates.append(successGates[13])
tempGates.append(successGates[12])
for i in range(20,len(successGates)):
    tempGates.append(successGates[i])
tempGates.pop(16)
successGates = tempGates

successGatesLength = len(successGates)

for i in range(0,successGatesLength):
    masterGatesToDo.append(i)

scaler = MinMaxScaler(feature_range=(0, 1))


pygame.init
pygame.display.set_caption("DRL CAR")
car_img = pygame.image.load("images/car.png")
screen = pygame.display.set_mode(((X_range[1] - X_range[0]) * visualization_scale_multiplier,(y_range[1] - y_range[0]) * visualization_scale_multiplier))
car_img = pygame.transform.scale(car_img,(visualization_scale_multiplier,visualization_scale_multiplier))
screen.fill((255,255,255))
screen.blit(car_img,(20,60))
pygame.display.update()
print("success")
time.sleep(2)

