import gymnasium as gym
import random
import torch
import torch.nn as nn
from itertools import cycle
from copy import deepcopy
import gymnasium as gym
import numpy as np

POPULATION_SIZE = 100
INPUT_SIZE = 11
OUTPUT_SIZE = 2
LEARNING_RATE = 0.015
AMOUNT_OF_ANCESTORS = 10

# Individual is a potential solution for the reacher problem -> a simple feedforward NN
class Individual(nn.Module):
    def __init__(self, neuronAmount):
        super().__init__()
        self.neuronAmount = neuronAmount
        self.fitness = 0
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(INPUT_SIZE, self.neuronAmount)
        self.layer2 = nn.Linear(self.neuronAmount, self.neuronAmount)
        self.layer3 = nn.Linear(self.neuronAmount, OUTPUT_SIZE)
        self.environment = gym.make("Reacher-v4")
        self.observation, self.info = self.environment.reset()

    # Takes a population and find the fittest
    @classmethod
    def findFittest(self, oldPopulation, amountOfAncestors):
        newPopulation = []
        newPopulation = sorted(oldPopulation, key = lambda x:x.fitness, reverse=True)
        return newPopulation[0:amountOfAncestors]
    
    # Makes a new population based on the fittest individuals from previous run
    @classmethod
    def findNewPopulation(self, listOfFittest):
        newPopulation = listOfFittest
        amountNewcommers = POPULATION_SIZE - len(listOfFittest)

        index = 0
        # Iterates through the fittest, mutates them, and appends to new population. The fittest will be present in new population
        for _ in range(amountNewcommers):

            temp = deepcopy(listOfFittest[index])
            temp.mutate()
            newPopulation.append(temp)

            if index == len(listOfFittest)-1:
                index = 0

            index += 1

        for individual in newPopulation:
            individual.fitness = 0
        return newPopulation
    
    @classmethod
    def testPopulation(self, population):
        seed = random.randint(0, 1000)
        for individual in population:
            individual.run(seed)

    @classmethod
    def sumFitness(self, population):
        sum = 0
        for individual in population:
            sum += individual.fitness
        
        return sum
    
    @classmethod
    def uniformCrossover(self, population):

        new_population = []
        
        with torch.no_grad():
        # Mutate each individual in the list with a random other guy from the list
            for index, individual in enumerate(population):
                # Remove the current index from the list of possible other dudes to crossover with
                list_ = list(range(len(population))) 
                del list_[index]

                # Get the random other dude to crossover with
                partner = deepcopy(population[random.choice(list_)]) 
                # Make a copy of the current individual 
                current_individual = deepcopy(individual)

                # Go through each layer and choose a gene from either individual randomly
                for i, neuronConnection in enumerate(current_individual.layer1.weight):
                    for j, _ in enumerate(neuronConnection):
                        current_individual.layer1.weight[i, j] = random.choice([current_individual.layer1.weight[i, j], partner.layer1.weight[i, j]])

                for i, neuronConnection in enumerate(current_individual.layer2.weight):
                    for j, _ in enumerate(neuronConnection):
                         current_individual.layer2.weight[i, j] = random.choice([current_individual.layer2.weight[i, j], partner.layer2.weight[i, j]])

                for i, neuronConnection in enumerate(current_individual.layer3.weight):
                    for j, _ in enumerate(neuronConnection):
                         current_individual.layer3.weight[i, j] = random.choice([current_individual.layer3.weight[i, j], partner.layer3.weight[i, j]])

                new_population.append(current_individual)
        
        return new_population

    @classmethod
    def singePointCrossover(self):
        print("")

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = nn.functional.tanh(x)
        x = self.layer2(x)
        x = nn.functional.tanh(x)
        x = self.layer3(x)
        return torch.nn.functional.tanh(x)

    def printWeights(self):
        print(f"Layer 1 {self.layer1.weight}")
        print("")
        print(f"Layer 2 {self.layer2.weight}")

    # Goes throug both layers of the individual and changes their weight according to LEARNING_RATE
    def mutate(self):
        with torch.no_grad():
            for i, neuronConnection in enumerate(self.layer1.weight):
                for j, _ in enumerate(neuronConnection):
                    self.layer1.weight[i, j] = self.layer1.weight[i, j] +  random.uniform(-LEARNING_RATE, LEARNING_RATE)
            
            for i, neuronConnection in enumerate(self.layer2.weight):
                for j, _ in enumerate(neuronConnection):
                    self.layer2.weight[i, j] = self.layer2.weight[i, j] + random.uniform(-LEARNING_RATE, LEARNING_RATE)

            for i, neuronConnection in enumerate(self.layer3.weight):
                for j, _ in enumerate(neuronConnection):
                    self.layer3.weight[i, j] = self.layer3.weight[i, j] + random.uniform(-LEARNING_RATE, LEARNING_RATE)

    def run(self, seed):
        for j in range(10):
            for x in range(50):
                # input = [self.observation[8], self.observation[9]]
                action = self((torch.Tensor([self.observation])))
                action = action.detach().numpy()
                [action] = action
                self.observation, reward, terminated, truncated, info = self.environment.step(action)
                self.fitness += reward

                if terminated:
                    self.observation, self.info = self.environment.reset(seed=seed)
                    break
                elif truncated:
                    self.fitness -= 10
                    # niggative reward here
                    self.observation, self.info = self.environment.reset(seed=seed)
                    break
            self.observation, self.info = self.environment.reset(seed=seed)

    def runLong(self, seed):
        for x in range(250):
            # input = [self.observation[8], self.observation[9]]
            action = self((torch.Tensor([self.observation])))
            action = action.detach().numpy()
            [action] = action
            self.observation, reward, terminated, truncated, info = self.environment.step(action)
            self.fitness += reward

            if terminated:
                self.fitness += 10
                self.observation, self.info = self.environment.reset(seed=seed)
                break
            elif truncated:
                self.observation, self.info = self.environment.reset(seed=seed)
                break

        self.observation, self.info = self.environment.reset(seed=seed)

                  