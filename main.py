from ReacherGA import Individual, POPULATION_SIZE, AMOUNT_OF_ANCESTORS
import gymnasium as gym
import random

population = [Individual(neuronAmount=20) for x in range(POPULATION_SIZE)]

for x in range(10):
    Individual.testPopulation(population)    
    print("Iteration: ", x, "   Fitness. ",Individual.sumFitness(population))
    fittest = Individual.findFittest(population, AMOUNT_OF_ANCESTORS)
    population = Individual.findNewPopulation(fittest)


individual = sorted(population, key = lambda x:x.fitness, reverse=True)[0]
individual.printWeights()
individual.environment = gym.make("Reacher-v4", render_mode="human").env
individual.observation, individual.info = individual.environment.reset()


while(1):
    seed = random.randint(0, 1000)
    individual.runLong(seed)