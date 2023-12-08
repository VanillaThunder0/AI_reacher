from ReacherGA import Individual, POPULATION_SIZE, AMOUNT_OF_ANCESTORS
import gymnasium as gym
import random
EPOCH = 10
population = [Individual(neuronAmount=10) for x in range(POPULATION_SIZE)]

for x in range(EPOCH):

    Individual.testPopulation(population)    
    print("Iteration: ", x, "   Fitness. ",Individual.sumFitness(population))
    fittest = Individual.findFittest(population, AMOUNT_OF_ANCESTORS)
    crossovered_population = Individual.uniformCrossover(fittest)
    population = Individual.findNewPopulation(crossovered_population)


individual = sorted(population, key = lambda x:x.fitness, reverse=True)[0]
individual.printWeights()
individual.environment = gym.make("Reacher-v4", render_mode="human").env
individual.observation, individual.info = individual.environment.reset()


while(1):
    seed = random.randint(0, 1000)
    individual.runLong(seed)











# bobby = Individual(neuronAmount=1)
# geh = Individual(neuronAmount=1)
# fuck = Individual(neuronAmount=1)
# print("Bobby genes: ")
# print(bobby.layer1.weight)
# print(bobby.layer2.weight)
# print("")
# print("------------------")
# print("geh genes: ")
# print(geh.layer1.weight)
# print(geh.layer2.weight)
# print("")
# print("------------------")
# print("fuck genes: ")
# print(fuck.layer1.weight)
# print(fuck.layer2.weight)
# print("")
# print("------------------")
# print("Crossover genes:")
# crossover = Individual.uniformCrossover([bobby, geh, fuck])
# print("1 genes: ")
# print(crossover[0].layer1.weight)
# print(crossover[0].layer2.weight)
# print("")
# print("------------------")
# print("2 genes: ")
# print(crossover[1].layer1.weight)
# print(crossover[1].layer2.weight)
# print("")
# print("------------------")
# print("3 genes: ")
# print(crossover[2].layer1.weight)
# print(crossover[2].layer2.weight)