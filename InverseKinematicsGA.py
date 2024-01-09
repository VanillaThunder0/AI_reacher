import pygad
import gymnasium as gym
import time
import math
import sys
import os


sys.stdout.reconfigure(line_buffering=True)
ENV_SEED = 1
GA_SEED = 1

NR_OF_TESTING_ROUNDS_PER_INDIVIDUAL = 10


function_inputs = [1,1,1,1,1,1] # Function inputs. k1,d1,k2,d2,i1,i2

num_generations = 200 # Number of generations.
num_parents_mating = 40 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 100 # Number of solutions in the population.
num_genes = len(function_inputs) # Number of paramters in the solution


def limit_action(action):
    out = []
    for a in action:
        if a>1:
            out.append(1)
        elif a<-1:
            out.append(-1)
        else:
            out.append(a)
    return out
l1 = 0.1
l2 = 0.1


def fitness_func(ga_instance, solution, solution_idx):
    global l1,l2 
    fitness = 0
    k1 = solution[0] # Extract current parameters from solution
    d1 = solution[1]
    k2 = solution[2]
    d2 = solution[3]
    i1 = solution[4]
    i2 = solution[5]
    for j in range(NR_OF_TESTING_ROUNDS_PER_INDIVIDUAL):
        e1_prev = 0
        e2_prev = 0
        e1_sum = 0 
        e2_sum = 0
        env = gym.make("Reacher-v4")
        observation,info = env.reset(seed=j)
        for i in range(50):
            q1_c = observation[0]
            q1_s = observation[2]
            q2_c = observation[1]
            q2_s = observation[3]
            q1 = (math.atan2(q1_s,q1_c)+2*math.pi)%(math.pi*2) # Current angle of the two joints
            q2 = (math.atan2(q2_s,q2_c)+2*math.pi)%(math.pi*2)#math.acos(observation[1]) 
            x = observation[4]
            y = observation[5]

            # Inverse kinematics to get desired angles
            q2_d = ((math.acos((x**2+y**2-l1**2-l2**2)/(2*l1*l2)))+2*math.pi)%(math.pi*2)
            q1_d = ((math.atan2(y,x)-math.atan2(l2*math.sin(q2_d),l1+l2*math.cos(q2_d)))+2*math.pi)%(math.pi*2)
            # Error
            e1 = q1_d-q1
            e2 = q2_d-q2
            # Integrated error
            e1_sum += e1
            e2_sum += e2
            # Constrain error
            if(e2<-math.pi):
                e2 += 2*math.pi
            elif(e2>math.pi):
                e2 -= 2*math.pi
            if(e1<-math.pi):
                e1 += 2*math.pi
            elif(e1>math.pi):
                e1 -= 2*math.pi
            # Derivative of error
            e1_d = e1-e1_prev
            e2_d = e2-e2_prev
            # New actions
            a1 = k1*e1+d1*e1_d+i1*e1_sum
            a2 = k2*e2+d2*e2_d+i2*e2_sum
            #Limited to [-1,1]
            a = limit_action([a1,a2])
            # Update previous error
            e1_prev = e1
            e2_prev = e2
            # Step and collect reward
            observation,reward,terminated,truncated,info = env.step(a)
            fitness += reward
            if terminated:
                break
            elif truncated:
                break
        env.close()
    return fitness

last_fitness = 0
t_last = time.time()
g_last = 0
def on_generation(ga_instance): # Callback function printing progress
    pass
    # global t_last
    # global last_fitness
    # global g_last
    # generation = ga_instance.generations_completed
    # print(f"Generation = {generation}")
    # print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    # change = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness
    # print(f"Change     = {change}")
    # if(change>0):
    #     g_last = generation
    # print("Last generation with change: ",g_last)
    # # print("k,d: ",ga_instance.best_solution()[0])
    # last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    # print("Generation time: ",time.time()-t_last)
    # t_last = time.time()

crossover_types = ["single_point", "two_points", "uniform", "scattered", None]
mutation_types = ["random", "swap", "inversion", "scramble", None]
best_fitness = -1000
best_mutation = ""
best_crossover = ""
table = []

# Make directory to save models in
dir_name = "results_"+"Gen:"+str(num_generations)+"ParentsMating:"+str(num_parents_mating)+"SolPerPop:"+str(sol_per_pop)
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, dir_name)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)




print("------ Start ------")
print("Generations to train for", num_generations)
print("Number of parents mating",num_parents_mating)
print("Solutions per population", sol_per_pop)

for i, crossover_type in enumerate(crossover_types):
    for j, mutation_type in enumerate(mutation_types): 
        print("")
        # Create the GA instance
        ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            sol_per_pop=sol_per_pop,
                            num_genes=num_genes,
                            fitness_func=fitness_func,
                            on_generation=on_generation,
                            save_solutions=False, 
                            random_seed=GA_SEED,
                            crossover_type=crossover_type,
                            mutation_type=mutation_type,
                            mutation_probability=0.3,
                            keep_elitism=0,
                            parent_selection_type="tournament",
                            K_tournament=5)
        # Running the GA to optimize the parameters of the function.
        ga_instance.run()

        #ga_instance.plot_fitness()
        #ga_instance.plot_genes()

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        print(f"Crossover type: {crossover_type}  and  mutation type: {mutation_type} ")
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Parameters of the best solution : {solution}")
        print(f"Index of the best solution : {solution_idx}")
        print(f"{(time.time()-t_last)/60} minutes since start")

        table.append((str(crossover_type) + " " + str(mutation_type), solution_fitness))

        if solution_fitness > best_fitness:
            best_fitness = solution_fitness
            best_mutation = mutation_type
            best_crossover = crossover_type
             # Saving the GA instance.
        filename = "results_"+"Gen:"+str(num_generations)+"ParentsMating:"+str(num_parents_mating)+"SolPerPop:"+str(sol_per_pop)+"/"+str(crossover_type)+str(mutation_type) # The filename to which the instance is saved. The name is without extension.
        ga_instance.save(filename=filename)

        if ga_instance.best_solution_generation != -1:
            print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

        
print("")
print("")
print("------ Done ------")
print(f"Best combination is {best_crossover} with {best_mutation}")
print("")
print("")
for entry in table:
    print(entry)


# Loading the saved GA instance.
# loaded_ga_instance = pygad.load(filename=filename)


# ## Visualization of the best solution:
# env = gym.make("Reacher-v4", render_mode="human")
# observation, info = env.reset(seed=42)
# l1 = 0.1
# l2 = 0.1
# e1_prev = 0
# e2_prev = 0
# e1_sum = 0
# e2_sum = 0
# dt = 0.02
# a = [0,0]
# k1 = ga_instance.best_solution()[0][0]
# d1 = ga_instance.best_solution()[0][1]
# k2 = ga_instance.best_solution()[0][2]
# d2 = ga_instance.best_solution()[0][3]
# i1 = ga_instance.best_solution()[0][4]
# i2 = ga_instance.best_solution()[0][5]
# print("k1,k2,d1,d2,i1,i2",k1,k2,d1,d2,i1,i2)
# while True:
#     observation, reward, terminated, truncated, info = env.step(a)
#     q1_c = observation[0]
#     q1_s = observation[2]
#     q2_c = observation[1]
#     q2_s = observation[3]
#     q1 = (math.atan2(q1_s,q1_c)+2*math.pi)%(math.pi*2)
#     q2 = (math.atan2(q2_s,q2_c)+2*math.pi)%(math.pi*2)#math.acos(observation[1])
#     x_c = observation[8]+observation[4]
#     y_c = observation[9]+observation[5]
#     x = observation[4]
#     y = observation[5]

#     q2_d = ((math.acos((x**2+y**2-l1**2-l2**2)/(2*l1*l2)))+2*math.pi)%(math.pi*2)
#     q1_d = ((math.atan2(y,x)-math.atan2(l2*math.sin(q2_d),l1+l2*math.cos(q2_d)))+2*math.pi)%(math.pi*2)

#     e1 = q1_d-q1
#     e2 = q2_d-q2
#     if(e2<-math.pi):
#         e2 += 2*math.pi
#     elif(e2>math.pi):
#         e2 -= 2*math.pi
#     if(e1<-math.pi):
#         e1 += 2*math.pi
#     elif(e1>math.pi):
#         e1 -= 2*math.pi
#     e1_d = e1-e1_prev
#     e2_d = e2-e2_prev
#     a1 = k1*e1+d1*e1_d+i1*e1_sum
#     a2 = k2*e2+d2*e2_d+i2*e2_sum
#     a = limit_action([a1,a2])
#     e1_prev = e1
#     e2_prev = e2
#     if terminated:
#         e1_prev = 0
#         e2_prev = 0
#         e1_sum = 0
#         e2_sum = 0
#         observation, info = env.reset()
#     elif truncated:
#         e1_prev = 0
#         e2_prev = 0
#         e1_sum = 0
#         e2_sum = 0
#         observation, info = env.reset()
