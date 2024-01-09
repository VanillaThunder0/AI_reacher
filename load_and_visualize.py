import pygad
import gymnasium as gym
import time
import math

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


# Loading the saved GA instance.
loaded_ga_instance = pygad.load(filename="results200gen10Mating100SolPerPop/geneticsingle_pointrandom")




## Visualization of the best solution:
env = gym.make("Reacher-v4", render_mode="human")
observation, info = env.reset(seed=42)
l1 = 0.1
l2 = 0.1
e1_prev = 0
e2_prev = 0
e1_sum = 0
e2_sum = 0
dt = 0.02
a = [0,0]
k1 = loaded_ga_instance.best_solution()[0][0]
d1 = loaded_ga_instance.best_solution()[0][1]
k2 = loaded_ga_instance.best_solution()[0][2]
d2 = loaded_ga_instance.best_solution()[0][3]
i1 = loaded_ga_instance.best_solution()[0][4]
i2 = loaded_ga_instance.best_solution()[0][5]
print("k1,k2,d1,d2,i1,i2",k1,k2,d1,d2,i1,i2)
while True:
    observation, reward, terminated, truncated, info = env.step(a)
    q1_c = observation[0]
    q1_s = observation[2]
    q2_c = observation[1]
    q2_s = observation[3]
    q1 = (math.atan2(q1_s,q1_c)+2*math.pi)%(math.pi*2)
    q2 = (math.atan2(q2_s,q2_c)+2*math.pi)%(math.pi*2)#math.acos(observation[1])
    x_c = observation[8]+observation[4]
    y_c = observation[9]+observation[5]
    x = observation[4]
    y = observation[5]

    q2_d = ((math.acos((x**2+y**2-l1**2-l2**2)/(2*l1*l2)))+2*math.pi)%(math.pi*2)
    q1_d = ((math.atan2(y,x)-math.atan2(l2*math.sin(q2_d),l1+l2*math.cos(q2_d)))+2*math.pi)%(math.pi*2)

    e1 = q1_d-q1
    e2 = q2_d-q2
    if(e2<-math.pi):
        e2 += 2*math.pi
    elif(e2>math.pi):
        e2 -= 2*math.pi
    if(e1<-math.pi):
        e1 += 2*math.pi
    elif(e1>math.pi):
        e1 -= 2*math.pi
    e1_d = e1-e1_prev
    e2_d = e2-e2_prev
    a1 = k1*e1+d1*e1_d+i1*e1_sum
    a2 = k2*e2+d2*e2_d+i2*e2_sum
    a = limit_action([a1,a2])
    e1_prev = e1
    e2_prev = e2
    if terminated:
        e1_prev = 0
        e2_prev = 0
        e1_sum = 0
        e2_sum = 0
        observation, info = env.reset()
    elif truncated:
        e1_prev = 0
        e2_prev = 0
        e1_sum = 0
        e2_sum = 0
        observation, info = env.reset()
