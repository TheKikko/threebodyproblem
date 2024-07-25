import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib as mpl

from itertools import product


def gravity(position1, position2, mass1, mass2, gravityConstant = 6.67430e-11):
    #print(f"Points: {position1}; {position2}. Masses: {mass1}; {mass2}")
    masses = mass1 * mass2
    radius = np.linalg.norm(position1 - position2)
    print(f"Masses {masses} radius {radius}")
    return gravityConstant * masses / np.pow(radius, 2)



nBodies = 3
nDimensions = 2
nIterations = int(1e2)

collision = False
lengthUniverse = 1e1
massScaling = 5e1

iteration = 0
dt = 1e-1

rng = np.random.default_rng()  

initialPositions = rng.random((nBodies, nDimensions)) * lengthUniverse # e.g. of size 3, 2
#print(initialPositions)
currentPositions = np.empty((nBodies, nDimensions, nIterations+1))
currentPositions[:, :, 0] = initialPositions
#print(currentPositions[:, :, 0])
masses = rng.random((nBodies,)) * massScaling
initialVelocities = np.empty(initialPositions.shape)
currentVelocities = np.empty(currentPositions.shape)

pairs = [(0, 1), (0, 2), (1, 2)] # TODO: generalize to nBody?
indices = range(0, nBodies)
pairs = [(i, j) for i, j in product(indices, indices) if i < j]
#print(pairs)

fig, ax = plt.subplots(nrows=1, ncols=1)

while not collision and iteration < nIterations:
    print(f"Iteration {iteration} of {nIterations}:")
    for i, j in pairs:
        f = gravity(currentPositions[i, :, iteration], currentPositions[j, :, iteration], masses[i], masses[j], 1) # natural units
        print(f"Gravity force between bodies {i} and {j} is: {f}")
        print(f"Current position: \n  {i}: ({currentPositions[i,0,iteration]}, {currentPositions[i,1,iteration]})")
        print(f"  {j}: ({currentPositions[j,0,iteration]}, {currentPositions[j,1,iteration]})")
        # TODO: double check math
        currentVelocities[i, :, iteration+1] = currentVelocities[i, :, iteration] - dt * f
        currentVelocities[j, :, iteration+1] = currentVelocities[j, :, iteration] - dt * f
    for i in range(nBodies):
        currentPositions[i, :, iteration+1] = currentPositions[i, :, iteration] + dt * currentVelocities[i, :, iteration+1]
    iteration += 1


def init():
    min_x = np.min(currentPositions[:, 0, :])
    min_y = np.min(currentPositions[:, 1, :])
    max_x = np.max(currentPositions[:, 0, :])
    max_y = np.max(currentPositions[:, 1, :])
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)


def animate(indexFrame):
    ax.cla()
    colors = ['b', 'g', 'r', 'y', 'k']

    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    time_template = 'time = %.1fs'

    # assumes nDimension = 2
    for i in range(nBodies):
        thisx = [currentPositions[i, 0, indexFrame]]
        thisy = [currentPositions[i, 1, indexFrame]]

        history_x = currentPositions[i, 0, :indexFrame]
        history_y = currentPositions[i, 1, :indexFrame]

        line, = ax.plot(history_x, history_y, '-', lw=2, alpha=0.2, color=colors[i])

        time_text.set_text(time_template % (indexFrame * dt))
        ax.scatter(currentPositions[i, 0, indexFrame-1], 
                   currentPositions[i, 1, indexFrame-1], color=colors[i])
    #return lines, traces, time_text


ani = animation.FuncAnimation(
    fig, animate, nIterations+1, interval=dt*1000, blit=False)

#fig, ax = plt.subplots(nrows=1, ncols=1)
#ax.scatter(initialPositions[:, 0], initialPositions[:, 1])

plt.show()



