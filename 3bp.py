import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from tqdm import tqdm

def gravity(position1, position2, mass1, mass2, gravityConstant = 6.67430e-11):
    masses = mass1 * mass2
    r_vec = position2 - position1
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:
        return np.zeros_like(position1)
    r_hat = r_vec / r_mag
    return gravityConstant * masses / r_mag ** 2 * r_hat

nBodies = 3
nDimensions = 2
nIterations = int(8e2)
finalIteration = nIterations

collision = False
collisionTolerance = 5e-1

iteration = 0
dt = 0.1

lengthUniverse = 40
massScaling = 150
velocitySpan = 5

rng = np.random.default_rng()  

initialPositions = rng.random((nBodies, nDimensions)) * lengthUniverse
currentPositions = np.zeros((nBodies, nDimensions, nIterations+1))
currentPositions[:, :, 0] = initialPositions

masses = rng.random((nBodies,)) * massScaling

initialVelocities = -velocitySpan + rng.random(initialPositions.shape) * 2 * velocitySpan
currentVelocities = np.zeros(currentPositions.shape)
currentVelocities[:, :, 0] = initialVelocities

fig, ax = plt.subplots(nrows=1, ncols=1)

progress_bar = tqdm(total=nIterations, desc='Simulation 3-body problem...')
while not collision and iteration < nIterations:
    forces = np.zeros((nBodies, nDimensions))
    for i in range(nBodies):
        for j in range(nBodies): 
            if i != j: 
                distance = np.linalg.norm(currentPositions[i, :, iteration] - currentPositions[j, :, iteration])
                if distance < collisionTolerance:
                    collision = True
                    finalIteration = iteration
                    collisionPosition = currentPositions[i, :, iteration].copy()
                f = gravity(currentPositions[i, :, iteration], currentPositions[j, :, iteration], masses[i], masses[j], 1) # natural units
                forces[i] += f
    for i in range(nBodies):
        currentVelocities[i, :, iteration+1] = currentVelocities[i, :, iteration] + dt * forces[i]/masses[i]
        currentPositions[i, :, iteration+1] = currentPositions[i, :, iteration] + dt * currentVelocities[i, :, iteration+1]

    if not collision:
        lastValidPositions = currentPositions[:, :, iteration+1].copy()
    iteration += 1
    progress_bar.update(1)

progress_bar.close()

if collision:
    for i in range(iteration, nIterations+1):
        currentPositions[:, :, i] = lastValidPositions

def init():
    ax.set_xlim(-lengthUniverse, lengthUniverse)
    ax.set_ylim(-lengthUniverse, lengthUniverse)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('3-Body Problem Simulation')

    global scatters, lines
    scatters = [ax.scatter([], [], color=color, s=50) for color in ['b', 'g', 'r', 'y', 'k']]
    lines = [ax.plot([], [], '-', lw=2, alpha=0.5, color=color)[0] for color in ['b', 'g', 'r', 'y', 'k']]
    
    return scatters + lines

def animate(indexFrame):
    if indexFrame > 0:
        for i in range(nBodies):
            history_x = currentPositions[i, 0, :indexFrame]
            history_y = currentPositions[i, 1, :indexFrame]

            lines[i].set_data(history_x, history_y)
            scatters[i].set_offsets([currentPositions[i, 0, indexFrame-1], currentPositions[i, 1, indexFrame-1]])

        ax.set_xlim(np.min(currentPositions[:, 0, :indexFrame])-5, np.max(currentPositions[:, 0, :indexFrame])+5)
        ax.set_ylim(np.min(currentPositions[:, 1, :indexFrame])-5, np.max(currentPositions[:, 1, :indexFrame])+5)

    if collision and indexFrame >= finalIteration:
        ax.text(collisionPosition[0], collisionPosition[1] + 1, 'Collision!', color='red', fontsize=12, ha='center')

    return scatters + lines

print("Creating animation...")
ani = animation.FuncAnimation(
    fig, animate, nIterations+1, init_func=init, interval=dt*500, blit=True)

f = r"/home/kikko/repos/3bodyproblem/animation.gif"
writergif = animation.PillowWriter(fps=30)
print("Saving animation...")
ani.save(f, writer=writergif)
print("Done.")

plt.show()

