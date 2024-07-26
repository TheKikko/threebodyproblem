import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from functools import partial

from tqdm import tqdm

import json


def gravity(position1, position2, mass1, mass2, gravityConstant=6.67430e-11):
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
nIterations = int(2e2)

collisionTolerance = 5e-1
outOfReachTolerance = 1

dt = 0.1

lengthUniverse = 40
massScaling = 250
velocitySpan = 5

rng = np.random.default_rng()

initialPositions = rng.random((nBodies, nDimensions)) * lengthUniverse
currentPositions = np.zeros((nBodies, nDimensions, nIterations + 1))
currentPositions[:, :, 0] = initialPositions

masses = rng.random((nBodies,)) * massScaling

initialVelocities = -velocitySpan + rng.random(initialPositions.shape) * 2 * velocitySpan
currentVelocities = np.zeros(currentPositions.shape)
currentVelocities[:, :, 0] = initialVelocities


def saveFinalPosition(iteration, position):
    return iteration, position.copy()

def simulate(nIterations):
    progress_bar = tqdm(total=nIterations, desc='Simulation 3-body problem...')

    collision = False
    outOfReach = False
    iteration = 0
    finalIteration = nIterations
    finalPosition = np.zeros(currentPositions[0,:,0].shape)
    while not (collision or outOfReach) and iteration < nIterations:
        forces = np.zeros((nBodies, nDimensions))
        for i in range(nBodies):
            for j in range(nBodies):

                # bodies shouldn't impact themselves
                if i != j:
                    distance = np.linalg.norm(
                        currentPositions[i, :, iteration] -
                        currentPositions[j, :, iteration])
                    if distance < collisionTolerance:
                        collision = True
                        finalIteration, finalPosition = saveFinalPosition(iteration, currentPositions[i, :, iteration])

                    f = gravity(currentPositions[i, :, iteration], currentPositions[j, :, iteration], masses[i],
                                masses[j], 1)  # natural units
                    forces[i] += f

        for i in range(nBodies):
            currentVelocities[i, :, iteration + 1] = currentVelocities[i, :, iteration] + dt * forces[i] / masses[i]
            currentPositions[i, :, iteration + 1] = currentPositions[i, :, iteration] + dt * currentVelocities[i, :,
                                                                                           iteration + 1]
            if np.linalg.norm(forces[i]) < outOfReachTolerance:
                outOfReach = True
                finalIteration, finalPosition = saveFinalPosition(iteration, currentPositions[i, :, iteration])

        if not (collision or outOfReach):
            lastValidPositions = currentPositions[:, :, iteration + 1].copy()
        iteration += 1
        progress_bar.update(1)

    progress_bar.close()

    if collision or outOfReach:
        for i in range(iteration, nIterations + 1):
            currentPositions[:, :, i] = lastValidPositions
    return currentPositions, collision, outOfReach, finalIteration, finalPosition


def init():
    ax.set_xlim(-lengthUniverse, lengthUniverse)
    ax.set_ylim(-lengthUniverse, lengthUniverse)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    global scatters, lines
    scatters = [ax.scatter([], [], color=color, s=50) for color in colors]
    lines = [ax.plot([], [], '-', lw=2, alpha=0.5, color=color)[0] for color in colors]

    return scatters + lines


def animate(currentPositions, collision, outOfReach, finalIteration, finalPosition, indexFrame):
    # later this will be filled in with how long time has passed
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    time_template = f'time = %.1fs'

    for line in lines:
        line.set_data([], [])
    for scatter in scatters:
        scatter.set_offsets(np.empty((0, 2)))
    for text in ax.texts:
        text.remove()
    if indexFrame > 0:
        for i in range(nBodies):
            history_x = currentPositions[i, 0, :indexFrame]
            history_y = currentPositions[i, 1, :indexFrame]

            lines[i].set_data(history_x, history_y)
            scatters[i].set_offsets(
                [currentPositions[i, 0, indexFrame - 1], currentPositions[i, 1, indexFrame - 1]])

        ax.set_xlim(np.min(currentPositions[:, 0, :indexFrame]) - 5, np.max(currentPositions[:, 0, :indexFrame]) + 5)
        ax.set_ylim(np.min(currentPositions[:, 1, :indexFrame]) - 5, np.max(currentPositions[:, 1, :indexFrame]) + 5)

    if collision and indexFrame >= finalIteration:
        ax.text(finalPosition[0], finalPosition[1] + 1, 'Collision!', color='red', fontsize=12, ha='center')
    if outOfReach and indexFrame >= finalIteration:
        ax.text(0.5, 0.5, 'Simulation stopped: Body left the reach of others', color='red', fontsize=12, ha='center')
    time_text.set_text(time_template % (indexFrame * dt))
    if indexFrame < finalIteration: 
        ax.set_title(f'3-Body Problem Simulation at time {indexFrame * dt:.1f}s')
    else:
        ax.set_title(f'3-Body Problem Simulation at time {finalIteration * dt:.1f}s')


    return scatters + lines


fig, ax = plt.subplots(nrows=1, ncols=1)
cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in np.linspace(0, 1, nBodies)]
legends = [ax.scatter([], [], color=colors[i], s=50, label=f'Body {i + 1}, Mass {masses[i]:.2f}') for i in
           range(nBodies)]
ax.legend()
positions, collision, outOfReach, finalIteration, finalPosition = simulate(nIterations)

anim = partial(animate, positions, collision, outOfReach, finalIteration, finalPosition)


print("Creating animation...")
ani = animation.FuncAnimation(
    fig, anim, nIterations+1, init_func=init, interval=dt*500, blit=True)

f = r"/home/kikko/repos/3bodyproblem/animation.gif"
writergif = animation.PillowWriter(fps=30)
print("Saving animation...")
ani.save(f, writer=writergif)
print("Done.")

# Prpare data for saving
data = {
    "parameters": {
        "nBodies": nBodies,
        "nDimensions": nDimensions,
        "nIterations": nIterations,
        "collisionTolerance": collisionTolerance,
        "outOfReachTolerance": outOfReachTolerance,
        "dt": dt,
        "lengthUniverse": lengthUniverse,
        "massScaling": massScaling,
        "velocitySpan": velocitySpan
    },
    "initialPositions": initialPositions.tolist(),
    "masses": masses.tolist(),
    "initialVelocities": initialVelocities.tolist()
}

# Save to a JSON file
with open('simulation_parameters.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Parameters saved")
plt.show()

