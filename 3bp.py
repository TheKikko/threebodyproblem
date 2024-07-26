import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from tqdm import tqdm
import json

# ================================================================================
# Parameter initialization
# ================================================================================

nBodies = 3
nDimensions = 2
nIterations = int(2e2)

collisionTolerance = 5e-1
outOfReachTolerance = 1

dt = 0.1

lengthUniverse = 40
massScaling = 250
velocitySpan = 5

# ================================================================================
# Function definitions
# ================================================================================

def gravity(body1, body2, gravityConstant=1):  # Using G = 1 for simplicity
    r_vec = body2.position - body1.position
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:
        return np.zeros_like(r_vec)
    r_hat = r_vec / r_mag
    force_magnitude = gravityConstant * body1.mass * body2.mass / r_mag ** 2
    return force_magnitude * r_hat

def saveFinalPosition(iteration, position):
    return iteration, position.copy()

def simulate(nIterations, bodies):
    progress_bar = tqdm(total=nIterations, desc='Simulation 3-body problem...')

    collision = False
    outOfReach = False
    iteration = 0
    finalIteration = nIterations
    finalPosition = np.zeros(2)
    while not (collision or outOfReach) and iteration < nIterations:
        forces = [np.zeros(2) for _ in bodies]
        for i in range(nBodies):
            for j in range(nBodies):
                if i != j:
                    distance = np.linalg.norm(bodies[i].position - bodies[j].position)
                    if distance < collisionTolerance:
                        collision = True
                        finalIteration, finalPosition = saveFinalPosition(iteration, bodies[i].position)
                    f = gravity(bodies[i], bodies[j])
                    forces[i] += f

        for i, body in enumerate(bodies):
            body.velocity += dt * forces[i] / body.mass
            body.move(dt)
            #body.position += dt * body.velocity
            if np.linalg.norm(forces[i]) < outOfReachTolerance:
                outOfReach = True
                finalIteration, finalPosition = saveFinalPosition(iteration, body.position)

        iteration += 1
        progress_bar.update(1)

    progress_bar.close()

    if collision or outOfReach:
        for body in bodies:
            body.position = finalPosition
    return bodies, collision, outOfReach, finalIteration, finalPosition

def init():
    ax.set_xlim(-lengthUniverse, lengthUniverse)
    ax.set_ylim(-lengthUniverse, lengthUniverse)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    global scatters, lines
    scatters = [ax.scatter([], [], color=color, s=50) for color in colors]
    lines = [ax.plot([], [], '-', lw=2, alpha=0.5, color=color)[0] for color in colors]

    return scatters + lines

def animate(bodies, collision, outOfReach, finalIteration, finalPosition, indexFrame):
    if indexFrame > 0:
        for i, body in enumerate(bodies):
            history_x = [pos[0] for pos in body.history[:indexFrame]]
            history_y = [pos[1] for pos in body.history[:indexFrame]]

            lines[i].set_data(history_x, history_y)
            if indexFrame <= finalIteration:
                scatters[i].set_offsets(body.history[indexFrame])
            else:
                scatters[i].set_offsets(body.history[-1])


        histories = np.array([body.history[:indexFrame] for body in bodies])

        ax.set_xlim(np.min(histories[:, :, 0])-5, np.max(histories[:, :, 0])+5)
        ax.set_ylim(np.min(histories[:, :, 1])-5, np.max(histories[:, :, 1])+5)
        #ax.set_xlim(np.min([body.history[:indexFrame][0] for body in bodies]) - 5,
        #            np.max([body.history[:indexFrame][0] for body in bodies]) + 5)
        #print(f"History[:{indexFrame}]: {body.history[:indexFrame]}")
        #ax.set_ylim(np.min([body.history[:indexFrame][0] for body in bodies]) - 5,
        #            np.max([body.history[:indexFrame][0] for body in bodies]) + 5)

    if collision and indexFrame >= finalIteration:
        ax.text(finalPosition[0], finalPosition[1] + 1, 'Collision!', color='red', fontsize=12, ha='center')
    if outOfReach and indexFrame >= finalIteration:
        ax.text(0.5, 0.5, 'Simulation stopped: Body left the reach of others', color='red', fontsize=12, ha='center')

    if (collision or outOfReach) and indexFrame < finalIteration:
        ax.set_title(f'3-Body Problem Simulation at time {indexFrame * dt:.1f} years')
    else:
        ax.set_title(f'3-Body Problem Simulation stopped at time {finalIteration * dt:.1f} years')

    return scatters + lines

# ================================================================================
# Body definition
# ================================================================================

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.history = [self.position.copy()]

    def move(self, time):
        self.position += self.velocity * time
        self.history.append(self.position.copy())

    def __str__(self):
        return f"Body(mass={self.mass}, position={self.position}, velocity={self.velocity})"


# ================================================================================
# Body initialization, simulation
# ================================================================================

rng = np.random.default_rng()

# Initialize bodies
bodies = []
initialVelocities = [] # velocity history isn't saved, 
for _ in range(nBodies):
    mass = rng.random() * massScaling
    position = rng.random(nDimensions) * lengthUniverse
    velocity = -velocitySpan + rng.random(nDimensions) * 2 * velocitySpan
    bodies.append(Body(mass, position, velocity))
    initialVelocities.append(velocity)

positions, collision, outOfReach, finalIteration, finalPosition = simulate(nIterations, bodies)

# ================================================================================
# Animation
# ================================================================================

fig, ax = plt.subplots(nrows=1, ncols=1)
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, nBodies)]
legends = [ax.scatter([], [], color=colors[i], s=50, label=f'Body {i + 1}, Mass {bodies[i].mass:.2f} solar masses') for i in range(nBodies)]
ax.legend()

# FuncAnimation requires an input function which takes one argument, the frame index
# do partial function application with all parameters except that one
anim = partial(animate, bodies, collision, outOfReach, finalIteration, finalPosition)

print("Creating animation...")
ani = animation.FuncAnimation(
    fig, anim, nIterations+1, init_func=init, interval=dt*500, blit=True)

# ================================================================================
# Saving
# ================================================================================

f = r"/home/kikko/repos/3bodyproblem/animation.gif"
writergif = animation.PillowWriter(fps=30)
print("Saving animation...")
ani.save(f, writer=writergif)
print("Done.")

# Prepare data for saving
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
    "initialPositions": [body.position.tolist() for body in bodies],
    "masses": [body.mass for body in bodies],
    "initialVelocities": [velocity.tolist() for velocity in initialVelocities]
}

with open('simulation_parameters.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Parameters saved")

