import numpy as np         # Loading numpy
from matplotlib import pyplot as plt    # Loading matplotlib

# Initial conditions
nx = 41    # Sparial discretization - try to change it to 82 and see what happens
dx = 2/(nx - 1) # Grid size, in this case (b - a) = 2
nt = 25    # temporal discretization
nu = 0.3
sigma = .2  # This is the stability condition for the diffusion equation nu*dt/dx^2
dt = sigma * dx**2 / nu  # time step size calculated using the stability limit
print('The time step is: ', dt)

u = np.ones(nx)  # initialization array for the velocity
u[int(0.5 / dx):int(1 / dx+1)] = 2   # setting u=2 between 0.5 and 1 as initial conditions
plt.plot(np.linspace(0, 2, nx), u);


# Calculation
un = np.ones(nx)
for n in range(nt):  # time integration
    un = u.copy() # Copying the existing value of u
    for i in range(1, nx-1):   # iteration in space
        u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])
    
        plt.plot(np.linspace(0, 2, nx), u);
        plt.xlabel('Space x')
        plt.ylabel('Velocity u') 
plt.show()

