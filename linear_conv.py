import numpy as np         # Loading numpy
from matplotlib import pyplot as plt    # Loading matplotlib

# Setting the boundary conditions of the problem
nx = 81    # Sparial discretization - try to change it to 82 and see what happens
dx = 2/(nx - 1) # Grid size, in this case (b - a) = 2
nt = 25    # temporal discretization
dt = 0.025  # time step size
c = 1  # Assiming a constant velocity c

u = np.ones(nx)   # starting the array for the result
u[int(0.5 / dx):int(1 / dx+1)] = 2   # setting u=2 between 0.5 and 1 as initial conditions
print(u)
plt.plot(np.linspace(0, 2, nx), u);

# Let us check the CFL number
CFL = c*dt/dx
print('The CFL number is:')
print(CFL)
if CFL>1:
    print('CFL greater than 1, instable!')

# Starting the calculation for the new velocity
un = np.ones(nx)  # Initialize the temporary array
for n in range(nt): # Loop for value of n from 0 to nt
    un = u.copy()   # Copying the existing values of u in un
    for i in range(1,nx):
        u[i]= un[i] -c*dt/dx * (un[i] - un[i-1])
plt.plot(np.linspace(0,2,nx),u, '-bo');      
plt.xlabel('Space x')
plt.ylabel('Velocity u')  

plt.show()