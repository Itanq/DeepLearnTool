
import numpy as np
import matplotlib.pyplot as plt

circle1 = plt.Circle((0.5,0.5),0.3,color='r')
circle2 = plt.Circle((0.5,0.4),0.2,color='g')
circle3 = plt.Circle((0.5,0.3),0.1,color='y')

plt.text((0.3,0.6),"ML")

fig,ax = plt.subplots()

ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)

fig.savefig('tet.png')
