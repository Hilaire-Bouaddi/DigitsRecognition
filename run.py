import matplotlib.pyplot as plt
import animation

fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
animation.draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, [15,12,10])
plt.show()
