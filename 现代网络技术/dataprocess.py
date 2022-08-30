import numpy as np
import matplotlib.pyplot as plt

x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
local = [168, 167, 170, 169, 168, 170, 171, 173, 170, 169]
random = [13, 19, 28, 37, 50, 71, 95, 123, 138, 164]
my_approach = [170, 171, 173, 179, 186, 192, 195, 198, 200, 200]
local = np.array(local)
random = np.array(random)
my_approach = np.array(my_approach)
f1 = np.polyfit(x, local, 5)
p1 = np.poly1d(f1)
y1 = p1(x)
f2 = np.polyfit(x, random, 5)
p2 = np.poly1d(f2)
y2 = p2(x)
f3 = np.polyfit(x, my_approach, 5)
p3 = np.poly1d(f3)
y3 = p3(x)

plot1 = plt.plot(x, y1, 'r', label='local')
plot2 = plt.plot(x, y2, 'g', label='random', linestyle=':')
plot3 = plt.plot(x, y3, 'b', label='my_approach', linestyle='--')
plt.legend(loc=0,)

plt.xlabel('the value of alpha beta')
plt.ylabel('number of accommodate task')
plt.show()
