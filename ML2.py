import time

start = time.time()

import matplotlib.pyplot as plt
import numpy as np

x_data = [1,2,3,4,5,6,7,8,9,10]
y_data = [3,4,5,6,7,8,9,10,11,12]

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
Z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(len(x), len(y))
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] += (y_data[n] - b - w * x_data[n])**2
        Z[j][i] = Z[j][i]/len(x_data)

b = -5
w = -5
lr = 1
iteration = 100000

b_his = [b]
w_his = [w]

lr_b = 0
lr_w = 0

for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0*(y_data[n] - b - w * x_data[n]) * 1
        w_grad = w_grad - 2.0*(y_data[n] - b - w * x_data[n]) * x_data[n]
    
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2
    
    b = b - lr/np.sqrt(lr_b) * b_grad
    w = w - lr/np.sqrt(lr_w) * w_grad
    w_his.append(w)
    b_his.append(b)

plt.contourf(x, y, Z, 100, alpha = 0.5, cmap = plt.get_cmap('jet'))
plt.plot([2], [1], 'x', ms = 12, markeredgewidth = 3, color = 'orange') # 橘色叉叉
plt.plot(b_his, w_his, 'o-', ms = 3, lw = 1.5, color = 'black')
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize = 16)
plt.ylabel(r'$w$', fontsize = 16)
plt.show()

end = time.time()

print(end - start)