import matplotlib.pyplot as plt
import numpy as np

x_data = [ 338., 333., 328., 207., 226., 25., 179., 60., 208.,  606.]
y_data = [ 640., 633., 619., 393., 428., 27., 193., 66., 223., 1591.]

x = np.arange(-200, -100, 1)
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

b = -120
w = -4
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
plt.plot([-188.4], [2.67], 'x', ms = 12, markeredgewidth = 3, color = 'orange') # 橘色叉叉
plt.plot(b_his, w_his, 'o-', ms = 3, lw = 1.5, color = 'black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize = 16)
plt.ylabel(r'$w$', fontsize = 16)
plt.show()