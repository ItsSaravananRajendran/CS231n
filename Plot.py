import matplotlib.pyplot as plt
import numpy as np

data = None
with open('./nohup.out','r') as input:
    data = input.read()
    data = data.split('\n')

loss = []
r1 = []
r2 = []
learn = []
dead = []
valid = []

for line in data:
	con = map(float,line.split(' '))
	if len(con) == 6:
		loss.append(con[0])
		r1.append(con[1])
		r2.append(con[2])
		learn.append(con[3])
        dead.append(con[4])
        valid.append(con[5])

width = 100
temp = []
for I in range(len(loss)-width):
	temp.append(np.mean(loss[I:I+width]))
loss = temp


temp = []
for I in range(len(dead)-width):
	temp.append(np.mean(dead[I:I+width]))
dead = temp


temp = []
for I in range(len(valid)-width):
	temp.append(np.mean(valid[I:I+width]))
valid = temp

plt.plot([I for I in range(len(loss))], loss,'r')
plt.title('Loss function')
plt.show()
plt.title('Learning Rate')
plt.plot([I for I in range(len(learn))], learn,'b')
plt.show()
plt.title('Regularization Loss')
plt.plot([I for I in range(len(r1))], r1,'y')
plt.plot([I for I in range(len(r2))], r2,'g')
plt.show()
plt.title('Dead Neurons')
plt.plot([I for I in range(len(dead))], dead,'r')
plt.show()
plt.title('Validation %')
plt.plot([I for I in range(len(valid))],valid,'b')
plt.show()
