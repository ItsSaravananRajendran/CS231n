import matplotlib.pyplot as plt

data = None
with open('./nohup.out','r') as input:
    data = input.read()
    data = data.replace('loss = ',' ')
    data = data.replace('regLoss = ','')
    data = data.replace('epoch = ','')
    data = data.replace(' validation = ','')
    data = data.split('\n')

loss= []
regLoss = []
valid = []

for line in data:
        dat = line.strip().split(' ')
	Loss,RegLoss,_,Valid = map(float,dat)
	loss.append(Loss)
	regLoss.append(RegLoss)
	valid.append(Valid)

plt.plot([I for I in range(len(loss))], loss)
plt.show()
plt.plot([I for I in range(len(regLoss))], regLoss)
plt.show()
plt.plot([I for I in range(len(valid))], valid)
plt.show()
