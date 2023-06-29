import numpy as np
import random
import matplotlib.pyplot as plt
color=['r','b']
n_train=1000
n_test=500
N=1500; #N points for each cluster
# Train partition
p =[6,10,12,5] # Cluster centers
x=np.zeros((2*N,2)) # Pre allocation of the input and output
t =np.zeros((2*N,1)) # Target


# Synthetic Data
r =np.random.normal ( 0 , 1 , 2*N)
the=2*np.pi*np.random.rand(2*N)
x[0:N,0] = p[0] + r [ 0 :N]*np.cos(the[0:N] )
x[0:N,1] = p[1] + r [ 0 :N]*np.sin(the[0:N] )
t[0:N]=0
x[N:2*N,0] = p[2] + r[N:2*N]*np.cos(the[N:2*N] )
x[N:2*N,1] = p[3] + r[N:2*N]*np.sin(the[N:2*N] )
t[N:2*N]=1



# Create the model
model = Sequential()
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


indices=range(2*N)
indices=random.sample(indices,n_train)
train_points=x[indices]
train_labels=t[indices]

# Train the model
model.fit(train_points, train_labels, epochs=100)

# Predict on new data
indices=range(2*N)
indices=random.sample(indices,n_test)
test_points=x[indices]
new_data = test_points
predictions = model.predict(new_data)
predictions=[int(i<0.5) for i in predictions]

print(predictions,'*')

# Get the parameters
parameters = model.get_weights()

# Extract the slope and intercept
slope = -parameters[0][0] / parameters[1][0]
intercept = -parameters[0][1] / parameters[1][0]

x_vals=np.linspace(1,20,20)
# Plot the line
y_vals = slope * x_vals + intercept

# Print the parameters
for layer_weights in parameters:
    print(layer_weights,'*')


plt.figure(figsize=(5,5))
plt.scatter(x[:N,0] , x[:N,1],color=color[0])
plt.scatter(x[N:2*N,0] , x[N:2*N,1],color=color[1])
plt.xlim(0,20)
plt.ylim(0,20)
plt.plot(x_vals, y_vals)


plt.figure(figsize=(5,5))
plt.xlim(0,20)
plt.ylim(0,20)
plt.scatter(new_data[:,0] , new_data[:,1],
            color=[color[i] for i in predictions[:]])
plt.plot(x_vals, y_vals)
