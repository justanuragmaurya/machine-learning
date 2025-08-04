import pandas as pd
import matplotlib.pyplot as plt

# Load dataset with study_time and score columns
data = pd.read_csv("data.csv")

# Visualize raw data
plt.scatter(data.study_time,data.score)

# Mean Squared Error function (currently unused)
# Calculates average squared difference between predicted and actual values
# def mean_squared_error(m,b,points):
#     total_error = 0
#     for i in range(len(points)):
#         x = points.iloc[i].study_time
#         y = points.iloc[i].score
#         total_error += (y - (m*x+b))**2
    
#     total_error /= float(len(points)) 

#     return m,b

# Gradient descent optimization function
# Updates slope (m) and intercept (b) to minimize prediction error
# Uses partial derivatives to find optimal parameter adjustments
def gradient_descent(m_now,b_now,points,L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    # Calculate gradients for all data points
    for i in range(n):
         x = points.iloc[i].study_time
         y = points.iloc[i].score

         # Partial derivatives of cost function
         m_gradient += -(2/n) * x * ( y - ( m_now * x + b_now ))
         b_gradient += -(2/n) * ( y - ( m_now * x + b_now ))

    # Update parameters using learning rate
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m,b

# Model training: y = m * x + b
# Initialize parameters and hyperparameters
m = 0
b = 0
L = 0.001 # learning rate
epochs = 10000 # no of iterations

# Train model using gradient descent
for i in range(epochs):
     m,b = gradient_descent(m,b,data,L)

# Display results and plot fitted line
print(m,b)
plt.scatter(data.study_time,data.score,color="black")
plt.plot(list(range(0, 10)),[m*x+b for x in range (0,10)],color="red")
plt.show()