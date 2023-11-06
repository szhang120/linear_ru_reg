import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(1151)

# support (i should condense these into one variable)
x_train_test = np.linspace(0, 10, 150)

# increase st dev of noise based on p, parametrizing covariate shift
def generate_shift_noise(size, p):
    return np.random.normal(0, 0.2 + p/2.5, size)

# Train on p = 0.2
p_train = 0.2
y_train = 0.1*x_train_test + np.sin(x_train_test) + generate_shift_noise(len(x_train_test), p_train)

# introduce parameter p; Generate y values for test distributions
p_vals = [0.1, 0.2, 0.5, 0.7, 0.9]
y_test = []

# y_test generation for values of p
for p in p_vals:
    y = 0.1 * x_train_test + np.sin(x_train_test) + generate_shift_noise(len(x_train_test), p)
    y_test.append(y)

# Step 4a: create and train model
x_train_test_sk = np.array(x_train_test).reshape(-1, 1)
y_train_sk = np.array(y_train).reshape(-1, 1)
lr = LinearRegression()
lr.fit(x_train_test_sk, y_train_sk)

# generate plots for different test distributions
fig, axs = plt.subplots(1, len(p_vals), figsize=(20, 4))
mse_vals = []
for i, p in enumerate(p_vals):
  axs[i].scatter(x_train_test, y_test[i], color='blue', label=f'Test (with shift, p={p})', s=50, alpha=0.5)
  y_pred_test = lr.predict(x_train_test_sk)
  axs[i].plot(x_train_test, y_pred_test, color = 'red', label = "Predictions on test data")

  # find and show mse
  mse_vals.append(mean_squared_error(y_test[i], y_pred_test))
  axs[i].set_title(f'p={p} (MSE: {mse_vals[i]:.2f})')
  axs[i].legend()
  axs[i].grid(True)
  axs[i].set_ylim([-4.0, 4.0])
plt.show()

# Adjust layout to make room for the table

fig.text(0.5, 0.04, 'X', ha='center', va='center')
fig.text(0.06, 0.5, 'Y', ha='center', va='center', rotation='vertical')
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])


# Add table for MSE values
fig, ax = plt.subplots(figsize=(13.7, 1))  # Adjust the size as needed
cell_text = [mse_vals]
columns = [f'p = {p}' for p in p_vals]
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')

table.scale(1.5, 1.5)
plt.show()
