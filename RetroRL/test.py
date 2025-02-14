import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Example data
estimated_steps = np.array([
    8.860792 , 2.9172564, 4.5525603, 6.7202544, 5.028397 , 8.782892 ,
    5.4923296, 5.770047 , 8.923357 , 4.2066097, 2.8898087, 2.6888852,
    2.0163813, 6.9395823, 1.9651723, 1.9548368, 4.0307446, 3.4739523,
    4.514614 , 3.5976005, 8.323462 , 6.7540984, 4.6213593, 5.528825 ,
    4.03715  , 5.1458473, 5.449377 , 3.4500432, 6.009703 , 9.138113 ,
    1.7434301, 8.243933 , 5.052349 , 3.4012446, 6.462631 , 1.8468137
], dtype=np.float32)

real_steps = np.array([
    6,  3,  2, 13,  9,  2,  4,  3,  2,  4,  1,  1,  1, 10,  1,  1,  1,
    3,  3,  5,  5,  5,  1, 15,  1,  4,  2,  5,  2,  8,  3,  9, 12,  1,
    14,  1
])

# Calculate metrics
mse = mean_squared_error(real_steps, estimated_steps)
rmse = np.sqrt(mse)
mae = mean_absolute_error(real_steps, estimated_steps)
r2 = r2_score(real_steps, estimated_steps)
corr = np.corrcoef(estimated_steps, real_steps)[0, 1]

# Plot
fig, ax = plt.subplots(figsize=(7,6))

# Scatter plot
ax.scatter(estimated_steps, real_steps,
           alpha=0.7, color='blue', edgecolors='k', s=80, label='Data points')

# Plot the diagonal y=x for reference
min_val = min(estimated_steps.min(), real_steps.min())
max_val = max(estimated_steps.max(), real_steps.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

# Add labels, legend, and grid
ax.set_xlabel('Estimated Steps', fontsize=12)
ax.set_ylabel('Real Steps', fontsize=12)
ax.set_title('Scatter Plot of Estimated vs. Real Steps', fontsize=14)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Prepare metrics text
metrics_text = (
    f"Correlation: {corr:.2f}\n"
    f"R²: {r2:.2f}\n"
    f"MSE: {mse:.2f}\n"
    f"RMSE: {rmse:.2f}\n"
    f"MAE: {mae:.2f}"
)

# Place the text box in the lower-right corner
ax.text(
    0.95, 0.05,  # (x, y) in axes coordinates: near lower-right
    metrics_text,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.2)
)

plt.show()
