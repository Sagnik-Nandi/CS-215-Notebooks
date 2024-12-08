# import matplotlib.pyplot as plt
# import numpy as np
# # import seaborn as sns
# import pandas as pd

# # Sample Data
# data = {'Height': [160, 162, 165, 170, 155, 172, 167, 160, 159, 166, 165, 161, 171, 168, 163, 162, 164, 170, 165, 168],
#         'Class': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'A', 'A', 'B', 'B', 'C']}

# df = pd.DataFrame(data)

# # Violin Plot
# plt.figure(figsize=(8, 6))
# plt.violinplot(data)
# plt.title('Violin Plot of Heights by Class')
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
x=np.array([10,40,30,10,30,50,60,70,50,40,60,50])
plt.violinplot(x)
plt.show()
uniform=np.arange(-5,50)
normal=np.random.normal(70,10,100)
plt.violinplot(normal)
plt.violinplot(uniform)
# normal=np.random.normal(70,10,100)
normal1=np.random.normal(80,10,100)
normal2=np.random.normal(90,10,100)
normal3=np.random.normal(100,10,100)
normalall=[normal,normal1,normal2,normal3]
plt.violinplot(normalall,showmeans=True,showmedians=True)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Sample Data
complaints = {'Late Delivery': 45, 'Damaged Goods': 25, 'Incorrect Item': 15, 'Poor Service': 10, 'Other': 5}
categories = list(complaints.keys())
frequencies = list(complaints.values())

# Sort Data
sorted_indices = np.argsort(frequencies)[::-1]
categories = np.array(categories)[sorted_indices]
frequencies = np.array(frequencies)[sorted_indices]

# Cumulative Percentage
cumulative = np.cumsum(frequencies) / np.sum(frequencies) * 100

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(categories, frequencies, color='skyblue')
ax1.set_xlabel('Complaint Type')
ax1.set_ylabel('Frequency')

ax2 = ax1.twinx()
ax2.plot(categories, cumulative, color='red', marker='o')
ax2.set_ylabel('Cumulative Percentage')

plt.title('Pareto Chart of Customer Complaints')
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Sample Data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
product_a = [10, 20, 15, 25, 30, 22]
product_b = [15, 25, 20, 30, 35, 27]

# Plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

theta = np.linspace(0.0, 2 * np.pi, len(months), endpoint=False)
radii_a = np.array(product_a)
radii_b = np.array(product_b)

bars = ax.bar(theta, radii_a, width=0.3, label='Product A', alpha=0.5)
bars = ax.bar(theta + 0.3, radii_b, width=0.3, label='Product B', alpha=0.5)

ax.set_xticks(theta)
ax.set_xticklabels(months)
ax.legend()

plt.title('Coxcomb Chart of Monthly Sales')
plt.show()
import matplotlib.pyplot as plt

# Sample Data
categories = ['Revenue', 'Cost of Goods Sold', 'Operating Expenses', 'Other Expenses', 'Net Profit']
values = [100, -40, -20, -10, 30]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Cumulative Sum
cumulative = [values[0]]
for i in range(1, len(values)):
    cumulative.append(cumulative[-1] + values[i])

# Bars
colors = ['green' if v > 0 else 'red' for v in values]
bars = ax.bar(categories, cumulative, color=colors)

# Add value labels
for bar, value in zip(bars, values):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{value}', ha='center', va='bottom')

plt.title('Waterfall Plot of Financial Performance')
plt.ylabel('Value')
plt.show()