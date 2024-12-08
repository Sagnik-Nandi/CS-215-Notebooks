#                                      Violin Plot
import matplotlib.pyplot as plt
import numpy as np
normal=np.random.normal(70,10,100)
normal1=np.random.normal(80,10,100)
normal2=np.random.normal(90,10,100)
normal3=np.random.normal(100,10,100)
normalall=[normal,normal1,normal2,normal3]
plt.violinplot(normalall,showmeans=True,showmedians=True)
plt.title("Violin Plot for different normal distribution")
plt.show()


#                                      Pareto Chart

import pandas as pd
df =pd.DataFrame({"Number of Students" : [60,30,20,10,5]})
df.index=["Safe quizzes","Safe Atendence","iPhone","Poor connection ","other"]
df=df.sort_values(by='Number of Students',ascending =False)
df["cumulative_percentage"]=round(df["Number of Students"].cumsum()/df["Number of Students"].sum()*100,2)
fig = plt.figure()
ax=fig.add_subplot(111)
ax.bar(df.index,df["Number of Students"])
ax.set_xlabel("Complaints of Students")
ax.set_ylabel("Number of Studnets")
ax2=ax.twinx()
ax2.plot(df.index,df["cumulative_percentage"],marker="x",color="red")
ax2.set_ylabel("Cumulatuve Percentage")
plt.title('Pareto Chart of Student Complaints abut Safe APP')
plt.show()

#                                       Coxcomb Chart

months = ['Aug', 'Sep', 'Oct', 'Nov']
Section_D3 = [280, 200, 140, 100]
Section_D4 = [280, 160, 120, 70]

# Plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

theta = np.linspace(0.0, 2 * np.pi, len(months), endpoint=False)
radii_a = np.array(Section_D3)
radii_b = np.array(Section_D4)
a=2*np.pi
b=len(months)
# Here I assumed that n = 2 * pi  
bars = ax.bar(theta, radii_a, width=a/8, label='Section_D3', alpha=0.5)
bars = ax.bar(theta + a/8, radii_b, width=a/8, label='Section_D4', alpha=0.5)

ax.set_xticks(theta)
ax.set_xticklabels(months)
ax.legend()

plt.title('Coxcomb Chart of Monthly Attendence in MA 105')
plt.show()

#                                       2D Waterfall Plot

# Sample Data
categories = ['Initial Attendence', 'Not attending due to laziness', 'Not attending due to DSA prep', 'Other Excuses', 'Net Attendence']
values = [100, -40, -20, -10, 0]

# Plot
fig, ax = plt.subplots(figsize=(15, 15))

# Calculate cumulative values for the bars
cumulative = [0]  # Start from 0 for the first bar
for i in range(1, len(values)):
    cumulative.append(cumulative[-1] + values[i - 1])

# Add the final cumulative value for the last bar
final_value = cumulative[-1] + values[-1]

# Bars
colors = ['green' if v > 0 else 'red' for v in values[:-1]] + ['blue']
bars = ax.bar(categories[:-1], values[:-1], bottom=cumulative[:-1], color=colors[:-1])
bars += ax.bar(categories[-1], final_value, bottom=0, color=colors[-1])  # Last bar starts from 0

# Add value labels
for bar, value, cum in zip(bars[:-1], values[:-1], cumulative[:-1]):
    if value >= 0:
        ax.text(bar.get_x() + bar.get_width() / 2, cum + value / 2, f'{value}', ha='center', va='center', color='white')
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, cum + value / 2, f'{value}', ha='center', va='center', color='white')

# Add value label for the last bar
ax.text(bars[-1].get_x() + bars[-1].get_width() / 2, final_value / 2, f'{final_value}', ha='center', va='center', color='white')

# Set title and labels
plt.title('Waterfall Plot of Attendence of a batch in IIT')
plt.ylabel('Number Of  Students Attending')
plt.show()

#                                       3D waterfall Plot

#  Generate sample data
# Let's assume we have spectral data as a function of frequency and noise level (or speed).
frequencies = np.linspace(0, 50, 100)   # X-axis (e.g., Frequency in Hz)
noise_levels = np.linspace(0, 10, 50)   # Y-axis (e.g., Noise level or Speed)
Z = np.zeros((len(noise_levels), len(frequencies)))

# Here we use a Gaussian function to simulate spectral data
for i, noise in enumerate(noise_levels):
    Z[i, :] = np.exp(-((frequencies - 25)**2) / (2 * (5 + noise)**2))  # Z-axis (Amplitude)
# Formula from the internet 

#  the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

#  Plot each spectrum as a separate line
for i in range(len(noise_levels)):
    ax.plot(frequencies, Z[i, :], zs=noise_levels[i], zdir='y', alpha=0.8)

#  Customize the plot
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Noise Level (dB) or Speed')
ax.set_zlabel('Amplitude')
ax.set_title('3D Waterfall Plot of Spectral Data')

#  Display the plot
plt.show()
#Refernce https://www.statisticshowto.com/waterfall-plot/