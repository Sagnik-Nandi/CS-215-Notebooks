import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('NED21.11.1-D-5.1.1-20111026-v20120814.csv')

# first 1500 datapoints
df_first_1500 = df.head(1500)

# objects with a distance less than 4 Mpc
df_filtered = df_first_1500[df_first_1500['D (Mpc)'] < 4]

# histogram with 10 bins
n, bins, patches = plt.hist(df_filtered['D (Mpc)'], bins=10, edgecolor='black', density=True)
plt.xlabel('Distance from Earth (Mpc)')
plt.ylabel('Estimated Probability')
plt.title('Histogram of Objects with Distance < 4 Mpc (First 1500 Datapoints)')

# Save 
plt.savefig('10binhistogram.png')
plt.show()

# estimated probabilities for each bin (p hat j)
bin_width = np.diff(bins)
estimated_probabilities = n * bin_width


data = df_filtered['D (Mpc)'].values
n = len(data)  # Number of data points

# Print the estimated probabilities
print("Printing values for part (a) ")
for i, prob in enumerate(estimated_probabilities):
    print(f"Estimated probability for bin {i + 1}: {prob:.4f}")

def cross_validation_score(num_bins):
    # Histogram calculation
    counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
    
    # Bin width (h)
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Estimated probabilities (p_hat j)
    estimated_probabilities = counts * bin_width
    
    # Cross-validation score formula
    J_hat = (2 / ((n - 1) * bin_width)) - ((n + 1) / ((n - 1) * bin_width)) * np.sum(estimated_probabilities ** 2)
    
    return J_hat

# cross-validation for bin counts from 1 to 1000
bin_counts = np.arange(1, 1001)
cv_scores = [cross_validation_score(b) for b in bin_counts]

# Plot the cross-validation 
plt.plot(bin_counts, cv_scores)
plt.xlabel('Number of Bins')
plt.ylabel('Cross-Validation Score')
plt.title('Cross-Validation Score vs Number of Bins')

# Save
plt.savefig('crossvalidation.png')
plt.show()

optimal_bin_count = bin_counts[np.argmin(cv_scores)]

#optimal bin width (h)
data_range = np.max(data) - np.min(data)
optimal_bin_width = data_range / optimal_bin_count
print ("Answer for Part (d)")
print(f"Optimal number of bins: {optimal_bin_count}")
print(f"Optimal bin width (h): {optimal_bin_width:.4f} Mpc")


#histogram with 10 bins (from part a)
plt.figure(figsize=(10, 5))

#Histogram with 10 bins (from part a)
plt.subplot(1, 2, 1)
plt.hist(data, bins=10, edgecolor='black', density=True)
plt.xlabel('Distance from Earth (Mpc)')
plt.ylabel('Estimated Probability')
plt.title('Histogram with 10 Bins')

#Histogram with optimal bins
plt.subplot(1, 2, 2)
plt.hist(data, bins=optimal_bin_count, edgecolor='black', density=True)
plt.xlabel('Distance from Earth (Mpc)')
plt.ylabel('Estimated Probability')
plt.title(f'Histogram with Optimal Bin Width (h* = {optimal_bin_count} bins)')

# Save
plt.tight_layout()
plt.savefig('optimalhistogram.png')
plt.show()



#Answer for Part (e) -> It has more number of bins than number of bins in part(a) 
# and It seems just-fit for the optimal h 

#Answer for part B -> Underfit