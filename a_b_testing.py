import numpy as np
from scipy import stats

# Define the data
# Group A: 1000 users with 200 conversions
# Group B: 1000 users with 250 conversions

# Conversions
conversions_A = 200
conversions_B = 250

# Total users
total_A = 1000
total_B = 1000

# Conversion rates
rate_A = conversions_A / total_A
rate_B = conversions_B / total_B

print(f"Conversion rate for group A: {rate_A:.2%}")
print(f"Conversion rate for group B: {rate_B:.2%}")

# Perform a two-proportion z-test
# Proportions
p1 = conversions_A / total_A
p2 = conversions_B / total_B

# Pooled proportion
p_pool = (conversions_A + conversions_B) / (total_A + total_B)

# Standard error
se = np.sqrt(p_pool * (1 - p_pool) * (1 / total_A + 1 / total_B))

# Z-score
z = (p1 - p2) / se

# p-value from the z-score
p_value = stats.norm.sf(abs(z)) * 2  # two-tailed test

print(f"Z-score: {z:.2f}")
print(f"P-value: {p_value:.4f}")

# Interpret the results
alpha = 0.05  # significance level
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the two groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the two groups.")
