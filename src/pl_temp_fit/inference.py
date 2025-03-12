import numpy as np
from bayesflow.networks import AmortizedPosterior

# Load the trained network
amortized_posterior = AmortizedPosterior()
amortized_posterior.load('path_to_trained_model')

# Define your observed data
observed_data = np.array([...])  # Replace with your actual observed data

# Perform inference
posterior_samples = amortized_posterior.sample(observed_data, num_samples=1000)

# Analyze the posterior samples
mean_estimate = np.mean(posterior_samples['mean'])
std_estimate = np.mean(posterior_samples['std'])

print(f"Estimated mean: {mean_estimate}")
print(f"Estimated std: {std_estimate}")
