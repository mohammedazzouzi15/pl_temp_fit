import numpy as np
from bayesflow.networks import AmortizedPosterior
from bayesflow.trainers import Trainer
from bayesflow.simulation import Prior, GenerativeModel

# Define the prior distribution
def prior(batch_size):
    return {
        'mean': np.random.uniform(-10, 10, size=(batch_size, 1)),
        'std': np.random.uniform(0.1, 5, size=(batch_size, 1))
    }

# Define the generative model
def generative_model(params):
    return simulator(params)

# Create the prior and generative model objects
prior = Prior(prior)
generative_model = GenerativeModel(generative_model)

# Create the amortized posterior network
amortized_posterior = AmortizedPosterior()

# Create the trainer
trainer = Trainer(amortized_posterior, prior, generative_model)

# Train the network
trainer.train(epochs=10000)
