# Model Inversion

A jax implementation of the model inversion attack against machine learning models proposed by [Fredrickson et al.](https://dl.acm.org/doi/10.1145/2810103.2813677).
This attack takes a machine learning model and learns the data used to train it by looking at the outputs when given steadily optimized inputs.


## Running

First install the [JAX library](https://github.com/google/jax) then use pip to install the requirements from the requirements.txt file.

Then run the `create_model.py` file to train a model on the MNIST dataset.

And finally run the `attack.py` file to perform the attack on that model.


## Observations

This attack is less effective against the MNIST dataset, likely due to the variation in the orientations in the inputs. Although, agreeing what is reported in the original paper, the attack is most effective against the softmax model. Another interesting observation we see is that robustness to adversarial examples or equivalently improvements to model explainability make the attack substantially more effective.
