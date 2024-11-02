import jax
import jax.numpy as jnp
import numpy as np
# Ex 3.a
# The maximum number of coefficients to fully specify P(x,y,z) for given Nx, Ny, Nz is
#  (Nx+1)*(Ny+1)*(Nz+1) 
# since for each variable you have the terms from order 0 to order N.
# A suitable data structure in jax for storing the coiffiecients of P(x,y,z) is a 3D array.

# Ex 3.b
def generate_polynomial_coeffs(rng_key, N_x, N_y, N_z, t):
    """ Generate random coefficients for a polynomial of order Nx, Ny, Nz
        args:
        rng_key: random number generator key
        Nx, Ny, Nz: order of the polynomial in x, y, z
        t: number of coefficients to generate  """
    coeffs = jnp.zeros((N_x + 1, N_y + 1, N_z + 1))
    indices = jax.random.choice(rng_key, coeffs.size, shape=(t,), replace=False)
    coeffs = coeffs.at[jnp.unravel_index(indices, coeffs.shape)].set(
        jax.random.uniform(rng_key, (t,))
    )
    return coeffs

def evaluate_polynomial(coeffs, x, y, z):
    total = 0
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            for k in range(coeffs.shape[2]):
                total += coeffs[i, j, k] * (x ** i) * (y ** j) * (z ** k)
    return total

def generate_training_data(coeffs, N, rnd_seed=42):
    noise_frac = 0.25
    rnd_key = jax.random.PRNGKey(rnd_seed)
    x = jax.random.uniform(rnd_key, (N,))
    y = jax.random.uniform(rnd_key, (N,))
    z = jax.random.uniform(rnd_key, (N,))
    p_values = jnp.array([evaluate_polynomial(coeffs, x[i], y[i], z[i]) for i in range(N)])
    p_with_noise = p_values + p_values * noise_frac * jax.random.normal(rnd_key, (N,))
    return jnp.stack((x, y, z, p_with_noise), axis=1)

# Ex 3.c
# Loss function comparing polynomial predictions with target values
def loss(coeffs, data):
    predictions = jnp.array([evaluate_polynomial(coeffs, x, y, z) for x, y, z, _ in data])
    actual = data[:, 3]  # True values are in the last column
    return jnp.mean((predictions - actual) ** 2)

# Training function using SGD
def sgd_poly_reconstruction(data, N_x, N_y, N_z, t, learning_rate=0.01, num_epochs=300):
    # Initialize coefficients with t non-zero entries randomly selected
    rng_key = jax.random.PRNGKey(0)
    coeffs = jnp.zeros((N_x + 1, N_y + 1, N_z + 1))
    
    # Randomly select t unique positions to be non-zero
    total_terms = (N_x + 1) * (N_y + 1) * (N_z + 1)
    selected_indices = jax.random.choice(rng_key, total_terms, shape=(t,), replace=False)
    
    # Fill selected positions with random values
    flat_coeffs = coeffs.reshape(-1)
    random_values = jax.random.uniform(rng_key, shape=(t,))
    flat_coeffs = flat_coeffs.at[selected_indices].set(random_values)
    coeffs = flat_coeffs.reshape((N_x + 1, N_y + 1, N_z + 1))

    # Compute gradient of the loss function
    grad_loss = jax.grad(loss)

    # Run SGD
    for epoch in range(num_epochs):
        grad = grad_loss(coeffs, data)
        coeffs = coeffs - learning_rate * grad
        if epoch % 10 == 0:
            current_loss = loss(coeffs, data)
            print(f"Epoch {epoch}: Loss = {current_loss}")

    return coeffs

if __name__ == "__main__":
    N = 15
    test1 = False
    test2 = False
    if(test1):
    # Test 1: Nx = 2, Ny = 4, Nz = 6, t = 12
        print("Testing polynomial 1 with Nx=2, Ny=4, Nz=6, t=12")
        Nx, Ny, Nz, t = 2, 4, 6, 12
        coeffs = jnp.zeros((Nx + 1, Ny + 1, Nz + 1))
        rng_key = jax.random.PRNGKey(2)
        selected_indices = jax.random.choice(rng_key, coeffs.size, shape=(t,), replace=False)
        flat_coeffs = coeffs.reshape(-1).at[selected_indices].set(jax.random.uniform(rng_key, shape=(t,)))
        coeffs = flat_coeffs.reshape((Nx + 1, Ny + 1, Nz + 1))
        data = generate_training_data(coeffs, N)
        trained_coeffs_1 = sgd_poly_reconstruction(data, Nx, Ny, Nz, t, learning_rate=0.01, num_epochs=100)

    if(test2):    
        # Test 2: Nx = 3, Ny = 1, Nz = 2, t = 5
        print("\nTesting polynomial 2 with Nx=3, Ny=1, Nz=2, t=5")
        Nx, Ny, Nz, t = 3, 1, 2, 5
        coeffs = jnp.zeros((Nx + 1, Ny + 1, Nz + 1))
        rng_key = jax.random.PRNGKey(3)
        selected_indices = jax.random.choice(rng_key, coeffs.size, shape=(t,), replace=False)
        flat_coeffs = coeffs.reshape(-1).at[selected_indices].set(jax.random.uniform(rng_key, shape=(t,)))
        coeffs = flat_coeffs.reshape((Nx + 1, Ny + 1, Nz + 1))
        data = generate_training_data(coeffs,N)
        trained_coeffs_2 = sgd_poly_reconstruction(data, Nx, Ny, Nz, t, learning_rate=0.01, num_epochs=100)

    # Load data from the provided file
    data_file_path = 'mmd_data_secret_polyxyz.npy'
    data = np.load(data_file_path)
    data = jnp.array(data)  # Convert to JAX array

    # Define the problem parameters based on the exercise requirements
    N_x, N_y, N_z, t = 2, 2, 1, 5  # Given in the exercise instructions

    # Run SGD to reconstruct the polynomial coefficients
    coeffs = sgd_poly_reconstruction(data, N_x, N_y, N_z, t, learning_rate=0.01, num_epochs=50)

    # Output the reconstructed coefficients
    print("Reconstructed coefficients:")
    print(coeffs)