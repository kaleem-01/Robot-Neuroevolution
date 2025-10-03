
import numpy as np
import matplotlib.pyplot as plt

def Ackley(x):
    """source: https://www.sfu.ca/~ssurjano/ackley.html"""

    # Ackley function parameters
    a = 20
    b = 0.2
    c = 2 * np.pi
    dimension = len(x)

    # Individual terms
    term1 = -a * np.exp(-b * np.sqrt(sum(x**2) / dimension))
    term2 = -np.exp(sum(np.cos(c * xi) for xi in x) / dimension)
    return term1 + term2 + a + np.exp(1)

def fitness_landscape_plot():
    # Generate data for plotting
    boundary_point, resolution = 5, 500
    x = np.linspace(-boundary_point, boundary_point, resolution)
    y = np.linspace(-boundary_point, boundary_point, resolution)

    # Generate the coordinate points
    X, Y = np.meshgrid(x, y)
    positions = np.column_stack([X.ravel(), Y.ravel()])

    # Get depths for all coordinate positions
    z_multimodal = np.array(list(map(Ackley, positions))).reshape([resolution, resolution])

    # Create 3D plot
    fig = plt.figure(figsize=(15, 8))

    titles = ["Ackley Function"]
    for idx, z in enumerate([z_multimodal]):
        # Create sub-plot
        ax = fig.add_subplot(121 + idx, projection="3d")

        # Plot the surface
        ax.plot_surface(X, Y, z, cmap="viridis", edgecolor="k")

        # Set labels
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title(titles[idx])
        # ax.autoscale(True)

    # Show the plot
    plt.tight_layout()
    plt.show()