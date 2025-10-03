"""TODO(jmdm): description of script.

Author:     jmdm
Date:       2025-07-08
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro
Status:     In progress ⚙️

Notes
-----
    *

References
----------
    [1]

Todo
----
    [ ] documentation

"""

# Third-party libraries
import matplotlib.pyplot as plt

# Local libraries
from ariel.simulation.controllers.hopfs_cpg import HopfCPG


def main() -> None:
    """Entry point."""
    # Connection list
    num_neurons = 4
    adjacency_list: dict[int, list[int]] = {}
    for i in range(num_neurons):
        adjacency_list[i] = []
        for j in [i - 1, i + 1]:
            if 0 <= j < num_neurons:
                adjacency_list[i].append(j)

    # Create a CPG network with 4 neurons
    cpg = HopfCPG(num_neurons=num_neurons, adjacency_list=adjacency_list)

    # Simulate for 1000 steps
    x_history, _ = cpg.simulate(1_000)

    # Plot the results
    plt.figure(figsize=(10, 6))
    for i in range(cpg.num_neurons):
        plt.plot(x_history[:, i], label=f"Neuron {i + 1}")

    # Plotting settings
    plt.title("Hopf-based CPG Output")
    plt.xlabel("Time step")
    plt.ylabel("x output")
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
