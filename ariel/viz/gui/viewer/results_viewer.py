from PyQt5.QtWidgets import (QWidget, QVBoxLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer
import numpy as np


class EmbeddedPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create a layout for the widget
        layout = QVBoxLayout()
        
        # Create a matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Add canvas to layout
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_fitness(self, fitnesses_per_generation):
        """
        Plot fitness over generations, averaged.
        fitnesses_per_generation: list of lists of fitness values
        """
        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Collect statistics across generations
        generations = list(range(len(fitnesses_per_generation)))
        max_fitness = [max(gen) for gen in fitnesses_per_generation]
        mean_fitness = [np.mean(gen) for gen in fitnesses_per_generation]
        std_fitness = [np.std(gen) for gen in fitnesses_per_generation]
        
        # Plot max fitness
        ax.plot(
            generations,
            max_fitness,
            label="Max fitness",
            color="b",
        )
        ax.fill_between(
            generations,
            np.array(max_fitness) - np.array(std_fitness),
            np.array(max_fitness) + np.array(std_fitness),
            color="b",
            alpha=0.2,
        )
        
        # Plot mean fitness
        ax.plot(
            generations,
            mean_fitness,
            label="Mean fitness",
            color="r",
        )
        ax.fill_between(
            generations,
            np.array(mean_fitness) - np.array(std_fitness),
            np.array(mean_fitness) + np.array(std_fitness),
            color="r",
            alpha=0.2,
        )
        
        # Customize plot
        ax.set_xlabel("Generation index")
        ax.set_ylabel("Fitness")
        ax.set_title("Mean and max fitness with std as shade")
        ax.legend()
        
        # Adjust layout and redraw
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Optionally save the figure
        self.figure.savefig("src/ariel/viz/gui/resources/example.png")


# class EmbeddedPlotWidgetDynamic(QWidget):
#     def __init__(self, parent=None, num_generations=10, update_interval=1000):
#         super().__init__(parent)
        
#         # Create a layout for the widget
#         layout = QVBoxLayout()
        
#         # Create a matplotlib figure and canvas
#         self.figure = Figure(figsize=(10, 6), dpi=100)
#         self.canvas = FigureCanvas(self.figure)
        
#         # Add canvas to layout
#         layout.addWidget(self.canvas)
#         self.setLayout(layout)

#         # Store database path
#         self.database_path = None
        
#         # Setup timer for periodic updates
#         self.update_timer = QTimer(self)
#         self.update_timer.timeout.connect(self.update_plot)
#         self.update_timer.start(update_interval)  # Update every second
        
#         # Flag to prevent multiple simultaneous updates
#         self.is_updating = False
#         self.num_generations = num_generations

#         self.draw_base_plot(self.figure.add_subplot(111))

#     def set_database_path(self, database_path):
#         self.database_path = database_path

#     def set_num_generations(self, num_generations):
#         self.num_generations = num_generations

#     def update_plot(self):
#         """
#         Update the plot with the latest data from the database.
#         """
#         if self.is_updating or not self.database_path:
#             return

#         try:
#             self.is_updating = True

#             # Clear previous plot
#             self.figure.clear()
#             ax = self.figure.add_subplot(111)

#             # Open database
#             dbengine = open_database_sqlite(
#                 self.database_path,
#                 open_method=OpenMethod.OPEN_IF_EXISTS
#             )

#             # Read data
#             df = pd.read_sql(
#                 select(
#                     Experiment.id.label("experiment_id"),
#                     Generation.generation_index,
#                     Individual.fitness,
#                 )
#                 .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
#                 .join_from(Generation, Population, Generation.population_id == Population.id)
#                 .join_from(Population, Individual, Population.id == Individual.population_id),
#                 dbengine,
#             )

#             if df.empty or not ((df["generation_index"] == 0) & df["fitness"].notna()).any():
#                 self.draw_base_plot(ax)  # Show base plot
#                 return

#             # Aggregate data
#             agg_per_experiment_per_generation = (
#                 df.groupby(["experiment_id", "generation_index"])
#                 .agg({"fitness": ["max", "mean"]})
#                 .reset_index()
#             )
#             agg_per_experiment_per_generation.columns = [
#                 "experiment_id",
#                 "generation_index",
#                 "max_fitness",
#                 "mean_fitness",
#             ]

#             agg_per_generation = (
#                 agg_per_experiment_per_generation.groupby("generation_index")
#                 .agg({"max_fitness": ["mean", "std"], "mean_fitness": ["mean", "std"]})
#                 .reset_index()
#             )
#             agg_per_generation.columns = [
#                 "generation_index",
#                 "max_fitness",
#                 "max_fitness_std",
#                 "mean_fitness",
#                 "mean_fitness_std",
#             ]

#             ax.set_xlim(0, self.num_generations)

#             # Plot max fitness
#             ax.plot(
#                 agg_per_generation["generation_index"],
#                 agg_per_generation["max_fitness"],
#                 label="Max fitness",
#                 color="b",
#             )
#             ax.fill_between(
#                 agg_per_generation["generation_index"],
#                 agg_per_generation["max_fitness"] - agg_per_generation["max_fitness_std"],
#                 agg_per_generation["max_fitness"] + agg_per_generation["max_fitness_std"],
#                 color="b",
#                 alpha=0.2,
#             )

#             # Plot mean fitness
#             ax.plot(
#                 agg_per_generation["generation_index"],
#                 agg_per_generation["mean_fitness"],
#                 label="Mean fitness",
#                 color="r",
#             )
#             ax.fill_between(
#                 agg_per_generation["generation_index"],
#                 agg_per_generation["mean_fitness"] - agg_per_generation["mean_fitness_std"],
#                 agg_per_generation["mean_fitness"] + agg_per_generation["mean_fitness_std"],
#                 color="r",
#                 alpha=0.2,
#             )

#             # Customize plot
#             ax.set_xlabel("Generation index")
#             ax.set_ylabel("Fitness")
#             ax.set_title("Mean and max fitness across repetitions with std as shade")
#             ax.legend()

#             # Adjust layout and redraw
#             self.figure.tight_layout()
#             self.canvas.draw()

#         except Exception as e:
#             print(f"Error updating plot: {e}")

#         finally:
#             self.is_updating = False


#     def draw_base_plot(self, ax):
#         """
#         Draws a placeholder base plot until valid data is available.
#         """
#         ax.set_xlabel("Generation index")
#         ax.set_ylabel("Fitness")
#         ax.set_title("No data yet...")
#         ax.text(0.5, 0.5, "Waiting for first generation to complete", fontsize=14, ha="center", va="center", transform=ax.transAxes)
#         self.figure.tight_layout()
#         self.canvas.draw()
