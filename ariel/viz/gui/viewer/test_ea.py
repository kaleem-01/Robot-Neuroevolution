# --------------------------------------------------------------------
# from ariel.viz.analysis.parse_db import get_fitness_over_generations
# from results_viewer import EmbeddedPlotWidget

from QNodeEditor import Node
from PyQt5.QtWidgets import QApplication, QPushButton, QGraphicsProxyWidget
from QNodeEditor import NodeEditorDialog
from ariel.ec.a004 import BasicEA, EAStep, create_individual, \
    parent_selection as ps_op, crossover as co_op, mutation as mut_op, \
    evaluate as eval_op, survivor_selection as ss_op
from rich.console import Console

console = Console()

# ----------------------------
# EA Parameters Node
# ----------------------------
class EAParameters(Node):
    code = 0

    def create(self):
        self.title = 'Evolutionary Algorithm Parameters'

        # Editable values
        self.add_value_input("Population Size")
        self.add_value_input("Generations")

        # Outputs (different names to avoid clashes)
        self.add_label_output("PopSizeOut")
        self.add_label_output("GenerationsOut")

    def evaluate(self, values: dict):
        pop_size = values["Population Size"]
        generations = values["Generations"]

        print(f"EAParameters -> Pop={pop_size}, Gen={generations}")

        # Send values forward with output names
        self.set_output_value("PopSizeOut", pop_size)
        self.set_output_value("GenerationsOut", generations)

# ----------------------------
# EA Run Node
# ----------------------------
class EARun(Node):
    code = 1

    def create(self):
        self.title = 'Run EA'

        # Inputs from EAParameters
        self.add_value_input("PopSizeOut")
        self.add_value_input("GenerationsOut")

        # Optional: Add an output for the result
        self.add_label_output("Best Individual")

    def evaluate(self, values: dict):

        """Run the EA using the parameters provided by EAParameters."""
        # Get parameter values from inputs
        pop_size = values["PopSizeOut"]
        generations = values["GenerationsOut"]

        console.log(f"Running EA -> with population={pop_size}, generations={generations}")

        # Build initial population
        population_list = [create_individual() for _ in range(int(pop_size))]
        population_list = eval_op(population_list)

        # Define EA steps
        ops = [
            EAStep("parent_selection", ps_op),
            EAStep("crossover", co_op),
            EAStep("mutation", mut_op),
            EAStep("evaluation", eval_op),
            EAStep("survivor_selection", ss_op),
        ]

        # Run EA
        ea = BasicEA(population_list, operations=ops, num_of_generations=int(generations))
        ea.run()

        # Collect solutions
        best = ea.get_solution(only_alive=False)
        median = ea.get_solution("median", only_alive=False)
        worst = ea.get_solution("worst", only_alive=False)

        console.log(f"[green]Best:[/green] {best}")
        console.log(f"[yellow]Median:[/yellow] {median}")
        console.log(f"[red]Worst:[/red] {worst}")

        # Save result to output so other nodes can use it
        self.set_output_value("Best Individual", best)
        self.result = best 

        return best

class EAResult(Node):
    code = 2

    def create(self):
        self.title = "EA Result - Best Individual"
        self.add_value_input("Result")  # Input from EARun

        print('The Best Individual from EARun is : ', self.get_entry("Result").calculate_value())

# ----------------------------
# Main
# ----------------------------
app = QApplication([])
dialog = NodeEditorDialog()
dialog.editor.available_nodes = {
    'EA Parameters': EAParameters,
    'Run EA': EARun,
    'EA Result' : EAResult
}

# Add output node instance
ea_result_node = EAResult()
dialog.editor.output_node = ea_result_node

if dialog.exec():
    print("Editor finished. Best individual:",  dialog.result)

# ------------------------------------------------------------------------------
# from QNodeEditor import Node
# from PyQt5.QtWidgets import QApplication, QPushButton, QGraphicsProxyWidget
# from QNodeEditor import NodeEditorDialog

# class Addition(Node):
#     code = 0

#     def create(self):
#         self.title = 'Add Two Numbers'
#         self.add_label_output('Output')  # This line is super weird, but necessary to create an output

#         self.add_value_input('Value 1')  # Add input (value_input means numerical input) for the first value
#         self.add_value_input('Value 2')  # same for the second value

#     def evaluate(self, values: dict):
#         value1 = values['Value 1'] # Somehow evaluate() implicitly receives the input values as a dictionary
#         value2 = values['Value 2']

#         self.set_output_value('Output', value1 + value2) # This is where we actually set the value of the output
#                                                          # that we created above with add_label_output()

# class Output(Node):
#     code = 1
#     def create(self):
#         self.title = 'Output Node'
#         self.add_value_input('Out')  # You have to pay attention to the naming very carefully  

# app = QApplication([])
# dialog = NodeEditorDialog()
# dialog.editor.available_nodes = {
#     'Add': Addition,
#     'Out': Output,
# }
# # Add output node instance we need to have at least one
# # (I'm not sure if we can have multiple but it kind of seems like we can!)
# output_node = Output()
# dialog.editor.output_node = output_node

# if dialog.exec():
#     print("Editor finished...",  dialog.result)
