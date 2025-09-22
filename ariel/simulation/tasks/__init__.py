from .gate_learning import xy_displacement_ff, x_speed_ff, y_speed_ff
from .targeted_locomotion import distance_to_target_ff
from .turning_in_place import turning_in_place_ff

tasks = [
    "Gate Learning",
    "Targeted Locomotion",
    "Turning In Place"]

_task_fitness_function_map_ = {
    "Gate Learning": [xy_displacement_ff, x_speed_ff, y_speed_ff],
    "Targeted Locomotion": [distance_to_target_ff],
    "Turning In Place": [turning_in_place_ff],
}
