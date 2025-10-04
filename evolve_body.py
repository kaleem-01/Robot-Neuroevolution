"""
Minimal NN-controlled MuJoCo run using your `nn_controller` + `experiment`.
- Keeps your function signatures intact.
- Adds a Controller wrapper, safe scaling to ctrlrange, and a runnable __main__.
"""

# --- Standard library
from pathlib import Path
from typing import Any, Literal, Optional
import os

# --- Third-party
import numpy as np
import numpy.typing as npt
import mujoco as mj
from mujoco import viewer

# --- Local (from your template/context)
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena

# If you have a handy prebuilt body, we can demo with it:
try:
    from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko as build_gecko
except Exception:
    build_gecko = None  # Fallback if not available

# ---------- Globals & types ----------
ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

SEED = int(os.environ.get("SEED", "42"))
RNG = np.random.default_rng(SEED)

ROOT = Path.cwd()
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True, parents=True)

# A reasonable spawn position so the robot doesn't start intersecting the ground.
SPAWN_POS = np.array([0.0, 0.0, 0.12])

# ---------- Helper: scale NN outputs into actuator ctrl ranges ----------
def _scale_to_ctrlrange(model: mj.MjModel, raw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Map raw actions in [-pi, pi] to each actuator's ctrlrange.
    Falls back to clipping if an actuator lacks a range.
    """
    assert raw.shape[0] == model.nu, f"Expected {model.nu} controls, got {raw.shape[0]}."
    # MuJoCo stores ctrlrange as shape (nu, 2): [lo, hi] per actuator
    cr = model.actuator_ctrlrange  # (nu, 2) float64
    out = raw.copy()
    for i in range(model.nu):
        lo, hi = cr[i]
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            # raw is ~[-pi, pi]; scale to [lo, hi]
            out[i] = lo + (raw[i] + np.pi) * (hi - lo) / (2.0 * np.pi)
            # Safety clip
            if out[i] < lo: out[i] = lo
            if out[i] > hi: out[i] = hi
        else:
            # If no valid range, just clip to [-1, 1]
            if out[i] < -1.0: out[i] = -1.0
            if out[i] > 1.0:  out[i] =  1.0
    return out


# =========================
# Your provided functions
# =========================
def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    # Normally, you would use the genes of an individual as the weights,
    # Here we set them randomly for simplicity.
    w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs
    return outputs * np.pi


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            from ariel.utils.runners import simple_runner
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            from ariel.utils.renderers import single_frame_renderer
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            from ariel.utils.renderers import video_renderer
            from ariel.utils.video_recorder import VideoRecorder
            path_to_video_folder = str(DATA / "videos")
            (DATA / "videos").mkdir(exist_ok=True, parents=True)
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "launcher":
            # This opens a live viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )
    # ==================================================================== #


# =========================
# Extra: a Controller wrapper that uses your nn_controller
# =========================
class SimpleNNController(Controller):
    """
    Minimal Controller that:
    - calls your `nn_controller(model, data)` to get an action each step,
    - scales it safely into each actuator's control range,
    - writes to data.ctrl in-place.
    """
    def __init__(self, tracker: Optional[Any] = None, action_gain: float = 1.0):
        super().__init__(tracker=tracker)
        self.action_gain = action_gain

    def set_control(self, model: mj.MjModel, data: mj.MjData, *args, **kwargs) -> None:
        # 1) get raw action from your NN
        raw = nn_controller(model, data)  # shape (nu,)

        # 2) optional gain
        raw = np.asarray(raw) * float(self.action_gain)

        # 3) scale/clamp to ctrlrange
        u = _scale_to_ctrlrange(model, raw)

        # 4) write to MuJoCo
        data.ctrl[:] = u


# =========================
# Example: build a robot and run
# =========================
def _example_robot():
    """
    Try to build a known-good robot phenotype (gecko) if available,
    otherwise raise with a helpful message.
    """
    if build_gecko is None:
        raise RuntimeError(
            "Couldn't import prebuilt 'gecko' robot. "
            "Provide a robot with a `.spec` compatible with OlympicArena.spawn()."
        )
    return build_gecko()  # returns an object with `.spec`


if __name__ == "__main__":
    # Reproducibility
    np.random.seed(SEED)

    # Build a robot phenotype
    robot = _example_robot()

    # Make a controller (you can pass a Tracker if you have one)
    ctrl = SimpleNNController(action_gain=1.0)

    # Run! Options: "simple" (fast, headless), "launcher" (interactive), "video", "frame", "no_control"
    experiment(
        robot=robot,
        controller=ctrl,
        duration=15,          # seconds
        mode="launcher",      # change to "simple" for a quick headless run
    )
