"""TODO(jmdm): description of script.

Date:       2025-07-08
Status:     In progress ⚙️

Notes
-----
    *

Todo:
----
    [ ] ".rotate" as superclass method?
    [ ] Better documentation

References
----------
    [1]

"""

# Standard library
import math

# Third-party libraries
import mujoco
from PIL import Image
from rich.console import Console

from ariel.utils.video_recorder import VideoRecorder

# Global functions
console = Console()


def single_frame_renderer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    steps: int = 10,
) -> None:
    """
    Render a single frame of the simulation using MuJoCo's rendering engine.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model to render.
    data : mujoco.MjData
        The MuJoCo data to render.
    steps : int, optional
        The number of simulation steps to take before rendering, by default 10
    """
    # Enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Call rendering engine
    msg = f"Rendering single frame with [bold blue] {steps} [/bold blue] steps."
    console.log(f"[bold yellow] --> {msg} [/bold yellow]")
    with mujoco.Renderer(model) as renderer:
        # Move simulation forward one iteration/step
        mujoco.mj_step(model, data, nstep=steps)

        # Update rendering engine
        renderer.update_scene(data, scene_option=scene_option)

        # Generate frame using rendering engine
        frame = renderer.render()

        # Convert frame into an image which can be shown
        img = Image.fromarray(frame)
        img.show()
    console.log("[bold green] --> Rendering done![/bold green]")


def video_renderer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float = 10.0,
    video_recorder: VideoRecorder | None = None,
) -> None:
    """
    Render a video of the simulation using MuJoCo's rendering engine.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model to render.
    data : mujoco.MjData
        The MuJoCo data to render.
    duration : float, optional
        The duration of the video in seconds, by default 10.0
    video_recorder : VideoRecorder | None, optional
        The video recorder to use, by default None
    """
    # Get video recorder
    if video_recorder is None:
        video_recorder = VideoRecorder()

    # Enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Calculate steps per frame to avoid single iterations (see 'Notes'.)
    options = mujoco.MjOption()
    steps_per_frame = duration / (
        options.timestep * duration * video_recorder.fps
    )

    # Call rendering engine
    with mujoco.Renderer(
        model,
        width=video_recorder.width,
        height=video_recorder.height,
    ) as renderer:
        while data.time < duration:
            # Move simulation forward one iteration/step
            mujoco.mj_step(model, data, nstep=math.floor(steps_per_frame))

            # Update rendering engine
            renderer.update_scene(data, scene_option=scene_option)

            # Save frame
            video_recorder.write(frame=renderer.render())

    # Exit (and save locally) the generated video
    console.log(video_recorder.frame_count)
    video_recorder.release()
