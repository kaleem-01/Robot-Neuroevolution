# =========================
# Evolutionary Gecko Controller (MuJoCo)
# =========================
# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Callable
import pickle
import threading

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.simulation.environments.rugged_heightmap import RuggedTerrainWorld
from ariel.simulation.environments.simple_tilted_world import TiltedFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
import multiprocessing


# =========================
# Config
# =========================
SEED = 42
SIM_DURATION = 10.0          # seconds per evaluation
SMOOTH_ALPHA = 0.2          # 0..1, higher tracks target faster (smoother with lower)
POP_SIZE = 50
ELITES = POP_SIZE // 10                   # number of elites to carry over each generation
GENERATIONS = 20
MUTATION_SIGMA = 0.25       # base mutation scale
TOURNAMENT_K = 5            # tournament size for parent selection

# IMPORTANT: make these positive if you want *penalties*
ENERGY_PENALTY = 0.00       # penalty * mean(|ctrl|)
SPEED_PENALTY  = -0.05       # penalty * mean(|dctrl|)

# Frequency bounds (shared across joints)
FREQ_MIN = 0.5
FREQ_MAX = 2.0

# Joint limits (hinges)
CTRL_MIN = -np.pi/2
CTRL_MAX =  np.pi/2

# Keep track of data / history for the "watch best" run

# =========================
# Helpers: world + model
# =========================
def build_world_and_model(terrain=None):
    mujoco.set_mjcb_control(None)  # reset callbacks (as recommended)

    if terrain == "rough":
        world = RuggedTerrainWorld()
    elif terrain == "tilted":
        world = TiltedFlatWorld()
    else:   #TODO TiltedFlatWorld
        world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    if not to_track:
        raise RuntimeError("Couldn't find core geom to track.")
    return world, gecko_core, model, data, to_track


# =========================
# Controller parameterization
# =========================
@dataclass
class ControllerParams:
    amplitudes: np.ndarray   # shape (nu,), each in [0,1]
    phases: np.ndarray       # shape (nu,), each in [-pi, pi]
    frequency: float         # shared, in [FREQ_MIN, FREQ_MAX]


def sample_params(rng: np.random.Generator, nu: int) -> ControllerParams:
    A = rng.uniform(0.2, 0.9, size=nu)     # start with decent amplitudes
    P = rng.uniform(-np.pi, np.pi, size=nu)
    f = rng.uniform(FREQ_MIN, FREQ_MAX)
    return ControllerParams(A, P, f)


def clip_params(p: ControllerParams) -> ControllerParams:
    p.amplitudes = np.clip(p.amplitudes, 0.0, 1.0)
    # wrap phases to [-pi, pi]
    p.phases = ((p.phases + np.pi) % (2*np.pi)) - np.pi
    p.frequency = float(np.clip(p.frequency, FREQ_MIN, FREQ_MAX))
    return p


def blend_crossover(rng: np.random.Generator, p1: ControllerParams, p2: ControllerParams) -> ControllerParams:
    # BLX-alpha style blend
    alpha = 0.5
    def blend(a,b):
        lo, hi = np.minimum(a,b), np.maximum(a,b)
        span = hi - lo
        lo_b = lo - alpha * span
        hi_b = hi + alpha * span
        return rng.uniform(lo_b, hi_b)

    A = blend(p1.amplitudes, p2.amplitudes)
    P = blend(p1.phases, p2.phases)
    f = blend(np.array([p1.frequency]), np.array([p2.frequency]))[0]
    return clip_params(ControllerParams(A, P, f))


def mutate(rng: np.random.Generator, p: ControllerParams, sigma=MUTATION_SIGMA) -> ControllerParams:
    A = p.amplitudes + rng.normal(0.0, sigma, size=p.amplitudes.shape)
    P = p.phases     + rng.normal(0.0, sigma*np.pi, size=p.phases.shape)
    f = p.frequency  + rng.normal(0.0, sigma)
    return clip_params(ControllerParams(A, P, f))


# =========================
# Controller callback
# =========================
def make_control_fn(params: ControllerParams) -> Callable[[mujoco.MjModel, mujoco.MjData], None]:
    """
    Returns a MuJoCo control callback that sets data.ctrl each step
    using sinusoidal targets tracked with exponential smoothing.
    """
    # we keep previous ctrl inside the closure for smoothness cost
    prev_ctrl = {"val": None}  # dict to allow write in closure

    def ctrl_fn(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        t = data.time  # MuJoCo sim time
        nu = model.nu
        # Make sure ctrl shapes are OK
        if data.ctrl.shape[0] != nu:
            raise RuntimeError(f"Expected ctrl size {nu}, got {data.ctrl.shape[0]}")

        # target angle for each joint
        # target_i(t) = A_i * (pi/2) * sin(2*pi*f*t + phase_i)
        target = params.amplitudes * (np.pi/2) * np.sin(2*np.pi*params.frequency * t + params.phases)

        # smooth tracking: ctrl <- (1-alpha)*ctrl + alpha*target
        if prev_ctrl["val"] is None:
            # initialize smoothly from zero
            new_ctrl = SMOOTH_ALPHA * target
        else:
            new_ctrl = (1.0 - SMOOTH_ALPHA) * prev_ctrl["val"] + SMOOTH_ALPHA * target

        # clip to hinge range
        new_ctrl = np.clip(new_ctrl, CTRL_MIN, CTRL_MAX)

        # set and store
        data.ctrl[:] = new_ctrl
        prev_ctrl["val"] = new_ctrl.copy()

    return ctrl_fn


# =========================
# Rollout / Fitness
# =========================
@dataclass
class FitnessResult:
    fitness: float
    x_progress: float
    # y_progess: float
    energy: float
    speed: float


def rollout_and_score(params: ControllerParams, duration=SIM_DURATION, terrain="flat") -> FitnessResult:
    """
    Headless simulation: returns fitness and components.
    """
    # _, _, _, _, to_track = build_world_and_model(terrain=terrain)
    
    
    # set control callback
    mujoco.set_mjcb_control(make_control_fn(params))

    # Identify timestep & steps
    dt = model.opt.timestep
    n_steps = int(np.ceil(duration / dt))

    # direction =  # 1 is forward direction, 0 is Right direction???,  2 is turning around
    # track core geom x position and control stats
    start_x = to_track[0].xpos[1]
    last_ctrl = data.ctrl.copy()
    energy_acc = 0.0
    speed_acc = 0.0
    count = 0

    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        # accumulate simple energy/speed proxies
        energy_acc += float(np.mean(np.abs(data.ctrl)))
        speed_acc  += float(np.mean(np.abs(data.ctrl - last_ctrl)))
        last_ctrl = data.ctrl.copy()
        count += 1

    end_x = to_track[0].xpos[1]
    x_prog = float(end_x - start_x)
    # y_prog = float(end_y - start_y)
    mean_energy = energy_acc / max(1, count)
    mean_speed  = speed_acc  / max(1, count)

    # Fitness: forward progress minus small penalties
    fitness = x_prog - ENERGY_PENALTY * mean_energy - SPEED_PENALTY * mean_speed # - 0.2  * abs(y_prog) # penality for movement in y direction
    return FitnessResult(fitness, x_prog, mean_energy, mean_speed)


# =========================
# Evolutionary Algorithm
# =========================
def tournament_select(rng, pop, fits, k=TOURNAMENT_K):
    idxs = rng.choice(len(pop), size=k, replace=False)
    best = max(idxs, key=lambda i: fits[i].fitness)
    return pop[best]


def evolve(terrain="flat"):
    """
    Runs an EA to evolve ControllerParams.
    Returns best params, its FitnessResult, and both curves:
      - gen_best_curve: best-of-generation scores
      - overall_best_curve: running best-so-far (monotonic)
    """
    # _, _, model, _, _ = build_world_and_model(terrain=terrain)
    
    
    # We need nu (number of actuated joints)
    nu = model.nu

    rng = np.random.default_rng(SEED)

    # Initialize population
    population = [sample_params(rng, nu) for _ in range(POP_SIZE)]

    best_overall = None
    best_fit: FitnessResult | None = None
    gen_best_curve = []
    overall_best_curve = []

    for gen in range(GENERATIONS):
        # Evaluate        
        with multiprocessing.Pool() as pool:
            fitnesses = pool.starmap(
            rollout_and_score,
            [(ind, SIM_DURATION, terrain) for ind in population]
            )

        # Track best of generation
        gen_best_idx = int(np.argmax([f.fitness for f in fitnesses]))
        gen_best = population[gen_best_idx]
        gen_best_fit = fitnesses[gen_best_idx]

        # Update overall best
        if (best_fit is None) or (gen_best_fit.fitness > best_fit.fitness):
            best_overall = ControllerParams(gen_best.amplitudes.copy(),
                                            gen_best.phases.copy(),
                                            gen_best.frequency)
            best_fit = gen_best_fit

        print(f"[Gen {gen+1:02d}] Best fitness: {gen_best_fit.fitness:.4f} | "
              f"X+ {gen_best_fit.x_progress:.3f}  | "
#              f"Y+ {gen_best_fit.y_progess:.3f}  | "
              f"energy {gen_best_fit.energy:.3f}  | speed {gen_best_fit.speed:.3f}  | f={gen_best.frequency:.2f}Hz")

        gen_best_curve.append(gen_best_fit.fitness)
        overall_best_curve.append(best_fit.fitness)  # monotonic non-decreasing

        # Elitism
        elite_idxs = np.argsort([-f.fitness for f in fitnesses])[:ELITES]
        elites = [population[i] for i in elite_idxs]
        print(f"Elites' fitnesses: {[fitnesses[i].fitness for i in elite_idxs]}")

        # Create next population
        next_pop = elites.copy()
        while len(next_pop) < POP_SIZE:
            # Parents via tournament
            p1 = tournament_select(rng, population, fitnesses)
            p2 = tournament_select(rng, population, fitnesses)
            child = blend_crossover(rng, p1, p2)
            # Adaptive-ish mutation: shrink a bit as gens go (optional)
            sigma = MUTATION_SIGMA * (0.9 ** gen)

            # sigma = MUTATION_SIGMA
            child = mutate(rng, child, sigma=sigma)
            next_pop.append(child)
        population = next_pop

    print("\n=== Evolution complete ===")
    assert best_fit is not None and best_overall is not None
    print(f"Best fitness: {best_fit.fitness:.4f} | X+ {best_fit.x_progress:.3f} | f={best_overall.frequency:.2f}Hz")
    return best_overall, best_fit, gen_best_curve, overall_best_curve


# =========================
# Viewer playback for best
# =========================
def show_qpos_history(history: List[np.ndarray]):
    pos_data = np.array(history)
    plt.figure(figsize=(8, 6))
    plt.plot(pos_data[:, 0], pos_data[:, 1], label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'o', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'x', label='End')
    plt.xlabel('X Position'); plt.ylabel('Y Position')
    plt.title('Robot Path in XY Plane')
    plt.legend(); plt.grid(True)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)
    plt.xlim(-max_range, max_range); plt.ylim(-max_range, max_range)
    plt.show()


def watch_controller(params: ControllerParams, duration=1, terrain="flat"):
    """
    Launches viewer to visualize the learned controller and records XY history.
    """
    global HISTORY
    HISTORY = []
    
    # Wrap the control to also log the core position
    base_ctrl = make_control_fn(params)
    def ctrl_and_log(m, d):
        base_ctrl(m, d)
        HISTORY.append(to_track[0].xpos.copy())
        # print(f"Time: {d.time:.2f}s, Y- Pos: {to_track[0].xpos[1]}, X- Pos: {to_track[0].xpos[0]}")

    mujoco.set_mjcb_control(ctrl_and_log)

    # Run viewer; you can close window any time
    viewer.launch(model=model, data=data)

    # After you close the viewer, plot the path
    if HISTORY:
        show_qpos_history(HISTORY)


# =========================
# Main
# =========================
def main(terrain="flat", save=False):
    # 1) Evolve controller headlessly
    global model, data, to_track
    _, _, model, data, to_track = build_world_and_model(terrain=terrain)
    
    best_params, best_fit, gen_best_curve, overall_best_curve = evolve(terrain=terrain)


    if save:
        # 2) Watch best controller in viewer
        print("\nLaunching viewer with the best controller...")
        watch_controller(best_params, duration=SIM_DURATION, terrain=terrain)
        PATH_TO_VIDEO_FOLDER = "./videos"
        video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER, )
        mujoco.set_mjcb_control(make_control_fn(best_params))
        # Run the video renderer for SIM_DURATION seconds, then close automatically
        video_renderer(model, data, video_recorder=video_recorder, duration=SIM_DURATION)
        with open(f"best_params_{terrain}.pkl", "wb") as f:
            pickle.dump(best_params, f)
        with open("best_fit.pkl", "wb") as f:
            pickle.dump(best_fit, f)
        with open(f"gen_best_curve_{terrain}.pkl", "wb") as f:
            pickle.dump(gen_best_curve, f)
        with open(f"overall_best_curve_{terrain}.pkl", "wb") as f:
            pickle.dump(overall_best_curve, f)

    return best_params, best_fit, gen_best_curve, overall_best_curve




def random_move(model, data, to_track) -> None:
    """Generate random movements for the robot's joints.
    
    The mujoco.set_mjcb_control() function will always give 
    model and data as inputs to the function. Even if you don't use them,
    you need to have them as inputs.

    Parameters
    ----------

    model : mujoco.MjModel
        The MuJoCo model of the robot.
    data : mujoco.MjData
        The MuJoCo data of the robot.

    Returns
    -------
    None
        This function modifies the data.ctrl in place.
    """

    # Get the number of joints
    num_joints = model.nu 
    
    # Hinges take values between -pi/2 and pi/2
    hinge_range = np.pi/2
    rand_moves = np.random.uniform(low= -hinge_range, # -pi/2
                                   high=hinge_range, # pi/2
                                   size=num_joints) 

    # There are 2 ways to make movements:
    # 1. Set the control values directly (this might result in junky physics)
    # data.ctrl = rand_moves

    # 2. Add to the control values with a delta (this results in smoother physics)
    delta = 0.05
    data.ctrl += rand_moves * delta 

    # Bound the control values to be within the hinge limits.
    # If a value goes outside the bounds it might result in jittery movement.
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)

    # Save movement to history
    HISTORY.append(to_track[0].xpos.copy())

    ##############################################
    #
    # Take all the above into consideration when creating your controller
    # The input size, output size, output range
    # Your network might return ranges [-1,1], so you will need to scale it
    # to the expected [-pi/2, pi/2] range.
    # 
    # Or you might not need a delta and use the direct controller outputs
    #
    ##############################################


if __name__ == "__main__":
    terrain = "tilted"
    print(f"Running EA on {terrain} terrain...")
    print(f"  Population size: {POP_SIZE}")
    print(f"  Generations: {GENERATIONS}")
    
    overall_best_curves = []
    gen_best_curves = []
    best_fits = []
    best_params_list = []
    # Repeat 3 times to see variance
    for i in range(2):
    # Run main EA and watch best on selected terrain
        best_params, best_fit, gen_best_curve, overall_best_curve = main(terrain=terrain, save=True)
        overall_best_curves.append(overall_best_curve)
        gen_best_curves.append(gen_best_curve)
        best_fits.append(best_fit)
        best_params_list.append(best_params)


    # Plot both curves: gen-best (can dip) vs overall-best (monotonic)
    plt.figure(figsize=(8,5))
    # plt.plot(gen_best_curve, label="Generation Best")
    # plt.plot(overall_best_curve, label="Overall Best (monotonic)", linestyle="--")
    
    # Plot variance (shaded area) if running multiple trials
    if len(overall_best_curves) > 1:
        # Pad curves to same length if needed
        min_len = min(len(curve) for curve in overall_best_curves)
        curves = np.array([curve[:min_len] for curve in overall_best_curves])
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        plt.plot(mean_curve, label="Mean Overall Best", color="black")
        plt.fill_between(np.arange(min_len), mean_curve-std_curve, mean_curve+std_curve, color="gray", alpha=0.3, label="Std Dev")

    plt.title(f"Fitness Curves on {terrain.capitalize()} Terrain")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f"fitness_curve_{terrain}.png", dpi=800)
    plt.show()