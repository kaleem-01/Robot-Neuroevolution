# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Callable

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# =========================
# Config
# =========================
SEED = 42
SIM_DURATION = 6.0          # seconds per evaluation
SMOOTH_ALPHA = 0.2          # 0..1, higher tracks target faster (smoother with lower)
POP_SIZE = 30
ELITES = 4
GENERATIONS = 20
MUTATION_SIGMA = 0.15       # base mutation scale
TOURNAMENT_K = 3
ENERGY_PENALTY = 0.002      # penalty * mean(|ctrl|)
SPEED_PENALTY  = 0.001      # penalty * mean(|dctrl|)

# Frequency bounds (shared across joints)
FREQ_MIN = 0.5
FREQ_MAX = 2.0

# Joint limits (hinges)
CTRL_MIN = -np.pi/2
CTRL_MAX =  np.pi/2

# Keep track of data / history for the "watch best" run
HISTORY: List[np.ndarray] = []


# =========================
# Helpers: world + model
# =========================
def build_world_and_model():
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

world, gecko_core, model, data, to_track = build_world_and_model()


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
    energy: float
    speed: float


def rollout_and_score(params: ControllerParams, duration=SIM_DURATION, seed=SEED) -> FitnessResult:
    """
    Headless simulation: returns fitness and components.
    """
    # Fresh world/model per rollout to avoid state bleed
    rng = np.random.default_rng(seed)

    # set control callback
    mujoco.set_mjcb_control(make_control_fn(params))

    # Identify timestep & steps
    dt = model.opt.timestep
    n_steps = int(np.ceil(duration / dt))

    # track core geom x position and control stats
    start_x = to_track[0].xpos[0]
    last_ctrl = data.ctrl.copy()
    energy_acc = 0.0
    speed_acc = 0.0
    count = 0

    # headless stepping
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        # accumulate simple energy/speed proxies
        energy_acc += float(np.mean(np.abs(data.ctrl)))
        speed_acc  += float(np.mean(np.abs(data.ctrl - last_ctrl)))
        last_ctrl = data.ctrl.copy()
        count += 1

    end_x = to_track[0].xpos[0]
    x_prog = float(end_x - start_x)
    mean_energy = energy_acc / max(1, count)
    mean_speed  = speed_acc  / max(1, count)

    # Fitness: forward progress minus small penalties
    fitness = x_prog - ENERGY_PENALTY * mean_energy - SPEED_PENALTY * mean_speed
    return FitnessResult(fitness, x_prog, mean_energy, mean_speed)


# =========================
# Evolutionary Algorithm
# =========================
def tournament_select(rng, pop, fits, k=TOURNAMENT_K):
    idxs = rng.choice(len(pop), size=k, replace=False)
    best = max(idxs, key=lambda i: fits[i].fitness)
    return pop[best]


def evolve(model_for_nu=None):
    """
    Runs an EA to evolve ControllerParams.
    Returns best params and its fitness.
    """
    # We need nu (number of actuated joints)
    if model_for_nu is None:
        _, _, model_for_nu, _, _ = build_world_and_model()
    nu = model_for_nu.nu

    rng = np.random.default_rng(SEED)

    # Initialize population
    population = [sample_params(rng, nu) for _ in range(POP_SIZE)]

    best_overall = None
    best_fit = None

    for gen in range(GENERATIONS):
        # Evaluate

        fitnesses = [rollout_and_score(ind, seed=SEED + gen) for ind in population]

        # Track best
        gen_best_idx = int(np.argmax([f.fitness for f in fitnesses]))
        gen_best = population[gen_best_idx]
        gen_best_fit = fitnesses[gen_best_idx]

        if (best_fit is None) or (gen_best_fit.fitness > best_fit.fitness):
            best_overall = ControllerParams(gen_best.amplitudes.copy(),
                                            gen_best.phases.copy(),
                                            gen_best.frequency)
            best_fit = gen_best_fit

        print(f"[Gen {gen+1:02d}] Best fitness: {gen_best_fit.fitness:.4f} | "
              f"X+ {gen_best_fit.x_progress:.3f}  | "
              f"energy {gen_best_fit.energy:.3f}  | speed {gen_best_fit.speed:.3f}  | f={gen_best.frequency:.2f}Hz")

        # Elitism
        elite_idxs = np.argsort([-f.fitness for f in fitnesses])[:ELITES]
        elites = [population[i] for i in elite_idxs]

        # Create next population
        next_pop = elites.copy()
        while len(next_pop) < POP_SIZE:
            # Parents via tournament
            p1 = tournament_select(rng, population, fitnesses)
            p2 = tournament_select(rng, population, fitnesses)
            child = blend_crossover(rng, p1, p2)
            # Adaptive-ish mutation: shrink a bit as gens go
            sigma = MUTATION_SIGMA * (0.9 ** gen)
            child = mutate(rng, child, sigma=sigma)
            next_pop.append(child)
        population = next_pop

    print("\n=== Evolution complete ===")
    print(f"Best fitness: {best_fit.fitness:.4f} | X+ {best_fit.x_progress:.3f} | f={best_overall.frequency:.2f}Hz")
    return best_overall, best_fit


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


def watch_controller(params: ControllerParams, duration=SIM_DURATION):
    """
    Launches viewer to visualize the learned controller and records XY history.
    """
    global HISTORY
    HISTORY = []

    mujoco.set_mjcb_control(None)  # reset callbacks (as recommended)
    world, gecko_core, model, data, to_track = build_world_and_model()

    # Wrap the control to also log the core position
    base_ctrl = make_control_fn(params)
    def ctrl_and_log(m, d):
        base_ctrl(m, d)
        HISTORY.append(to_track[0].xpos.copy())

    mujoco.set_mjcb_control(ctrl_and_log)

    # Run viewer; you can close window any time
    viewer.launch(model=model, data=data)

    # After you close the viewer, plot the path
    if HISTORY:
        show_qpos_history(HISTORY)


# =========================
# Main
# =========================
def main():
    # 1) Evolve controller headlessly
    _, _, model_for_nu, _, _ = build_world_and_model()
    best_params, best_fit = evolve(model_for_nu=model_for_nu)

    # 2) Watch best controller in viewer
    print("\nLaunching viewer with the best controller...")
    watch_controller(best_params, duration=SIM_DURATION)

    # Optional: record a video (uncomment if you want)
    # PATH_TO_VIDEO_FOLDER = "./__videos__"
    # video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)
    # world, gecko_core, model, data, to_track = build_world_and_model()
    # mujoco.set_mjcb_control(make_control_fn(best_params))
    # video_renderer(model, data, duration=10, video_recorder=video_recorder)


if __name__ == "__main__":
    main()
