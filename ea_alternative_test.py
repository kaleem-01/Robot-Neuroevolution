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
import multiprocessing

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.simulation.environments.rugged_heightmap import RuggedTerrainWorld
from ariel.simulation.environments.simple_tilted_world import TiltedFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


# =========================
# Config
# =========================
SEED = 42
SIM_DURATION = 10.0          # seconds per evaluation
SMOOTH_ALPHA = 0.2          # 0..1, higher tracks target faster (smoother with lower)
POP_SIZE = 80
ELITES = POP_SIZE // 10      # number of elites to carry over each generation (for GA only)
GENERATIONS = 40
MUTATION_SIGMA = 0.25       # base mutation scale
TOURNAMENT_K = 5            # tournament size for parent selection (for GA only)

# IMPORTANT: make these positive if you want *penalties*
ENERGY_PENALTY = 0.00       # penalty * mean(|ctrl|)
SPEED_PENALTY  = -0.05       # penalty * mean(|dctrl|)

# Frequency bounds (shared across joints)
FREQ_MIN = 0.5
FREQ_MAX = 2.0

# Joint limits (hinges)
CTRL_MIN = -np.pi/2
CTRL_MAX =  np.pi/2

# Global model variables for multiprocessing
model = None
data = None
to_track = None


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
# Evolutionary Algorithms
# =========================
def tournament_select(rng, pop, fits, k=TOURNAMENT_K):
    idxs = rng.choice(len(pop), size=k, replace=False)
    best = max(idxs, key=lambda i: fits[i].fitness)
    return pop[best]


def evolve_ga(terrain="flat"):
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

def flatten_params(params: ControllerParams) -> np.ndarray:
    return np.concatenate([params.amplitudes, params.phases, [params.frequency]])

def unflatten_params(x: np.ndarray) -> ControllerParams:
    n = len(x) - 1
    half = n // 2
    amplitudes = x[:half]
    phases = x[half:n]
    frequency = x[-1]
    return ControllerParams(amplitudes, phases, frequency)

def evolve_es(terrain="flat"):
    """
    CMA strategy to compare
    """
    rng = np.random.default_rng(SEED)

    sample = sample_params(rng, model.nu)
    mean = flatten_params(sample)
    sigma = MUTATION_SIGMA
    dim = len(mean)
    C = np.eye(dim)

    lam = POP_SIZE
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mueff = 1 / np.sum(weights**2)

    cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
    c1 = 2 / ((dim+1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff-2+1/mueff) / ((dim+2)**2 + mueff))

    cs = (mueff + 2) / (dim + mueff + 5)
    ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    ENN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

    ps = np.zeros(dim)
    pc = np.zeros(dim)

    best_overall, best_fit = None, None
    gen_best_curve, overall_best_curve = [], []

    for gen in range(GENERATIONS):
        A = np.linalg.cholesky(C)
        arz = rng.normal(size=(lam, dim))
        arx = mean + sigma * (arz @ A.T)
        population = [unflatten_params(x) for x in arx]
        # eval
        with multiprocessing.Pool() as pool:
            fitnesses = pool.starmap(
                rollout_and_score,
                [(ind, SIM_DURATION, terrain) for ind in population]
            )

        fit_vals = np.array([f.fitness for f in fitnesses])
        idx_sorted = np.argsort(-fit_vals)

        xold = mean.copy()
        mean = np.sum(weights[:, None] * arx[idx_sorted[:mu]], axis=0)

        gen_best_idx = idx_sorted[0]
        gen_best = population[gen_best_idx]
        gen_best_fit = fitnesses[gen_best_idx]

        # update overall best
        if (best_fit is None) or (gen_best_fit.fitness > best_fit.fitness):
            best_overall = ControllerParams(gen_best.amplitudes.copy(),
                                            gen_best.phases.copy(),
                                            gen_best.frequency)
            best_fit = gen_best_fit

        print(f"[CMA-ES Gen {gen+1:02d}] Best fitness: {gen_best_fit.fitness:.4f} | "
              f"X+ {gen_best_fit.x_progress:.3f}  | "
              f"energy {gen_best_fit.energy:.3f}  | speed {gen_best_fit.speed:.3f}  | f={gen_best.frequency:.2f}Hz")

        gen_best_curve.append(gen_best_fit.fitness)
        overall_best_curve.append(best_fit.fitness)

        # update paths 
        y = (mean - xold) / sigma
        z = np.sum(weights[:, None] * arz[idx_sorted[:mu]], axis=0)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (A @ z)
        hsig = int((np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen+1)))) / ENN < (1.4 + 2/(dim+1)))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y

        #covariance matrix
        C = (1 - c1 - cmu) * C \
            + c1 * np.outer(pc, pc) \
            + cmu * np.sum(
                weights[:, None, None] *
                np.array([np.outer(arx[i]-xold, arx[i]-xold) for i in idx_sorted[:mu]]),
                axis=0
            ) / (sigma**2)

        sigma *= np.exp((cs/ds) * (np.linalg.norm(ps)/ENN - 1))

    print("\n=== CMA-ES Evolution complete ===")
    assert best_fit is not None and best_overall is not None
    print(f"Best fitness: {best_fit.fitness:.4f} | X+ {best_fit.x_progress:.3f} | "
          f"energy {best_fit.energy:.3f} | speed {best_fit.speed:.3f} | f={best_overall.frequency:.2f}Hz")

    return best_overall, best_fit, gen_best_curve, overall_best_curve



def run_random_search(terrain="flat"):
    """
    Runs a Random Search baseline.
    """
    nu = model.nu
    rng = np.random.default_rng(SEED)
    best_overall_fit = -np.inf
    overall_best_curve = []

    for gen in range(GENERATIONS):
        population = [sample_params(rng, nu) for _ in range(POP_SIZE)]
        with multiprocessing.Pool() as pool:
            fitnesses = pool.starmap(
            rollout_and_score,
            [(ind, SIM_DURATION, terrain) for ind in population]
            )
        
        gen_best_fit = max(f.fitness for f in fitnesses)
        if gen_best_fit > best_overall_fit:
            best_overall_fit = gen_best_fit

        print(f"[Rnd Gen {gen+1:02d}] Best fitness: {gen_best_fit:.4f}")
        overall_best_curve.append(best_overall_fit)

    return overall_best_curve


# =========================
# Main execution
# =========================
def run_experiment(terrain, evolution_fn):
    """Wrapper to run a single evolution experiment."""
    global model, data, to_track
    world, gecko_core, model, data, to_track = build_world_and_model(terrain=terrain)
    
    if evolution_fn.__name__ == "run_random_search":
        curve = evolution_fn(terrain=terrain)
        return None, None, curve
    else:
        best_params, best_fit, gen_best_curve, overall_best_curve = evolution_fn(terrain=terrain)
        return best_params, best_fit, overall_best_curve


def plot_pairwise_comparison(results, terrain, alg1, alg2, filename):
    """Plots comparison between two algorithms."""
    plt.figure(figsize=(10, 6))
    
    colors = {'GA': 'blue', 'ES': 'red', 'Random': 'green'}
    
    curves1 = results[alg1]
    curves2 = results[alg2]
    
    for alg, curves, color in [(alg1, curves1, colors[alg1]), (alg2, curves2, colors[alg2])]:
        max_len = max(len(curve) for curve in curves)
        padded_curves = []
        for curve in curves:
            last_val = curve[-1]
            padded_curve = np.pad(curve, (0, max_len - len(curve)), 'constant', constant_values=last_val)
            padded_curves.append(padded_curve)
        
        curves_np = np.array(padded_curves)
        mean_curve = np.mean(curves_np, axis=0)
        std_curve = np.std(curves_np, axis=0)
        
        plt.plot(mean_curve, label=f"Mean Best Fitness ({alg})", color=color, linewidth=2)
        plt.fill_between(np.arange(max_len), mean_curve - std_curve, mean_curve + std_curve, 
                        color=color, alpha=0.2)

    plt.title(f"{alg1} vs {alg2} on {terrain.capitalize()} Terrain ({len(curves1)} runs)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Y-axis progress)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    NUM_RUNS = 5
    terrain = "tilted"

    print(f"Starting comparative analysis on '{terrain}' terrain for {NUM_RUNS} runs.")
    print(f"  Population size: {POP_SIZE}")
    print(f"  Generations: {GENERATIONS}")
    
    results = {
        "ES": [],
        "GA": [],
        "Random": []
    }

    algorithms = {
        "ES": evolve_es,
        "GA": evolve_ga,
        "Random": run_random_search
    }

    for i in range(NUM_RUNS):
        print(f"\n--- Starting Run {i+1}/{NUM_RUNS} ---")
        for name, func in algorithms.items():
            print(f"Running {name}...")
            _, _, curve = run_experiment(terrain=terrain, evolution_fn=func)
            results[name].append(curve)

    print("\n=== All experiments complete ===")
    
    # 3 plots like the prof wanted
    print("Creating comparison plots...")
    plot_pairwise_comparison(results, terrain, "Random", "ES", f"baseline_vs_es_{terrain}.png")
    plot_pairwise_comparison(results, terrain, "Random", "GA", f"baseline_vs_ga_{terrain}.png")
    plot_pairwise_comparison(results, terrain, "ES", "GA", f"es_vs_ga_{terrain}.png")
    
    print("All plots saved successfully!")