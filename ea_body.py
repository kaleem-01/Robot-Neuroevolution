"""Assignment 3 – Coevolution of Body (NDE) and Controller (CPG) – Clean, Sequential
Fixed to use template SPAWN_POS and fitness (negative distance to TARGET_POSITION),
and with robust viewer/load_best paths.
"""

# --- Standard library
import os
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, List

# --- Third-party
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco import viewer

# For fallback JSON->graph
import networkx as nx
from networkx.readwrite import json_graph

# --- Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

# =========================
# Config & Paths
# =========================
SEED = 42
RNG = np.random.default_rng(SEED)

# Evolution hyperparams
POP_SIZE       = 10
ELITES         = max(1, POP_SIZE // 10)
GENERATIONS    = 10
TOURNAMENT_K   = 10
MUT_BODY_SIGMA = 0.20     # Gaussian noise on NDE input vectors (clipped to [0,1])
MUT_CTRL_SIGMA = 0.15     # Gaussian noise on controller genes

# Simulation
SIM_DURATION   =10.0     # seconds

# >>> From the template <<<
SPAWN_POS       = [-0.8, 0.0, 0.10]
NUM_OF_MODULES  = 30
TARGET_POSITION = [5.0, 0.0, 0.5]

# Controller shaping
SMOOTH_ALPHA   = 0.2
CTRL_MIN       = -np.pi/2
CTRL_MAX       =  np.pi/2
FREQ_MIN       = 0.4
FREQ_MAX       = 2.5

# Body (NDE) genome
GENOTYPE_SIZE  = 64       # length of each NDE input vector

# Folders
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

OUT_DIR = DATA / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# UI types
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]



def harden_model(model: mj.MjModel) -> None:
    # Smaller timestep + more accurate integrator
    model.opt.timestep = min(model.opt.timestep, 0.002)          # e.g. 500 Hz
    model.opt.integrator = mj.mjtIntegrator.mjINT_RK4

    # Stronger solver settings (more iterations, smaller tolerance)
    model.opt.iterations    = max(model.opt.iterations, 50)
    model.opt.ls_iterations = max(model.opt.ls_iterations, 20)
    model.opt.tolerance     = min(model.opt.tolerance, 1e-8)

    # Damp the joints and add a bit of armature (inertia) to fight spikes
    if model.dof_damping is not None and model.dof_damping.size > 0:
        model.dof_damping[:] = np.maximum(model.dof_damping, 0.5)
    if model.dof_armature is not None and model.dof_armature.size > 0:
        model.dof_armature[:] = np.maximum(model.dof_armature, 0.01)

    # Ensure actuators have sensible ctrlrange (avoid unbounded torques)
    if model.nu > 0:
        if model.actuator_ctrlrange is not None and model.actuator_ctrlrange.size == 2 * model.nu:
            lo = model.actuator_ctrlrange[:, 0]
            hi = model.actuator_ctrlrange[:, 1]
            # If not set (zeros), give them a range consistent with your clamp
            needs = (hi - lo) < 1e-9
            lo[needs] = -np.pi/2
            hi[needs] =  np.pi/2
            model.actuator_ctrlrange[:, 0] = lo
            model.actuator_ctrlrange[:, 1] = hi

    # (Optional) slightly reduce sliding friction to limit contact chatter
    if model.geom_friction is not None and model.geom_friction.size >= 3:
        fr = model.geom_friction.reshape(-1, 3)
        fr[:, 0] = np.clip(fr[:, 0], 0.4, None)  # sliding μ

    return model

# =========================
# Helpers: geom lookup & plotting
# =========================
def get_core_geom_id(model: mj.MjModel) -> int:
    """Return the geom id for 'core' (try exact name, then contains 'core')."""
    try:
        return mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "core")
    except Exception:
        pass
    for gid in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
        if name and "core" in name:
            return gid
    return 0  # fallback


def show_xpos_history_over_background(history_xyz: list[np.ndarray]) -> None:
    """Render top-down background and overlay XY path (optional visualization)."""
    # Top-down camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    # model = harden_model(model)
    data = mj.MjData(model)

    save_path = str(DATA / "background.png")
    single_frame_renderer(model, data, camera=camera, save_path=save_path, save=True)

    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)

    pos = np.array(history_xyz)
    ax.plot(pos[:, 0], pos[:, 1], "-", label="Path")
    ax.plot(pos[0, 0], pos[0, 1], "go", label="Start")
    ax.plot(pos[-1, 0], pos[-1, 1], "ro", label="End")

    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(); ax.set_title("Robot XY path (world coords)")
    plt.tight_layout()
    plt.show()


# =========================
# Template fitness (negative distance to target)
# =========================
def fitness_function(history_xyz: list[np.ndarray]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history_xyz[-1]
    cartesian_distance = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
    return -float(cartesian_distance)


# =========================
# Controller genome (compact) -> per-joint params
# =========================
@dataclass
class ControllerGenome:
    frequency: float         # [FREQ_MIN, FREQ_MAX]
    amp_mean: float          # [0, 1]
    amp_std: float           # [0, 1]
    phase_mean: float        # [-pi, pi]
    phase_std: float         # [0, pi]
    seed: int                # base seed to expand per-joint params

def clip_ctrl_genome(g: ControllerGenome) -> ControllerGenome:
    g.frequency  = float(np.clip(g.frequency, FREQ_MIN, FREQ_MAX))
    g.amp_mean   = float(np.clip(g.amp_mean, 0.0, 1.0))
    g.amp_std    = float(np.clip(g.amp_std,  0.0, 1.0))
    g.phase_mean = float(((g.phase_mean + np.pi) % (2*np.pi)) - np.pi)
    g.phase_std  = float(np.clip(g.phase_std, 0.0, np.pi))
    return g

def sample_ctrl_genome(rng: np.random.Generator) -> ControllerGenome:
    return ControllerGenome(
        frequency  = float(rng.uniform(FREQ_MIN, FREQ_MAX)),
        amp_mean   = float(rng.uniform(0.2, 0.8)),
        amp_std    = float(rng.uniform(0.05, 0.35)),
        phase_mean = float(rng.uniform(-np.pi, np.pi)),
        phase_std  = float(rng.uniform(0.1, 0.8*np.pi)),
        seed       = int(rng.integers(0, np.iinfo(np.int32).max))
    )

def mutate_ctrl_genome(rng: np.random.Generator, g: ControllerGenome, sigma: float=MUT_CTRL_SIGMA) -> ControllerGenome:
    g2 = ControllerGenome(
        frequency  = g.frequency  + rng.normal(0.0, sigma),
        amp_mean   = g.amp_mean   + rng.normal(0.0, sigma),
        amp_std    = g.amp_std    + rng.normal(0.0, sigma),
        phase_mean = g.phase_mean + rng.normal(0.0, sigma*np.pi),
        phase_std  = g.phase_std  + rng.normal(0.0, sigma),
        seed       = g.seed if rng.random() > 0.1 else int(rng.integers(0, np.iinfo(np.int32).max))
    )
    return clip_ctrl_genome(g2)

def blend_ctrl_genome(rng: np.random.Generator, a: ControllerGenome, b: ControllerGenome) -> ControllerGenome:
    def blend(x, y, scale=0.5): return (1.0 - scale) * x + scale * y
    g = ControllerGenome(
        frequency  = blend(a.frequency,  b.frequency),
        amp_mean   = blend(a.amp_mean,   b.amp_mean),
        amp_std    = blend(a.amp_std,    b.amp_std),
        phase_mean = blend(a.phase_mean, b.phase_mean),
        phase_std  = blend(a.phase_std,  b.phase_std),
        seed       = a.seed if rng.random() < 0.5 else b.seed
    )
    return clip_ctrl_genome(g)

def expand_per_joint_params(model: mj.MjModel, ctrl_g: ControllerGenome, body_hash: int
                            ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Map compact controller genome to joint-wise amplitudes/phases for this body (any model.nu)."""
    nu = model.nu
    # Stable seed tied to both the controller seed and the *body* hash
    seed_mix = (ctrl_g.seed ^ body_hash) & 0x7FFFFFFF
    rng = np.random.default_rng(seed_mix)
    amps   = rng.normal(loc=ctrl_g.amp_mean,   scale=abs(ctrl_g.amp_std),  size=nu)
    phases = rng.normal(loc=ctrl_g.phase_mean, scale=abs(ctrl_g.phase_std), size=nu)
    amps   = np.clip(amps, 0.0, 1.0)
    phases = ((phases + np.pi) % (2*np.pi)) - np.pi
    freq   = float(np.clip(ctrl_g.frequency, FREQ_MIN, FREQ_MAX))
    return amps, phases, freq


# =========================
# Body genome (NDE inputs)
# =========================
@dataclass
class BodyGenome:
    type_p: np.ndarray   # float32 in [0,1], shape (GENOTYPE_SIZE,)
    conn_p: np.ndarray   # float32 in [0,1]
    rot_p:  np.ndarray   # float32 in [0,1]

def sample_body_genome(rng: np.random.Generator) -> BodyGenome:
    return BodyGenome(
        type_p = rng.random(GENOTYPE_SIZE, dtype=np.float32),
        conn_p = rng.random(GENOTYPE_SIZE, dtype=np.float32),
        rot_p  = rng.random(GENOTYPE_SIZE, dtype=np.float32),
    )

def mutate_body_genome(rng: np.random.Generator, g: BodyGenome, sigma: float=MUT_BODY_SIGMA) -> BodyGenome:
    def mut(x):
        y = x.astype(np.float32) + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
        return np.clip(y, 0.0, 1.0).astype(np.float32)
    return BodyGenome(mut(g.type_p), mut(g.conn_p), mut(g.rot_p))

def blend_body_genome(rng: np.random.Generator, a: BodyGenome, b: BodyGenome) -> BodyGenome:
    alpha = 0.5
    def blend(x, y): return np.clip((1-alpha)*x + alpha*y, 0.0, 1.0).astype(np.float32)
    return BodyGenome(blend(a.type_p, b.type_p),
                      blend(a.conn_p, b.conn_p),
                      blend(a.rot_p,  b.rot_p))


# =========================
# Individual (body + controller)
# =========================
@dataclass
class Individual:
    body: BodyGenome
    ctrl: ControllerGenome

def sample_individual(rng: np.random.Generator) -> Individual:
    return Individual(sample_body_genome(rng), sample_ctrl_genome(rng))

def crossover(rng: np.random.Generator, p1: Individual, p2: Individual) -> Individual:
    return Individual(
        body = blend_body_genome(rng, p1.body, p2.body),
        ctrl = blend_ctrl_genome(rng, p1.ctrl, p2.ctrl)
    )

def mutate_individual(rng: np.random.Generator, ind: Individual, gen_idx: int) -> Individual:
    body_sigma = MUT_BODY_SIGMA * (0.95 ** gen_idx)  # light annealing
    ctrl_sigma = MUT_CTRL_SIGMA  * (0.95 ** gen_idx)
    return Individual(
        body = mutate_body_genome(rng, ind.body, sigma=body_sigma),
        ctrl = mutate_ctrl_genome(rng, ind.ctrl, sigma=ctrl_sigma)
    )


# =========================
# Body build & rollout
# =========================
def body_fingerprint(graph_json_path: str) -> int:
    with open(graph_json_path, "rb") as f:
        h = hashlib.blake2b(f.read(), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False)

def build_robot_from_body(body: BodyGenome):
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_mats = nde.forward([body.type_p, body.conn_p, body.rot_p])
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    graph = hpd.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
    core = construct_mjspec_from_graph(graph)
    return core, graph

@dataclass
class FitnessResult:
    fitness: float
    distance: float
    reached: bool
    steps: int

def rollout(ind: Individual, duration: float=SIM_DURATION, record_path: Optional[str]=None
            ) -> Tuple[FitnessResult, list[np.ndarray]]:
    """Headless rollout. Returns fitness result and (optional) XYZ history."""
    mj.set_mjcb_control(None)

    # World + robot
    world = OlympicArena()
    core, graph = build_robot_from_body(ind.body)
    world.spawn(core.spec, spawn_position=SPAWN_POS)

    # Model & data
    model = world.spec.compile()
    data  = mj.MjData(model)

    if model.nu == 0:
        return FitnessResult(fitness=-1e6, distance=1e6, reached=False, steps=0), []

    # Stable hash of body for deterministic per-joint params
    tmp_json = str(OUT_DIR / "_tmp_body.json")
    save_graph_as_json(graph, tmp_json)
    body_hash = body_fingerprint(tmp_json)

    # Expand controller to per-joint parameters
    amps, phases, freq = expand_per_joint_params(model, ind.ctrl, body_hash)

    # Control callback (smooth tracking of sine targets)
    prev_ctrl = {"val": None}
    def ctrl_fn(m: mj.MjModel, d: mj.MjData):
        t = d.time
        target = amps * (np.pi/2) * np.sin(2*np.pi*freq* t + phases)
        if prev_ctrl["val"] is None:
            new_ctrl = SMOOTH_ALPHA * target
        else:
            new_ctrl = (1.0 - SMOOTH_ALPHA) * prev_ctrl["val"] + SMOOTH_ALPHA * target
        d.ctrl[:] = np.clip(new_ctrl, CTRL_MIN, CTRL_MAX)
        prev_ctrl["val"] = d.ctrl.copy()

    mj.set_mjcb_control(ctrl_fn)

    # Prep to run
    core_gid = get_core_geom_id(model)
    mj.mj_resetData(model, data)
    dt      = model.opt.timestep
    n_steps = int(np.ceil(duration / dt))

    # History for plotting
    history_xyz: list[np.ndarray] = []
    history_xyz.append(data.geom_xpos[core_gid].copy())

    # Step
    steps = 0
    for _ in range(n_steps):
        mj.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            # exploding sim → bad fitness
            dist_bad = 1e6
            return FitnessResult(fitness=-dist_bad, distance=dist_bad, reached=False, steps=steps), history_xyz
        steps += 1
        history_xyz.append(data.geom_xpos[core_gid].copy())

    # Fitness per template: negative distance to TARGET_POSITION
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history_xyz[-1]
    dist = float(np.sqrt((xt-xc)**2 + (yt-yc)**2 + (zt-zc)**2))
    fit  = -dist
    reached = dist < 0.25  # heuristically "close enough"

    # Optional video
    if record_path is not None:
        video_recorder = VideoRecorder(output_folder=str(OUT_DIR))
        video_renderer(model, data, duration=min(5.0, duration), video_recorder=video_recorder)

    return FitnessResult(fitness=fit, distance=dist, reached=reached, steps=steps), history_xyz


# =========================
# EA (sequential)
# =========================
def tournament_select(rng: np.random.Generator, pop: List[Individual], fits: List[FitnessResult], k=TOURNAMENT_K
                      ) -> Individual:
    idxs = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    best_idx = max(idxs, key=lambda i: fits[i].fitness)
    return pop[best_idx]

def evaluate_population(pop: List[Individual]) -> List[FitnessResult]:
    results: List[FitnessResult] = []
    for ind in pop:
        fit, _ = rollout(ind, duration=SIM_DURATION)
        results.append(fit)
    return results

def evolve() -> Tuple[Individual, FitnessResult, List[float], List[float]]:
    rng = np.random.default_rng(SEED)
    population = [sample_individual(rng) for _ in range(POP_SIZE)]

    best_ind: Optional[Individual] = None
    best_fit: Optional[FitnessResult] = None
    gen_best_curve: List[float] = []
    overall_best_curve: List[float] = []

    for gen in range(GENERATIONS):
        fitnesses = evaluate_population(population)

        gen_best_idx = int(np.argmax([f.fitness for f in fitnesses]))
        gen_best = population[gen_best_idx]
        gen_best_fit = fitnesses[gen_best_idx]

        # Update global best
        if (best_fit is None) or (gen_best_fit.fitness > best_fit.fitness):
            best_ind = Individual(
                body=BodyGenome(gen_best.body.type_p.copy(),
                                gen_best.body.conn_p.copy(),
                                gen_best.body.rot_p.copy()),
                ctrl=ControllerGenome(gen_best.ctrl.frequency,
                                      gen_best.ctrl.amp_mean,
                                      gen_best.ctrl.amp_std,
                                      gen_best.ctrl.phase_mean,
                                      gen_best.ctrl.phase_std,
                                      gen_best.ctrl.seed)
            )
            best_fit = gen_best_fit

        print(f"[Gen {gen+1:02d}] best_f={gen_best_fit.fitness:.3f} "
              f"| dist={gen_best_fit.distance:.3f} | reached={gen_best_fit.reached} | steps={gen_best_fit.steps}")

        gen_best_curve.append(gen_best_fit.fitness)
        overall_best_curve.append(best_fit.fitness if best_fit else gen_best_fit.fitness)

        # Elitism
        elite_idxs = np.argsort([-f.fitness for f in fitnesses])[:ELITES]
        elites = [population[i] for i in elite_idxs]

        # Next generation
        next_pop: List[Individual] = elites.copy()
        while len(next_pop) < POP_SIZE:
            p1 = tournament_select(rng, population, fitnesses, k=TOURNAMENT_K)
            p2 = tournament_select(rng, population, fitnesses, k=TOURNAMENT_K)
            child = crossover(rng, p1, p2)
            child = mutate_individual(rng, child, gen_idx=gen)
            next_pop.append(child)
        population = next_pop

    assert best_ind is not None and best_fit is not None
    print("\n=== Evolution complete ===")
    print(f"Best fitness: {best_fit.fitness:.3f} | dist={best_fit.distance:.3f} | reached={best_fit.reached}")
    return best_ind, best_fit, gen_best_curve, overall_best_curve


# =========================
# Save / Viz
# =========================
def save_best(best: Individual, best_fit: FitnessResult, gen_curve: List[float], overall_curve: List[float], tag: str="coevo_simple"):
    core, graph = build_robot_from_body(best.body)
    json_path = OUT_DIR / f"best_body_{tag}.json"
    save_graph_as_json(graph, str(json_path))

    with open(OUT_DIR / f"best_ctrl_{tag}.pkl", "wb") as f:
        pickle.dump(best.ctrl, f)
    with open(OUT_DIR / f"best_fit_{tag}.pkl", "wb") as f:
        pickle.dump(best_fit, f)
    with open(OUT_DIR / f"gen_best_curve_{tag}.pkl", "wb") as f:
        pickle.dump(gen_curve, f)
    with open(OUT_DIR / f"overall_best_curve_{tag}.pkl", "wb") as f:
        pickle.dump(overall_curve, f)

    console.log(f"Saved best body JSON → {json_path}")
    console.log(f"Saved controller + curves in {OUT_DIR}")

def plot_curves(gen_best_curve: List[float], overall_best_curve: List[float], title="Coevolution Fitness (−distance to target)"):
    plt.figure(figsize=(8,5))
    plt.plot(gen_best_curve, label="Gen best")
    plt.plot(overall_best_curve, "--", label="Overall best")
    plt.xlabel("Generation"); plt.ylabel("Fitness (higher is better)")
    plt.grid(True); plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fitness_curves.png", dpi=200)
    plt.show()


# =========================
# Loading helpers
# =========================
def load_graph_from_json_robust(json_path: Path) -> "nx.DiGraph":
    """First try decoder's loader (if available), else fall back to networkx node-link."""
    with open(json_path, "r", encoding="utf-8") as f:
        body_json = json.load(f)
    try:
        # If your HighProbabilityDecoder exposes a json_to_graph, use it:
        hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        graph = hpd.json_to_graph(body_json)
        return graph
    except Exception:
        # Fallback: assume node-link format
        return json_graph.node_link_graph(body_json)


# =========================
# Main
# =========================
def main(mode: Literal["evolve", "viewer", "load_best"] = "evolve", save_video: bool = False) -> None:
    if mode == "evolve":
        best_ind, best_fit, gen_best_curve, overall_best_curve = evolve()

        # Launch viewer on the best individual
        mj.set_mjcb_control(None)
        core, graph = build_robot_from_body(best_ind.body)
        world = OlympicArena()
        world.spawn(core.spec, spawn_position=SPAWN_POS)
        model = world.spec.compile()
        data = mj.MjData(model)

        # Stable expansion based on this body
        tmp_json = str(OUT_DIR / "_tmp_best_view.json")
        save_graph_as_json(graph, tmp_json)
        b_hash = body_fingerprint(tmp_json)
        amps, phases, freq = expand_per_joint_params(model, best_ind.ctrl, b_hash)

        prev_ctrl = {"val": None}
        def ctrl_fn(m, d):
            t = d.time
            target = amps * (np.pi/2) * np.sin(2*np.pi*freq * t + phases)
            if prev_ctrl["val"] is None:
                new_ctrl = SMOOTH_ALPHA * target
            else:
                new_ctrl = (1.0 - SMOOTH_ALPHA) * prev_ctrl["val"] + SMOOTH_ALPHA * target
            d.ctrl[:] = np.clip(new_ctrl, CTRL_MIN, CTRL_MAX)
            prev_ctrl["val"] = d.ctrl.copy()

        mj.set_mjcb_control(ctrl_fn)
        viewer.launch(model=model, data=data)

        # Persist and visualize
        save_best(best_ind, best_fit, gen_best_curve, overall_best_curve, tag="coevo_simple")
        plot_curves(gen_best_curve, overall_best_curve, title="Coevolution fitness (−distance to target)")

        # Optional: a short video of the best
        if save_video:
            _ = rollout(best_ind, duration=min(5.0, SIM_DURATION), record_path="best_coevo_simple")
            console.log("Saved video to outputs/.")

        # Optional XY path figure
        fit, history = rollout(best_ind, duration=SIM_DURATION, record_path=None)
        show_xpos_history_over_background(history)

    elif mode == "load_best":
        # Paths saved by save_best(..., tag="coevo_simple")
        json_path = OUT_DIR / "best_body_coevo_simple.json"
        ctrl_path = OUT_DIR / "best_ctrl_coevo_simple.pkl"

        if not json_path.exists():
            raise FileNotFoundError(f"Body JSON not found: {json_path}")
        if not ctrl_path.exists():
            raise FileNotFoundError(f"Controller PKL not found: {ctrl_path}")

        # --- Load the saved body graph JSON robustly
        graph = load_graph_from_json_robust(json_path)
        core = construct_mjspec_from_graph(graph)

        # --- Load controller genome
        with open(ctrl_path, "rb") as f:
            ctrl = pickle.load(f)
        if isinstance(ctrl, dict):
            ctrl = ControllerGenome(**ctrl)

        # --- Launch viewer
        mj.set_mjcb_control(None)
        world = OlympicArena()
        world.spawn(core.spec, spawn_position=SPAWN_POS)
        model = world.spec.compile()
        data  = mj.MjData(model)

        # Expand controller params for this body
        b_hash = body_fingerprint(str(json_path))
        amps, phases, freq = expand_per_joint_params(model, ctrl, b_hash)

        prev_ctrl = {"val": None}
        def ctrl_fn(m, d):
            t = d.time
            target = amps * (np.pi/2) * np.sin(2*np.pi*freq * t + phases)
            if prev_ctrl["val"] is None:
                new_ctrl = SMOOTH_ALPHA * target
            else:
                new_ctrl = (1.0 - SMOOTH_ALPHA) * prev_ctrl["val"] + SMOOTH_ALPHA * target
            d.ctrl[:] = np.clip(new_ctrl, CTRL_MIN, CTRL_MAX)
            prev_ctrl["val"] = d.ctrl.copy()

        mj.set_mjcb_control(ctrl_fn)
        viewer.launch(model=model, data=data)

    else:  # "viewer" -> random individual quick demo
        ind = sample_individual(RNG)
        mj.set_mjcb_control(None)

        world = OlympicArena()
        core, graph = build_robot_from_body(ind.body)
        world.spawn(core.spec, spawn_position=SPAWN_POS)
        model = world.spec.compile()
        data  = mj.MjData(model)

        tmp_json = str(OUT_DIR / "_tmp_view.json")
        save_graph_as_json(graph, tmp_json)
        b_hash = body_fingerprint(tmp_json)
        amps, phases, freq = expand_per_joint_params(model, ind.ctrl, b_hash)

        prev_ctrl = {"val": None}
        def ctrl_fn(m, d):
            t = d.time
            target = amps * (np.pi/2) * np.sin(2*np.pi*freq * t + phases)
            if prev_ctrl["val"] is None:
                new_ctrl = SMOOTH_ALPHA * target
            else:
                new_ctrl = (1.0 - SMOOTH_ALPHA) * prev_ctrl["val"] + SMOOTH_ALPHA * target
            d.ctrl[:] = np.clip(new_ctrl, CTRL_MIN, CTRL_MAX)
            prev_ctrl["val"] = d.ctrl.copy()

        mj.set_mjcb_control(ctrl_fn)
        viewer.launch(model=model, data=data)


if __name__ == "__main__":
    # Examples:
    # main(mode="evolve", save_video=True)
    # main(mode="viewer")
    # main(mode="load_best")
    main(mode="evolve", save_video=True)
