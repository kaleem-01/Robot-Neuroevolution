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
from ariel.utils.video_recorder import VideoRecorder

# =========================
# Config & Paths
# =========================
SEED = 42
RNG = np.random.default_rng(SEED)

# Evolution hyperparams (outer/body EA)
POP_SIZE       = 10
ELITES         = max(1, POP_SIZE // 10)
GENERATIONS    = 3
TOURNAMENT_K   = 10

# Mutation scales
MUT_BODY_SIGMA = 0.10     # Gaussian noise on NDE input vectors (clipped to [0,1])
MUT_CTRL_SIGMA = 0.15     # Gaussian noise on controller genes

# Simulation
SIM_DURATION   = 10.0     # seconds

# Template values
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

# Inner-controller budget (per body)
CTRL_INNER_POP   = 12
CTRL_INNER_GENS  = 6
CTRL_INNER_K     = 6
CTRL_INNER_ELITE = max(1, CTRL_INNER_POP // 8)

# Final controller refinement budget for the best body
CTRL_FINAL_POP   = 24
CTRL_FINAL_GENS  = 20
CTRL_FINAL_K     = 8
CTRL_FINAL_ELITE = max(1, CTRL_FINAL_POP // 8)

# Folders
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

OUT_DIR = DATA / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# =========================
# Stabilization helpers (optional tuning)
# =========================
def harden_model(model: mj.MjModel) -> mj.MjModel:
    model.opt.timestep   = min(getattr(model.opt, "timestep", 0.005), 0.002)
    model.opt.integrator = mj.mjtIntegrator.mjINT_RK4
    model.opt.iterations    = max(getattr(model.opt, "iterations", 50), 50)
    model.opt.ls_iterations = max(getattr(model.opt, "ls_iterations", 20), 20)
    model.opt.tolerance     = min(getattr(model.opt, "tolerance", 1e-6), 1e-8)

    if model.dof_damping is not None and model.dof_damping.size > 0:
        model.dof_damping[:] = np.maximum(model.dof_damping, 0.5)
    if model.dof_armature is not None and model.dof_armature.size > 0:
        model.dof_armature[:] = np.maximum(model.dof_armature, 0.01)

    if model.nu > 0:
        if model.actuator_ctrlrange is not None and model.actuator_ctrlrange.size == 2 * model.nu:
            lo = model.actuator_ctrlrange[:, 0]
            hi = model.actuator_ctrlrange[:, 1]
            needs = (hi - lo) < 1e-9
            lo[needs] = CTRL_MIN
            hi[needs] = CTRL_MAX
            model.actuator_ctrlrange[:, 0] = lo
            model.actuator_ctrlrange[:, 1] = hi

    if model.geom_friction is not None and model.geom_friction.size >= 3:
        fr = model.geom_friction.reshape(-1, 3)
        fr[:, 0] = np.clip(fr[:, 0], 0.4, None)
    return model

def get_core_geom_id(model: mj.MjModel) -> int:
    try:
        return mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "core")
    except Exception:
        pass
    for gid in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
        if name and "core" in name:
            return gid
    return 0

# =========================
# Fitness
# =========================
def fitness_function(history_xyz: List[np.ndarray]) -> float:
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
# Build body → MuJoCo model
# =========================
def build_robot_from_body(body: BodyGenome):
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_mats = nde.forward([body.type_p, body.conn_p, body.rot_p])
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    graph = hpd.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
    core = construct_mjspec_from_graph(graph)
    return core, graph

def body_fingerprint(graph_json_path: str) -> int:
    with open(graph_json_path, "rb") as f:
        h = hashlib.blake2b(f.read(), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False)

# =========================
# Rollouts
# =========================
@dataclass
class FitnessResult:
    fitness: float
    distance: float
    reached: bool
    steps: int

def rollout(ind_body: BodyGenome, ind_ctrl: ControllerGenome,
            duration: float=SIM_DURATION, record_path: Optional[str]=None
            ) -> Tuple[FitnessResult, List[np.ndarray]]:
    """Headless rollout for a fixed body+controller. Returns fitness and XYZ history."""
    mj.set_mjcb_control(None)

    # World + robot
    world = OlympicArena()
    core, graph = build_robot_from_body(ind_body)
    world.spawn(core.spec, spawn_position=SPAWN_POS)

    model = world.spec.compile()
    harden_model(model)
    data  = mj.MjData(model)

    if model.nu == 0:
        return FitnessResult(fitness=-1e6, distance=1e6, reached=False, steps=0), []

    # Stable hash of body for deterministic per-joint params
    tmp_json = str(OUT_DIR / "_tmp_body.json")
    save_graph_as_json(graph, tmp_json)
    b_hash = body_fingerprint(tmp_json)

    # Expand controller to per-joint parameters
    amps, phases, freq = expand_per_joint_params(model, ind_ctrl, b_hash)

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

    core_gid = get_core_geom_id(model)
    mj.mj_resetData(model, data)
    dt      = model.opt.timestep
    n_steps = int(np.ceil(duration / dt))

    history_xyz: List[np.ndarray] = []
    history_xyz.append(data.geom_xpos[core_gid].copy())

    steps = 0
    for _ in range(n_steps):
        mj.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            dist_bad = 1e6
            return FitnessResult(fitness=-dist_bad, distance=dist_bad, reached=False, steps=steps), history_xyz
        steps += 1
        history_xyz.append(data.geom_xpos[core_gid].copy())

    # Fitness per template: negative distance to TARGET_POSITION
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history_xyz[-1]
    dist = float(np.sqrt((xt-xc)**2 + (yt-yc)**2 + (zt-zc)**2))
    fit  = -dist
    reached = dist < 0.25

    # Optional video
    if record_path is not None:
        video_recorder = VideoRecorder(output_folder=str(OUT_DIR) + "/BestVideo.mp4")
        video_renderer(model, data, duration=min(5.0, duration), video_recorder=video_recorder)

    return FitnessResult(fitness=fit, distance=dist, reached=reached, steps=steps), history_xyz

# =========================
# Hierarchical EA
# =========================
def _build_model_for_body(body: BodyGenome) -> Tuple[mj.MjModel, mj.MjData, int, "nx.DiGraph", int]:
    """Compile a model once for a given body (faster for inner controller EA)."""
    world = OlympicArena()
    core, graph = build_robot_from_body(body)
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    harden_model(model)
    data  = mj.MjData(model)

    tmp_json = str(OUT_DIR / "_tmp_body_for_model.json")
    save_graph_as_json(graph, tmp_json)
    b_hash = body_fingerprint(tmp_json)

    core_gid = get_core_geom_id(model)
    return model, data, core_gid, graph, b_hash

def _rollout_fixed_model(model: mj.MjModel, data: mj.MjData, core_gid: int,
                         ctrl: ControllerGenome, body_hash: int,
                         duration: float = SIM_DURATION) -> Tuple[FitnessResult, List[np.ndarray]]:
    """Faster rollout if reusing a compiled model for a given body."""
    mj.set_mjcb_control(None)
    if model.nu == 0:
        return FitnessResult(fitness=-1e6, distance=1e6, reached=False, steps=0), []

    amps, phases, freq = expand_per_joint_params(model, ctrl, body_hash)
    prev_ctrl = {"val": None}
    def ctrl_fn(m: mj.MjModel, d: mj.MjData):
        t = d.time
        target = amps * (np.pi/2) * np.sin(2*np.pi*freq * t + phases)
        if prev_ctrl["val"] is None:
            new_ctrl = SMOOTH_ALPHA * target
        else:
            new_ctrl = (1.0 - SMOOTH_ALPHA) * prev_ctrl["val"] + SMOOTH_ALPHA * target
        d.ctrl[:] = np.clip(new_ctrl, CTRL_MIN, CTRL_MAX)
        prev_ctrl["val"] = d.ctrl.copy()
    mj.set_mjcb_control(ctrl_fn)

    mj.mj_resetData(model, data)
    dt      = model.opt.timestep
    n_steps = int(np.ceil(duration / dt))

    history_xyz: List[np.ndarray] = [data.geom_xpos[core_gid].copy()]
    steps = 0
    for _ in range(n_steps):
        mj.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            dist_bad = 1e6
            return FitnessResult(fitness=-dist_bad, distance=dist_bad, reached=False, steps=steps), history_xyz
        steps += 1
        history_xyz.append(data.geom_xpos[core_gid].copy())

    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history_xyz[-1]
    dist = float(np.sqrt((xt-xc)**2 + (yt-yc)**2 + (zt-zc)**2))
    return FitnessResult(fitness=-dist, distance=dist, reached=(dist < 0.25), steps=steps), history_xyz

def _ctrl_tournament_select(rng, pop_ctrl: List[ControllerGenome], fits: List[float], k: int) -> ControllerGenome:
    idxs = rng.choice(len(pop_ctrl), size=min(k, len(pop_ctrl)), replace=False)
    best_idx = max(idxs, key=lambda i: fits[i])
    return pop_ctrl[best_idx]

def _ctrl_crossover(rng, a: ControllerGenome, b: ControllerGenome) -> ControllerGenome:
    return blend_ctrl_genome(rng, a, b)

def _ctrl_mutate(rng, g: ControllerGenome, gen_idx: int, base_sigma=MUT_CTRL_SIGMA) -> ControllerGenome:
    sigma = base_sigma * (0.98 ** gen_idx)
    return mutate_ctrl_genome(rng, g, sigma=sigma)

def evolve_controller_for_body(body: BodyGenome,
                               rng_seed: int,
                               pop_size: int = CTRL_INNER_POP,
                               gens: int = CTRL_INNER_GENS,
                               tourn_k: int = CTRL_INNER_K,
                               elites: int = CTRL_INNER_ELITE,
                               reuse_compiled: bool = True) -> Tuple[ControllerGenome, float]:
    """
    Inner EA that optimizes controllers for a fixed body.
    Returns (best_controller, best_fitness).
    """
    rng = np.random.default_rng(rng_seed)
    pop_ctrl = [sample_ctrl_genome(rng) for _ in range(pop_size)]

    if reuse_compiled:
        model, data, core_gid, graph, b_hash = _build_model_for_body(body)
        use_fast = True
    else:
        model = data = core_gid = b_hash = None
        use_fast = False

    best_g: Optional[ControllerGenome] = None
    best_fit: float = -np.inf

    for gidx in range(gens):
        fits: List[float] = []
        if use_fast:
            for g in pop_ctrl:
                fr, _ = _rollout_fixed_model(model, data, core_gid, g, b_hash, duration=SIM_DURATION)
                fits.append(fr.fitness)
        else:
            for g in pop_ctrl:
                fr, _ = rollout(body, g, duration=SIM_DURATION)
                fits.append(fr.fitness)

        gen_best_i = int(np.argmax(fits))
        if fits[gen_best_i] > best_fit:
            best_fit = float(fits[gen_best_i])
            best_g   = pop_ctrl[gen_best_i]

        elite_idxs = np.argsort([-f for f in fits])[:elites]
        elites_pop = [pop_ctrl[i] for i in elite_idxs]

        next_pop: List[ControllerGenome] = elites_pop.copy()
        while len(next_pop) < pop_size:
            p1 = _ctrl_tournament_select(rng, pop_ctrl, fits, k=tourn_k)
            p2 = _ctrl_tournament_select(rng, pop_ctrl, fits, k=tourn_k)
            child = _ctrl_crossover(rng, p1, p2)
            child = _ctrl_mutate(rng, child, gen_idx=gidx)
            next_pop.append(child)
        pop_ctrl = next_pop

    assert best_g is not None
    return best_g, best_fit

# Memoization so the same body isn't re-solved repeatedly within a generation
_body_hash_cache: dict[int, Tuple[ControllerGenome, float]] = {}

def _hash_body_genome(body: BodyGenome) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(body.type_p.tobytes())
    h.update(body.conn_p.tobytes())
    h.update(body.rot_p.tobytes())
    return int.from_bytes(h.digest(), 'little', signed=False)

def evaluate_body_population(pop_bodies: List[BodyGenome],
                             ctrl_budget_gens: int = CTRL_INNER_GENS,
                             ctrl_budget_pop: int  = CTRL_INNER_POP) -> Tuple[List[float], List[ControllerGenome]]:
    """
    For each body in the population, run a small inner controller EA and
    return its best fitness and the corresponding controller.
    """
    results_fits: List[float] = []
    results_ctrls: List[ControllerGenome] = []

    for i, body in enumerate(pop_bodies):
        bh = _hash_body_genome(body)
        if bh in _body_hash_cache:
            best_ctrl, best_fit = _body_hash_cache[bh]
        else:
            seed = int((SEED + 10007*i) & 0x7FFFFFFF)
            best_ctrl, best_fit = evolve_controller_for_body(
                body, seed,
                pop_size=ctrl_budget_pop,
                gens=ctrl_budget_gens,
                tourn_k=CTRL_INNER_K,
                elites=CTRL_INNER_ELITE,
                reuse_compiled=True,
            )
            _body_hash_cache[bh] = (best_ctrl, best_fit)

        results_fits.append(best_fit)
        results_ctrls.append(best_ctrl)
    return results_fits, results_ctrls

def _body_tournament_select(rng, pop_bodies: List[BodyGenome], fits: List[float], k: int) -> BodyGenome:
    idxs = rng.choice(len(pop_bodies), size=min(k, len(pop_bodies)), replace=False)
    best_idx = max(idxs, key=lambda i: fits[i])
    b = pop_bodies[best_idx]
    return BodyGenome(b.type_p.copy(), b.conn_p.copy(), b.rot_p.copy())

def _body_crossover(rng, a: BodyGenome, b: BodyGenome) -> BodyGenome:
    return blend_body_genome(rng, a, b)

def _body_mutate(rng, g: BodyGenome, gen_idx: int, base_sigma=MUT_BODY_SIGMA) -> BodyGenome:
    sigma = base_sigma * (0.97 ** gen_idx)
    return mutate_body_genome(rng, g, sigma=sigma)

def evolve_bodies_then_controllers(
        outer_pop_size: int = POP_SIZE,
        outer_gens: int = GENERATIONS,
        outer_k: int = TOURNAMENT_K,
        outer_elites: int = ELITES,
        inner_pop: int = CTRL_INNER_POP,
        inner_gens: int = CTRL_INNER_GENS
    ) -> Tuple[BodyGenome, ControllerGenome, FitnessResult, List[float], List[float]]:
    """
    Outer EA over bodies. Each body's fitness = best controller fitness
    obtained by an inner controller EA with a small budget.
    Returns best body, its refined best controller and final FitnessResult, plus curves.
    """
    rng = np.random.default_rng(SEED)
    pop_bodies: List[BodyGenome] = [sample_body_genome(rng) for _ in range(outer_pop_size)]

    best_body: Optional[BodyGenome] = None
    best_ctrl_for_best_body: Optional[ControllerGenome] = None
    best_fit_value: float = -np.inf

    gen_best_curve: List[float] = []
    overall_best_curve: List[float] = []

    for gen in range(outer_gens):
        body_fits, body_ctrls = evaluate_body_population(pop_bodies, ctrl_budget_gens=inner_gens, ctrl_budget_pop=inner_pop)

        gen_best_idx = int(np.argmax(body_fits))
        gen_best_fit_val = float(body_fits[gen_best_idx])
        gen_best_body = pop_bodies[gen_best_idx]
        gen_best_ctrl = body_ctrls[gen_best_idx]

        if gen_best_fit_val > best_fit_value:
            best_fit_value = gen_best_fit_val
            best_body = BodyGenome(gen_best_body.type_p.copy(),
                                   gen_best_body.conn_p.copy(),
                                   gen_best_body.rot_p.copy())
            best_ctrl_for_best_body = ControllerGenome(
                gen_best_ctrl.frequency, gen_best_ctrl.amp_mean, gen_best_ctrl.amp_std,
                gen_best_ctrl.phase_mean, gen_best_ctrl.phase_std, gen_best_ctrl.seed
            )

        print(f"[Outer Gen {gen+1:02d}] best_f={gen_best_fit_val:.3f}")

        gen_best_curve.append(gen_best_fit_val)
        overall_best_curve.append(best_fit_value)

        elite_idxs = np.argsort([-f for f in body_fits])[:outer_elites]
        elites_bodies = [pop_bodies[i] for i in elite_idxs]

        next_bodies: List[BodyGenome] = [BodyGenome(b.type_p.copy(), b.conn_p.copy(), b.rot_p.copy())
                                         for b in elites_bodies]
        while len(next_bodies) < outer_pop_size:
            p1 = _body_tournament_select(rng, pop_bodies, body_fits, k=outer_k)
            p2 = _body_tournament_select(rng, pop_bodies, body_fits, k=outer_k)
            child = _body_crossover(rng, p1, p2)
            child = _body_mutate(rng, child, gen_idx=gen)
            next_bodies.append(child)

        pop_bodies = next_bodies

    assert best_body is not None and best_ctrl_for_best_body is not None

    # Final, larger controller optimization for the best body
    print("\n[Final Controller Refinement] Starting larger inner EA on the best body...")
    final_ctrl, final_fit_val = evolve_controller_for_body(
        best_body, rng_seed=SEED + 99991,
        pop_size=CTRL_FINAL_POP, gens=CTRL_FINAL_GENS,
        tourn_k=CTRL_FINAL_K, elites=CTRL_FINAL_ELITE, reuse_compiled=True
    )

    final_fit, _ = rollout(best_body, final_ctrl, duration=SIM_DURATION)

    print(f"=== Hierarchical evolution complete ===")
    print(f"Best hierarchical fitness: {final_fit.fitness:.3f} | dist={final_fit.distance:.3f} | reached={final_fit.reached}")

    return best_body, final_ctrl, final_fit, gen_best_curve, overall_best_curve

# =========================
# Save / Viz
# =========================
def save_best_hier(best_body: BodyGenome, best_ctrl: ControllerGenome,
                   final_fit: FitnessResult,
                   gen_curve: List[float], overall_curve: List[float],
                   tag: str = "hierarchical"):
    core, graph = build_robot_from_body(best_body)
    json_path = OUT_DIR / f"best_body_{tag}.json"
    save_graph_as_json(graph, str(json_path))

    with open(OUT_DIR / f"best_ctrl_{tag}.pkl", "wb") as f:
        pickle.dump(best_ctrl, f)
    with open(OUT_DIR / f"best_fit_{tag}.pkl", "wb") as f:
        pickle.dump(final_fit, f)
    with open(OUT_DIR / f"gen_best_curve_{tag}.pkl", "wb") as f:
        pickle.dump(gen_curve, f)
    with open(OUT_DIR / f"overall_best_curve_{tag}.pkl", "wb") as f:
        pickle.dump(overall_curve, f)

    console.log(f"[hier] Saved best body JSON → {json_path}")
    console.log(f"[hier] Saved controller + curves in {OUT_DIR}")

def plot_curves(gen_best_curve: List[float], overall_best_curve: List[float], title="Hierarchical EA (−distance to target)"):
    plt.figure(figsize=(8,5))
    plt.plot(gen_best_curve, label="Gen best (outer)")
    plt.plot(overall_best_curve, "--", label="Overall best (outer)")
    plt.xlabel("Outer Generation"); plt.ylabel("Fitness (higher is better)")
    plt.grid(True); plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fitness_curves_hierarchical.png", dpi=200)
    plt.show()

# =========================
# Robust loader (if you later want to view saved best)
# =========================
def load_graph_from_json_robust(json_path: Path) -> "nx.DiGraph":
    with open(json_path, "r", encoding="utf-8") as f:
        body_json = json.load(f)
    try:
        hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        graph = hpd.json_to_graph(body_json)  # if provided by your HPD
        return graph
    except Exception:
        return json_graph.node_link_graph(body_json)

# =========================
# Main
# =========================
def main(mode: Literal["hierarchical", "viewer_saved"] = "hierarchical",
         save_video: bool = False) -> None:
    if mode == "hierarchical":
        best_body, best_ctrl, final_fit, gen_curve, overall_curve = evolve_bodies_then_controllers()

        # Launch viewer on the best hierarchical solution
        mj.set_mjcb_control(None)
        world = OlympicArena()
        core, graph = build_robot_from_body(best_body)
        world.spawn(core.spec, spawn_position=SPAWN_POS)
        model = world.spec.compile()
        harden_model(model)
        data  = mj.MjData(model)

        tmp_json = str(OUT_DIR / "_tmp_best_hier_view.json")
        save_graph_as_json(graph, tmp_json)
        b_hash = body_fingerprint(tmp_json)
        amps, phases, freq = expand_per_joint_params(model, best_ctrl, b_hash)

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

        save_best_hier(best_body, best_ctrl, final_fit, gen_curve, overall_curve, tag="hierarchical")
        plot_curves(gen_curve, overall_curve, title="Hierarchical EA (outer body EA, inner controller EA)")

        if save_video:
            _ = rollout(best_body, best_ctrl, duration=min(5.0, SIM_DURATION), record_path="best_hierarchical")
            console.log("Saved video to outputs/.")

    elif mode == "viewer_saved":
        json_path = OUT_DIR / "best_body_hierarchical.json"
        ctrl_path = OUT_DIR / "best_ctrl_hierarchical.pkl"
        if not json_path.exists() or not ctrl_path.exists():
            raise FileNotFoundError("Saved hierarchical artifacts not found.")
        graph = load_graph_from_json_robust(json_path)
        core = construct_mjspec_from_graph(graph)
        with open(ctrl_path, "rb") as f:
            ctrl = pickle.load(f)
        if isinstance(ctrl, dict):
            ctrl = ControllerGenome(**ctrl)

        mj.set_mjcb_control(None)
        world = OlympicArena()
        world.spawn(core.spec, spawn_position=SPAWN_POS)
        model = world.spec.compile()
        harden_model(model)
        data  = mj.MjData(model)

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

if __name__ == "__main__":
    # Run the hierarchical pipeline by default
    main(mode="hierarchical", save_video=False)
