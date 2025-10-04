"""Assignment 3 template code — with non-learner filter, dynamic duration,
multi-spawn training, per-generation save/resume, CPG option, and warning capture.
"""

# === Standard library
from __future__ import annotations
import os
import sys
import io
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Tuple, List, Optional
from dataclasses import dataclass
import contextlib
import time

# === Third-party
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco import viewer

# === Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

if TYPE_CHECKING:
    from networkx import DiGraph

# ---------- Config toggles ----------
USE_CPG: bool = True            # switch to CPG for speedier learning
RUN_ONE_GENERATION: bool = True # run a single generation per execution (resume later)
SAVE_GRAPH_AND_BEST: bool = True
MAX_PROCESSES: int = 15          # cap multiprocessing

# ---------- Randomness, paths ----------
SEED = 42
RNG = np.random.default_rng(SEED)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

WARN_LOG = DATA / "mujoco_warnings.txt"

# ---------- World / task ----------
NUM_OF_MODULES = 30
SPAWN_START  = [-0.8, 0.0, 0.1]
SPAWN_MID    = [ 2.0, 0.0, 0.1]   # around rugged start
SPAWN_LATE   = [ 4.0, 0.0, 0.1]   # near the end section
SPAWN_POS_LIST = [SPAWN_START, SPAWN_MID, SPAWN_LATE]

TARGET_POSITION = [5.0, 0.0, 0.5]

# ---------- Evolution hyperparams ----------
POP_SIZE       = 60
ELITES         = max(1, POP_SIZE // 10)
GENERATIONS    = 50                # total desired (if not chunked)
TOURNAMENT_K   = 8
MUT_BODY_SIGMA = 0.40
GENOTYPE_SIZE  = 64

# Base duration; dynamic schedule will override
BASE_SIM_DURATION = 15

ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# =========================
# Body genome (NDE inputs)
# =========================
@dataclass
class BodyGenome:
    type_p: np.ndarray   # float32 in [0,1], shape (GENOTYPE_SIZE,)
    conn_p: np.ndarray   # float32 in [0,1]
    rot_p:  np.ndarray   # float32 in [0,1]
    fitness: float = -np.inf
    progress_x: float = 0.0

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

def _sbx_pair(rng, x, y, eta: float = 15.0):
    u = rng.random(x.shape)
    beta = np.where(u <= 0.5, (2*u)**(1/(eta+1)), (1/(2*(1-u)))**(1/(eta+1)))
    c1 = 0.5*((1+beta)*x + (1-beta)*y)
    c2 = 0.5*((1-beta)*x + (1+beta)*y)
    choose = rng.random(x.shape) < 0.5
    child = np.where(choose, c1, c2)
    return np.clip(child, 0.0, 1.0)

def crossover_sbx_linked(rng: np.random.Generator, a: BodyGenome, b: BodyGenome, eta: float = 15.0) -> BodyGenome:
    return BodyGenome(
        type_p = _sbx_pair(rng, a.type_p, b.type_p, eta).astype(np.float32),
        conn_p = _sbx_pair(rng, a.conn_p, b.conn_p, eta).astype(np.float32),
        rot_p  = _sbx_pair(rng, a.rot_p,  b.rot_p,  eta).astype(np.float32),
    )

# =========================
# Build, control, fitness
# =========================
def build_robot_from_body(body: BodyGenome):
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_mats = nde.forward([body.type_p, body.conn_p, body.rot_p])
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    graph = hpd.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
    core = construct_mjspec_from_graph(graph)
    return core, graph

def _prepare_matrix_from_genome(vec: np.ndarray, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    m, n = shape
    need = m * n
    base = np.asarray(vec, dtype=np.float64).ravel()
    if base.size == 0:
        base = np.array([0.0], dtype=np.float64)
    if base.size < need:
        reps = (need + base.size - 1) // base.size
        base = np.tile(base, reps)[:need]
    else:
        base = base[:need]
    mat = base.reshape(m, n)
    mat = 0.0138 + 0.5 * (mat - 0.5) + 0.02 * rng.normal(size=mat.shape)
    return mat

def nn_controller(model: mj.MjModel, data: mj.MjData, body: Optional[BodyGenome]) -> np.ndarray:
    input_size  = len(data.qpos)
    hidden_size = 8
    output_size = model.nu
    if output_size == 0:
        return np.zeros(0, dtype=np.float64)

    seed = 12345
    if body is not None:
        seed = int(1e6 * float(np.mean(body.type_p) + np.mean(body.conn_p) + np.mean(body.rot_p))) & 0x7FFFFFFF
    rng = np.random.default_rng(seed ^ input_size ^ (hidden_size << 8) ^ (output_size << 16))

    if body is not None:
        w1 = _prepare_matrix_from_genome(body.type_p, (input_size,  hidden_size), rng)
        w2 = _prepare_matrix_from_genome(body.conn_p, (hidden_size, hidden_size), rng)
        w3 = _prepare_matrix_from_genome(body.rot_p,  (hidden_size, output_size), rng)
    else:
        w1 = 0.0138 + rng.normal(scale=0.5, size=(input_size, hidden_size))
        w2 = 0.0138 + rng.normal(scale=0.5, size=(hidden_size, hidden_size))
        w3 = 0.0138 + rng.normal(scale=0.5, size=(hidden_size, output_size))

    x = np.asarray(data.qpos, dtype=np.float64)
    h1 = np.tanh(x @ w1)
    h2 = np.tanh(h1 @ w2)
    y  = np.tanh(h2 @ w3)  # [-1,1]

    if model.actuator_ctrlrange is not None and model.actuator_ctrlrange.size == 2 * output_size:
        lo = model.actuator_ctrlrange[:, 0]
        hi = model.actuator_ctrlrange[:, 1]
        wid = hi - lo
        wid[wid < 1e-9] = np.pi
        target = lo + 0.5 * (y + 1.0) * wid
    else:
        target = (np.pi / 2.0) * y
    return np.clip(target, -np.pi/2, np.pi/2)

# ---- Simple CPG (coupled oscillators) ----
class CPGState:
    def __init__(self, nu: int):
        self.phase = np.zeros(nu, dtype=np.float64)
        # different natural frequencies to break symmetry
        self.omega = 2*np.pi * (0.4 + 0.2*np.linspace(0,1,nu))
        # small coupling matrix (nearest-neighbour ring)
        self.K = 0.6
        self.alpha = 0.05   # amplitude smoothing
        self.amp = np.ones(nu, dtype=np.float64) * 0.6

def cpg_controller_factory(model: mj.MjModel) -> callable:
    nu = model.nu
    state = CPGState(nu)
    # choose actuator phases alternating to encourage gait
    for i in range(nu):
        state.phase[i] = (i % 2) * np.pi
    last_time = {"t": 0.0}

    def cpg_cb(m: mj.MjModel, d: mj.MjData, body: Optional[BodyGenome] = None):
        # dt = time per step
        t = d.time
        dt = max(1e-4, t - last_time["t"])
        last_time["t"] = t

        # Kuramoto-like update on phases
        phase = state.phase
        for i in range(nu):
            coupling = 0.0
            if nu > 1:
                coupling += np.sin(phase[(i-1) % nu] - phase[i])
                coupling += np.sin(phase[(i+1) % nu] - phase[i])
            dphi = state.omega[i] + state.K * coupling
            phase[i] = (phase[i] + dphi * dt) % (2*np.pi)

        # slowly adapt amplitudes based on joint limits (if available)
        amp = state.amp
        if m.actuator_ctrlrange is not None and m.actuator_ctrlrange.size == 2*nu:
            lo = m.actuator_ctrlrange[:,0]; hi = m.actuator_ctrlrange[:,1]
            span = hi - lo
            span[span<1e-6] = 1.0
            target_amp = 0.45 * span
            amp += state.alpha * (target_amp - amp)

        u = amp * np.sin(phase)
        # hard clamp to ctrlrange or ±π/2
        if m.actuator_ctrlrange is not None and m.actuator_ctrlrange.size == 2*nu:
            lo = m.actuator_ctrlrange[:,0]; hi = m.actuator_ctrlrange[:,1]
            d.ctrl[:nu] = np.clip(u, lo, hi)
        else:
            d.ctrl[:nu] = np.clip(u, -np.pi/2, np.pi/2)

    return cpg_cb

# =========================
# Simulation helpers
# =========================
def fitness_function(history: List[List[float]]) -> float:
    # history: list of [x,y,z]
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    return -float(np.sqrt((xt-xc)**2 + (yt-yc)**2 + (zt-zc)**2))

def last_x(history: List[List[float]]) -> float:
    return float(history[-1][0])

def experiment(
    robot: Any,
    controller: Controller,
    duration: float = 15.0,
    mode: ViewerTypes = "simple",
    spawn_pos: List[float] = SPAWN_START,
) -> Tuple[List[List[float]], float]:
    """Run the simulation; return (xpos_history, sim_time)."""
    mj.set_mjcb_control(None)

    world = OlympicArena()
    world.spawn(robot.spec, spawn_position=spawn_pos)
    model = world.spec.compile()
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)

    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # choose callback wrapper (Controller wraps the user function)
    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))

    t0 = time.time()
    if mode == "simple":
        simple_runner(model, data, duration=duration)
    elif mode == "launcher":
        viewer.launch(model=model, data=data)
    elif mode == "video":
        path_to_video_folder = str(DATA / "videos")
        video_recorder = VideoRecorder(output_folder=path_to_video_folder)
        video_renderer(model, data, duration=duration, video_recorder=video_recorder)
    elif mode == "frame":
        save_path = str(DATA / "robot.png")
        single_frame_renderer(model, data, save=True, save_path=save_path)
    elif mode == "no_control":
        mj.set_mjcb_control(None)
        viewer.launch(model=model, data=data)

    sim_time = time.time() - t0
    return controller.tracker.history["xpos"][0], sim_time

# ---------- Non-learner filter ----------
def quick_motion_screen(body: BodyGenome, spawn: List[float]) -> bool:
    """Short random/CPG rollout to kill non-learners early."""
    try:
        robot, _ = build_robot_from_body(body)
    except Exception:
        return False

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    if USE_CPG:
        # CPG screen (very cheap)
        # build model to make the cpg callback (needs model.nu)
        world = OlympicArena()
        world.spawn(robot.spec, spawn_position=spawn)
        model = world.spec.compile()
        _ = mj.MjData(model)
        cpg_cb = cpg_controller_factory(model)
        ctrl = Controller(controller_callback_function=lambda m,d: cpg_cb(m,d,body=body), tracker=tracker)
    else:
        ctrl = Controller(controller_callback_function=lambda m,d: nn_controller(m,d,body), tracker=tracker)

    try:
        hist, _ = experiment(robot=robot, controller=ctrl, duration=1.2, mode="simple", spawn_pos=spawn)
        dx = last_x(hist) - spawn[0]
        # Heuristic thresholds: must move a bit AND have some variance
        return (dx > 0.05) and (np.std(np.diff(np.array(hist)[:,0])) > 1e-4)
    except Exception:
        return False

# ---------- Dynamic duration schedule ----------
def schedule_duration(best_progress_x: float) -> float:
    """
    Increase allowed duration only when checkpoints are reached.
    Checkpoints roughly: reach x>=0.0 (entering rugged), pass rugged x>=3.0, finish x>=5.0
    """
    if best_progress_x < 0.0:
        return 15.0
    elif best_progress_x < 3.0:
        return 45.0
    else:
        return 100.0

# ---------- Multi-spawn curriculum ----------
def choose_spawn(gen: int, rng: np.random.Generator) -> List[float]:
    # Early gens: mostly start; later: mix in mid/late
    if gen < 5:
        return SPAWN_START
    elif gen < 15:
        return SPAWN_POS_LIST[rng.integers(0, 2)]  # start or mid
    else:
        return SPAWN_POS_LIST[rng.integers(0, 3)]  # any of 3

# ---------- Rollout & score (with curriculum & dynamic duration) ----------
def rollout_and_score(body: BodyGenome, duration: float, spawn: List[float]) -> Tuple[float,float]:
    """Return (fitness, last_x)."""
    robot, _ = build_robot_from_body(body)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    if USE_CPG:
        # need a model to size the CPG
        world = OlympicArena()
        world.spawn(robot.spec, spawn_position=spawn)
        model = world.spec.compile()
        _ = mj.MjData(model)
        cpg_cb = cpg_controller_factory(model)
        ctrl = Controller(controller_callback_function=lambda m,d: cpg_cb(m,d,body=body), tracker=tracker)
    else:
        ctrl = Controller(controller_callback_function=lambda m,d: nn_controller(m,d,body), tracker=tracker)

    hist, _sim_t = experiment(robot=robot, controller=ctrl, duration=duration, mode="simple", spawn_pos=spawn)
    fit = fitness_function(hist)
    px  = last_x(hist)
    return fit, px

# ---------- Tournament selection ----------
def tournament_select(rng: np.random.Generator, pop: List[BodyGenome], k=TOURNAMENT_K) -> BodyGenome:
    idxs = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    best_idx = max(idxs, key=lambda i: pop[i].fitness)
    return pop[best_idx]

# ---------- Persistence ----------
STATE_PATH = DATA / "evo_state.pkl"
CURVES_PATH = DATA / "curves.pkl"
BEST_JSON_PATH = DATA / "best_body.json"
BEST_CTRL_PATH = DATA / "best_ctrl.pkl"  # placeholder if you later evolve controllers

def save_state(gen: int, population: List[BodyGenome], best: BodyGenome, gen_curve: List[float], overall_curve: List[float], best_progress: float):
    # store minimal info to resume
    state = {
        "gen": gen,
        "population": [(ind.type_p, ind.conn_p, ind.rot_p, ind.fitness, ind.progress_x) for ind in population],
        "best": (best.type_p, best.conn_p, best.rot_p, best.fitness, best.progress_x),
        "best_progress_x": best_progress,
    }
    with open(STATE_PATH, "wb") as f:
        pickle.dump(state, f)
    with open(CURVES_PATH, "wb") as f:
        pickle.dump({"gen_best": gen_curve, "overall_best": overall_curve}, f)
    if SAVE_GRAPH_AND_BEST:
        try:
            core, graph = build_robot_from_body(best)
            save_graph_as_json(graph, BEST_JSON_PATH)
        except Exception:
            pass

def load_state() -> Tuple[int, List[BodyGenome], BodyGenome, List[float], List[float], float]:
    if not STATE_PATH.exists():
        raise FileNotFoundError
    with open(STATE_PATH, "rb") as f:
        s = pickle.load(f)
    pop = []
    for (tp, cp, rp, fit, px) in s["population"]:
        ind = BodyGenome(tp.astype(np.float32), cp.astype(np.float32), rp.astype(np.float32), float(fit), float(px))
        pop.append(ind)
    bt = s["best"]
    best = BodyGenome(bt[0].astype(np.float32), bt[1].astype(np.float32), bt[2].astype(np.float32), float(bt[3]), float(bt[4]))
    if CURVES_PATH.exists():
        with open(CURVES_PATH, "rb") as f:
            curves = pickle.load(f)
        gen_curve = curves.get("gen_best", [])
        overall_curve = curves.get("overall_best", [])
    else:
        gen_curve, overall_curve = [], []
    best_progress_x = float(s.get("best_progress_x", 0.0))
    return int(s["gen"]), pop, best, gen_curve, overall_curve, best_progress_x

# ---------- Evaluation (multiprocessing) ----------
def _worker_eval(args):
    ind, duration, spawn = args
    try:
        fit, px = rollout_and_score(ind, duration=duration, spawn=spawn)
        return (fit, px)
    except Exception:
        return (-1e9, -1e9)

def evaluate_population(pop: List[BodyGenome], duration: float, gen: int) -> List[Tuple[float,float]]:
    # choose spawn per evaluation to encourage generalisation
    spawn_choices = [choose_spawn(gen, RNG) for _ in pop]
    # with mj.disable_warnings():  # suppress native warnings where available (failsafe wrapper)
    from multiprocessing import Pool
    with Pool(processes=min(MAX_PROCESSES, len(pop))) as pool:
        results = pool.map(_worker_eval, [(ind, duration, spawn_choices[i]) for i, ind in enumerate(pop)])
    return results

# ---------- Initial population with non-learner filter ----------
def init_population(rng: np.random.Generator, size: int) -> List[BodyGenome]:
    pop = []
    trials = 0
    while len(pop) < size and trials < size * 20:
        g = sample_body_genome(rng)
        spawn = SPAWN_START
        if quick_motion_screen(g, spawn):
            pop.append(g)
        trials += 1
    # Fallback: if too strict, fill remaining slots without screen
    while len(pop) < size:
        pop.append(sample_body_genome(rng))
    return pop

# ---------- Evolve ----------
def evolve(total_generations: int = GENERATIONS, run_one_generation: bool = RUN_ONE_GENERATION):
    rng = np.random.default_rng(SEED)

    # Resume or init
    try:
        start_gen, population, best_body, gen_best_curve, overall_best_curve, best_progress_x = load_state()
        console.log(f"[Resume] Loaded generation {start_gen}, best fitness {best_body.fitness:.4f}, best progress x={best_progress_x:.2f}")
        gen0 = start_gen
    except FileNotFoundError:
        population = init_population(rng, POP_SIZE)
        best_body = population[0]
        best_body.fitness = -np.inf
        gen_best_curve = []
        overall_best_curve = []
        best_progress_x = 0.0
        gen0 = 0

    max_generations_to_run = 1 if run_one_generation else (total_generations - gen0)
    for gen in range(gen0, min(total_generations, gen0 + max_generations_to_run)):
        # dynamic duration based on best progress so far
        duration = schedule_duration(best_progress_x)
        console.log(f"Generation {gen+1}/{total_generations} — duration={duration:.1f}s")

        # Evaluate
        results = evaluate_population(population, duration=duration, gen=gen)
        for ind, (fit, px) in zip(population, results):
            ind.fitness = fit
            ind.progress_x = px

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        gen_best = population[0].fitness
        gen_best_curve.append(gen_best)

        if gen_best > best_body.fitness:
            best_body = population[0]
            best_progress_x = max(best_progress_x, best_body.progress_x)

        overall_best_curve.append(best_body.fitness)
        console.log(f"  Best gen fit: {gen_best:.4f} | Overall best: {best_body.fitness:.4f} | Best x: {best_progress_x:.2f}")

        # Save state each generation
        save_state(gen+1, population, best_body, gen_best_curve, overall_best_curve, best_progress_x)

        # Reproduce (elitism + SBX + mutation)
        new_population: List[BodyGenome] = population[:ELITES]
        # elites copied (no mutation) — but keep as-is to stabilize
        while len(new_population) < POP_SIZE:
            p1 = tournament_select(rng, population)
            p2 = tournament_select(rng, population)
            child = crossover_sbx_linked(rng, p1, p2)
            sigma = float(MUT_BODY_SIGMA * (0.98 ** (gen - gen0)))
            child = mutate_body_genome(rng, child, sigma=sigma)
            # soft screen during reproduction (to avoid filling with duds)
            if quick_motion_screen(child, choose_spawn(gen, rng)):
                new_population.append(child)

        population = new_population

    return best_body, gen_best_curve, overall_best_curve

# ---------- Plotting / viz ----------
def show_xpos_history(history: List[List[float]]) -> None:
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(model, data, camera=camera, save_path=save_path, save=True)

    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    pos_data = np.array(history)
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_START[0]
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    plt.title("Robot Path in XY Plane")
    plt.show()

# ---------- MuJoCo warning silencer ----------
# DOESN'T FUCKING WORK FOR SOME GODDAMN REASON
class _FilterStderr(io.TextIOBase):
    def __init__(self, real_stderr, log_path: Path):
        self._real = real_stderr
        self._log = open(log_path, "a", buffering=1, encoding="utf-8")
        self._buf = ""

    def write(self, s):
        # Buffer until newline to filter linewise
        self._buf += s
        out = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if "WARNING: " in line:
                # divert to file only
                self._log.write(line + "\n")
            else:
                out.append(line + "\n")
        if out:
            return self._real.write("".join(out))
        return 0

    def flush(self):
        self._real.flush()
        self._log.flush()

    def close(self):
        try:
            self._log.close()
        except Exception:
            pass
        return super().close()

@contextlib.contextmanager
def silence_mujoco_warnings():
    filt = _FilterStderr(sys.stderr, WARN_LOG)
    old = sys.stderr
    sys.stderr = filt
    try:
        yield
    finally:
        sys.stderr = old
        try:
            filt.close()
        except Exception:
            pass

# ---------- Main ----------
def main() -> None:
    with silence_mujoco_warnings():
        best_body, gen_best_curve, overall_best_curve = evolve()

        # Build best body & save artifacts
        core, graph = build_robot_from_body(best_body)
        if SAVE_GRAPH_AND_BEST:
            save_graph_as_json(graph, DATA / "robot_graph.json")

        # Quick demo rollout from full start, using chosen controller
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        if USE_CPG:
            # Need model to size CPG callback
            world = OlympicArena()
            world.spawn(core.spec, spawn_position=SPAWN_START)
            model = world.spec.compile()
            _ = mj.MjData(model)
            cpg_cb = cpg_controller_factory(model)
            ctrl = Controller(controller_callback_function=lambda m,d: cpg_cb(m,d,best_body), tracker=tracker)
        else:
            ctrl = Controller(controller_callback_function=lambda m,d: nn_controller(m,d,best_body), tracker=tracker)

        # Choose duration based on current best progress
        try:
            _, _, _, _, _, best_px = load_state()
        except Exception:
            best_px = 0.0
        demo_duration = schedule_duration(best_px)

        experiment(robot=core, controller=ctrl, duration=demo_duration, mode="launcher", spawn_pos=SPAWN_START)

        # Plot curves (if you run multiple gens, these will show growth)
        if len(gen_best_curve) and len(overall_best_curve):
            plt.figure()
            plt.plot(gen_best_curve, label="Gen Best")
            plt.plot(overall_best_curve, label="Overall Best")
            plt.xlabel("Generation")
            plt.ylabel("Fitness (−distance)")
            plt.legend()
            plt.title("Fitness Curves")
            plt.tight_layout()
            plt.savefig(DATA / "fitness_curves.png", dpi=150)
            console.log(f"Saved curves → {DATA / 'fitness_curves.png'}")

        # Optional trajectory viz if just ran a demo
        # show_xpos_history(tracker.history["xpos"][0])

        console.log(f"Best fitness so far: {overall_best_curve[-1]:.4f}" if overall_best_curve else "Run complete.")
        console.log(f"MuJoCo warnings (if any) logged at: {WARN_LOG}")

if __name__ == "__main__":
    # On windows/mp start method guard
    try:
        import multiprocessing as _mp
        if hasattr(_mp, "set_start_method"):
            _mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
