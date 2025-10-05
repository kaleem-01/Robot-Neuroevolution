"""
Assignment 3 template code — with non-learner filter, dynamic duration,
multi-spawn training, per-generation save/resume, CPG option, warning capture,
AND: Initialize with NDE Genotypes, then evolve with a Graph-based EA.
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
import multiprocessing as mp
import hashlib

# === Third-party
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco import viewer
import networkx as nx
from networkx.readwrite import json_graph

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
USE_CPG: bool = False            # switch to CPG for speedier learning
RUN_ONE_GENERATION: bool = True  # run a single generation per execution (resume later)
SAVE_GRAPH_AND_BEST: bool = True
MAX_PROCESSES: int = 15          # cap multiprocessing

# ---------- Randomness, paths ----------
SEED = 42
RNG = np.random.default_rng(SEED)

SCRIPT_NAME = __file__.split("/")[-1][:-3] if "__file__" in globals() else "graph_ea_full"
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
POP_SIZE       = 10
ELITES         = max(1, POP_SIZE // 10)
GENERATIONS    = 3                # total desired (if not chunked)
TOURNAMENT_K   = max(2, POP_SIZE // 10)
MUT_BODY_SIGMA = 0.40
GENOTYPE_SIZE  = 64

# Base duration; dynamic schedule will override
BASE_SIM_DURATION = 15

ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# =========================
# Body genome (NDE inputs)
# =========================

@dataclass 
class GraphBody:
    core: Any
    graph: DiGraph
    fitness: float = -np.inf
    progress_x: float = 0.0

def sample_body_genome(rng: np.random.Generator) -> BodyGenome:
    return BodyGenome(
        type_p = rng.random(GENOTYPE_SIZE, dtype=np.float32),
        conn_p = rng.random(GENOTYPE_SIZE, dtype=np.float32),
        rot_p  = rng.random(GENOTYPE_SIZE, dtype=np.float32),
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
    rng = np.random.default_rng(seed)

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
        self.omega = 2*np.pi * (0.4 + 0.2*np.linspace(0,1,nu))
        self.K = 0.6
        self.alpha = 0.05
        self.amp = np.ones(nu, dtype=np.float64) * 0.6

def cpg_controller_factory(model: mj.MjModel) -> callable:
    nu = model.nu
    state = CPGState(nu)
    for i in range(nu):
        state.phase[i] = (i % 2) * np.pi
    last_time = {"t": 0.0}

    def cpg_cb(m: mj.MjModel, d: mj.MjData, body: Optional[BodyGenome] = None):
        t = d.time
        dt = max(1e-4, t - last_time["t"])
        last_time["t"] = t

        phase = state.phase
        for i in range(nu):
            coupling = 0.0
            if nu > 1:
                coupling += np.sin(phase[(i-1) % nu] - phase[i])
                coupling += np.sin(phase[(i+1) % nu] - phase[i])
            dphi = state.omega[i] + state.K * coupling
            phase[i] = (phase[i] + dphi * dt) % (2*np.pi)

        amp = state.amp
        if m.actuator_ctrlrange is not None and m.actuator_ctrlrange.size == 2*nu:
            lo = m.actuator_ctrlrange[:,0]; hi = m.actuator_ctrlrange[:,1]
            span = hi - lo
            span[span<1e-6] = 1.0
            target_amp = 0.45 * span
            amp += state.alpha * (target_amp - amp)

        u = amp * np.sin(phase)
        if m.actuator_ctrlrange is not None and m.actuator_ctrlrange.size == 2*nu:
            lo = m.actuator_ctrlrange[:,0]; hi = m.actuator_ctrlrange[:,1]
            d.ctrl[:nu] = np.clip(u, lo, hi)
        else:
            d.ctrl[:nu] = np.clip(u, -np.pi/2, np.pi/2)

    return cpg_cb

# =========================
# Simulation helpers
# =========================
def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    cartesian_distance = np.sqrt((xt - xc)**2 + (yt - yc)**2 + (zt - zc)**2)
    return -cartesian_distance

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
    return controller.tracker.history["xpos"][-1], sim_time

# ---------- Non-learner filter ----------
def quick_motion_screen(body: BodyGenome, spawn: List[float]) -> bool:
    """Short random/CPG rollout to kill non-learners early."""
    mj.set_mjcb_control(None)
    try:
        robot, graph = build_robot_from_body(body)
    except Exception:
        return False

    world = OlympicArena()
    world.spawn(robot.spec, spawn_position=spawn)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    model = world.spec.compile()
    data = mj.MjData(model)    

    if USE_CPG:
        cpg_cb = cpg_controller_factory(model)
        ctrl = Controller(controller_callback_function=lambda m,d: cpg_cb(m,d,body=body), tracker=tracker)
    else:
        ctrl = Controller(controller_callback_function=lambda m,d: nn_controller(m,d,body), tracker=tracker)

    try:
        if ctrl.tracker is not None:
            ctrl.tracker.setup(world.spec, data)
        hist, _ = experiment(robot=robot, controller=ctrl, duration=1.2, mode="simple", spawn_pos=spawn)
        dx = last_x(hist) - spawn[0]
        return (dx > 0.05) and (np.std(np.diff(np.array(hist)[:,0])) > 1e-4)
    except Exception:
        return False

# ---------- Dynamic duration schedule ----------
def schedule_duration(best_progress_x: float) -> float:
    if best_progress_x < 0.0:
        return 15.0
    elif best_progress_x < 3.0:
        return 45.0
    else:
        return 100.0

# ---------- Multi-spawn curriculum ----------
def choose_spawn(gen: int, rng: np.random.Generator) -> List[float]:
    if gen < 5:
        return SPAWN_START
    elif gen < 15:
        return SPAWN_POS_LIST[1]
    else:
        return SPAWN_POS_LIST[2]

# =========================
# GRAPH HELPERS & VARIATION
# =========================
def _clone_graph(G: "DiGraph") -> "DiGraph":
    return json_graph.node_link_graph(json_graph.node_link_data(G))

def _hash_graph(G: "DiGraph") -> str:
    s = json.dumps(json_graph.node_link_data(G), sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _repair_graph(G: "DiGraph") -> "DiGraph":
    """Lightweight repair: ensure root, connectivity; leave attributes to builder."""
    C = _clone_graph(G)
    # ensure root id=0 exists
    if 0 not in C.nodes:
        # relabel smallest node to 0
        any_id = min(C.nodes)
        C = nx.relabel_nodes(C, {any_id: 0}, copy=True)
    # keep largest weak component containing 0
    if not nx.is_weakly_connected(C):
        comps = list(nx.weakly_connected_components(C))
        comps.sort(key=len, reverse=True)
        if 0 in comps[0]:
            C = C.subgraph(comps[0]).copy()
        else:
            C = C.subgraph(comps[0]).copy()
            # ensure a 0 node
            if 0 not in C.nodes:
                any_id = min(C.nodes)
                C = nx.relabel_nodes(C, {any_id: 0}, copy=True)
    return C

def graph_from_genome(body: BodyGenome) -> "DiGraph":
    core, graph = build_robot_from_body(body)
    return _repair_graph(graph)

def graph_crossover_graft(A: "DiGraph", B: "DiGraph") -> "DiGraph":
    A = _repair_graph(A); B = _repair_graph(B)
    candidates = [n for n in A.nodes if n != 0]
    if not candidates:
        return _clone_graph(B)
    cut = int(RNG.choice(candidates))
    sub = set(nx.dfs_tree(A, source=cut).nodes())

    child = _clone_graph(B)
    next_id = (max(child.nodes) + 1) if child.nodes else 1

    id_map = {}
    for n in sub:
        id_map[n] = next_id
        child.add_node(next_id, **A.nodes[n])
        next_id += 1

    for u, v, ed in A.edges(data=True):
        if u in sub and v in sub:
            child.add_edge(id_map[u], id_map[v], **ed)

    attach = int(RNG.choice(list(child.nodes)))
    if attach == id_map[cut]:
        attach = 0 if 0 in child.nodes else attach
    # minimal edge attrs; your builder may ignore or infer
    child.add_edge(attach, id_map[cut], **({"orient": 0, "port": 0}))
    return _repair_graph(child)


def mut_rewire_edge(G: "DiGraph") -> "DiGraph":
    if G.number_of_edges() == 0:
        return G
    C = _clone_graph(G)
    u, v = list(C.edges)[int(RNG.integers(0, C.number_of_edges()))]
    C.remove_edge(u, v)
    candidates = [n for n in C.nodes if n != v]
    if not candidates:
        return _repair_graph(C)
    new_parent = int(RNG.choice(candidates))
    if nx.has_path(C, v, new_parent):  # avoid cycle
        C.add_edge(u, v)
        return _repair_graph(C)
    C.add_edge(new_parent, v, **({"orient": 0, "port": 0}))
    return _repair_graph(C)

def mut_tweak_node(G: "DiGraph") -> "DiGraph":
    C = _clone_graph(G)
    n = int(RNG.choice(list(C.nodes)))
    nd = C.nodes[n]
    for key in ("mass", "damping"):
        if key in nd and isinstance(nd[key], (int, float)):
            nd[key] = float(np.clip(nd[key] + RNG.normal(0, 0.05), 0.01, 5.0))
    if "size" in nd and isinstance(nd["size"], (list, tuple)) and len(nd["size"]) == 3:
        nd["size"] = [float(np.clip(s + RNG.normal(0, 0.005), 0.005, 0.2)) for s in nd["size"]]
    return _repair_graph(C)

GRAPH_MUTS = [ mut_rewire_edge, mut_tweak_node]

def mutate_graph(G: "DiGraph", ops: int = 2) -> "DiGraph":
    C = _clone_graph(G)
    k = max(1, ops)
    for _ in range(k):
        C = RNG.choice(GRAPH_MUTS)(C)
    return _repair_graph(C)

# =========================
# GRAPH rollout & evaluation
# =========================
def rollout_and_score_graph(graph: "DiGraph", duration: float, spawn: List[float]) -> Tuple[float, float]:
    robot = construct_mjspec_from_graph(graph)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    if USE_CPG:
        world = OlympicArena()
        world.spawn(robot.spec, spawn_position=spawn)
        model = world.spec.compile()
        data = mj.MjData(model)
        cpg_cb = cpg_controller_factory(model)
        ctrl = Controller(controller_callback_function=lambda m, d: cpg_cb(m, d, body=None), tracker=tracker)
    else:
        ctrl = Controller(controller_callback_function=lambda m, d: nn_controller(m, d, None), tracker=tracker)

    hist, _sim_t = experiment(robot=robot, controller=ctrl, duration=duration, mode="simple", spawn_pos=spawn)
    fit = fitness_function(hist)
    px  = last_x(hist)
    return fit, px

def _worker_eval_graph(args):
    g_json, duration, spawn = args
    err_buf = io.StringIO()
    with contextlib.redirect_stderr(err_buf):
        G = json_graph.node_link_graph(json.loads(g_json))
        # try:
        return rollout_and_score_graph(G, duration=duration, spawn=spawn)
        # except Exception:
            # return (-1e9, -1e9)

def evaluate_population_graph(pop_graphs: List["DiGraph"], duration: float, gen: int) -> List[Tuple[float, float]]:
    spawn_choices = [choose_spawn(gen, RNG) for _ in pop_graphs]
    packs = [(json.dumps(json_graph.node_link_data(G)), duration, spawn_choices[i]) for i, G in enumerate(pop_graphs)]
    with mp.Pool(processes=min(mp.cpu_count(), MAX_PROCESSES)) as pool:
        return pool.map(_worker_eval_graph, packs)

# =========================
# Init: GENOTYPES -> GRAPHS (with non-learner screen)
# =========================
def init_population_from_genotypes(pop_size: int) -> Tuple[List[GraphBody], List[BodyGenome]]:
    genomes: List[BodyGenome] = []
    graphs:  List[GraphBody]  = []
    attempts = 0
    while len(graphs) < pop_size and attempts < pop_size * 20:
        attempts += 1
        g = sample_body_genome(RNG)
        if not quick_motion_screen(g, SPAWN_START):
            continue
        try:
            core, graph = build_robot_from_body(g)
            G = _repair_graph(graph)
            graphs.append(GraphBody(core=core, graph=G, fitness=-np.inf, progress_x=0.0))
            genomes.append(g)
        except Exception:
            continue
    if len(graphs) == 0:
        for _ in range(pop_size):
            g = sample_body_genome(RNG)
            core, graph = build_robot_from_body(g)
            G = _repair_graph(graph)
            graphs.append(GraphBody(core=core, graph=G, fitness=-np.inf, progress_x=0.0))
            genomes.append(g)
    return graphs, genomes

# =========================
# Tournament selection (graph)
# =========================
def tournament_select_graph(pop_graphs: List[GraphBody], k=TOURNAMENT_K) -> GraphBody:
    idxs = RNG.choice(len(pop_graphs), size=min(k, len(pop_graphs)), replace=False)
    best_i = int(idxs[0])
    best_f = pop_graphs[best_i].fitness
    for j in idxs[1:]:
        j = int(j)
        if pop_graphs[j].fitness > best_f:
            best_i = j; best_f = pop_graphs[j].fitness
    return pop_graphs[best_i]

# =========================
# Persistence (graphs)
# =========================
STATE_PATH = DATA / "evo_state.pkl"
CURVES_PATH = DATA / "curves.pkl"
BEST_JSON_PATH = DATA / "robot_graph.json"
BEST_CTRL_PATH = DATA / "best_ctrl.pkl"  # placeholder if you later evolve controllers

def save_state(gen: int, population_graphs: List[GraphBody], best_graph: GraphBody,
               gen_curve: List[float], overall_curve: List[float], best_progress: float):
    state = {
        "gen": gen,
        "population_graphs": [
            (json_graph.node_link_data(ind.graph), float(ind.fitness), float(ind.progress_x))
            for ind in population_graphs
        ],
        "best_graph": (json_graph.node_link_data(best_graph.graph),
                       float(best_graph.fitness), float(best_graph.progress_x)),
        "best_progress_x": float(best_progress),
    }
    with open(STATE_PATH, "wb") as f:
        pickle.dump(state, f)
    with open(CURVES_PATH, "wb") as f:
        pickle.dump({"gen_best": gen_curve, "overall_best": overall_curve}, f)
    if SAVE_GRAPH_AND_BEST:
        try:
            save_graph_as_json(best_graph.graph, BEST_JSON_PATH)
        except Exception:
            pass

def load_state() -> Tuple[int, List[GraphBody], GraphBody, List[float], List[float], float]:
    if not STATE_PATH.exists():
        raise FileNotFoundError
    with open(STATE_PATH, "rb") as f:
        s = pickle.load(f)

    pop_graphs: List[GraphBody] = []
    for g_json, fit, px in s["population_graphs"]:
        G = json_graph.node_link_graph(g_json)
        core = construct_mjspec_from_graph(G)
        pop_graphs.append(GraphBody(core=core, graph=G, fitness=float(fit), progress_x=float(px)))

    best_g_json, best_fit, best_px = s["best_graph"]
    bestG = json_graph.node_link_graph(best_g_json)
    best_core = construct_mjspec_from_graph(bestG)
    best_graph = GraphBody(core=best_core, graph=bestG, fitness=float(best_fit), progress_x=float(best_px))

    if CURVES_PATH.exists():
        with open(CURVES_PATH, "rb") as f:
            curves = pickle.load(f)
        gen_curve = curves.get("gen_best", [])
        overall_curve = curves.get("overall_best", [])
    else:
        gen_curve, overall_curve = [], []
    best_progress_x = float(s.get("best_progress_x", 0.0))
    return int(s["gen"]), pop_graphs, best_graph, gen_curve, overall_curve, best_progress_x

# =========================
# EVOLVE: Initialize with genotypes, then Graph EA
# =========================
def evolve() -> Tuple[BodyGenome, List[float], List[float]]:
    """Initialize with NDE genotypes, THEN evolve graphs with a graph EA."""
    # Resume or fresh
    try:
        start_gen, pop_graphs, best_graph, gen_best_curve, overall_best_curve, best_px = load_state()
        console.log(f"Resuming from gen {start_gen} with {len(pop_graphs)} graphs.")
        cur_gen = start_gen
    except Exception:
        pop_graphs, _genomes = init_population_from_genotypes(POP_SIZE)
        best_graph = pop_graphs[0]
        gen_best_curve, overall_best_curve = [], []
        best_px = 0.0
        cur_gen = 0

    while cur_gen < GENERATIONS:
        duration = schedule_duration(best_px)

        # Evaluate (graphs)
        results = evaluate_population_graph([gb.graph for gb in pop_graphs], duration=duration, gen=cur_gen)
        for i, (fit, px) in enumerate(results):
            pop_graphs[i].fitness = float(fit)
            pop_graphs[i].progress_x = float(px)

        # Rank & bookkeeping
        pop_graphs.sort(key=lambda gb: gb.fitness, reverse=True)
        gen_best = pop_graphs[0].fitness
        if (best_graph is None) or (gen_best > best_graph.fitness):
            best_graph = GraphBody(
                core=construct_mjspec_from_graph(pop_graphs[0].graph),
                graph=_clone_graph(pop_graphs[0].graph),
                fitness=pop_graphs[0].fitness,
                progress_x=pop_graphs[0].progress_x,
            )
        best_px = max(best_px, pop_graphs[0].progress_x)
        gen_best_curve.append(gen_best)
        overall_best_curve.append(best_graph.fitness)
        console.log(f"[Gen {cur_gen+1:03d}] best={gen_best:.3f} overall={best_graph.fitness:.3f} px={best_px:.2f}")

        # Elitism + offspring
        elites = pop_graphs[:ELITES]
        fits = [gb.fitness for gb in pop_graphs]

        def tselect(k=TOURNAMENT_K) -> GraphBody:
            idxs = RNG.choice(len(pop_graphs), size=min(k, len(pop_graphs)), replace=False)
            best_i = int(idxs[0]); best_f = fits[best_i]
            for j in idxs[1:]:
                j = int(j)
                if fits[j] > best_f:
                    best_i = j; best_f = fits[j]
            return pop_graphs[best_i]

        offspring: List[GraphBody] = []
        while len(offspring) < POP_SIZE - ELITES:
            p1 = tselect(); p2 = tselect()
            child_graph = graph_crossover_graft(p1.graph, p2.graph)
            child_graph = mutate_graph(child_graph, ops=int(RNG.integers(1, 4)))
            child_core = construct_mjspec_from_graph(child_graph)
            offspring.append(GraphBody(core=child_core, graph=child_graph, fitness=-np.inf, progress_x=0.0))

        pop_graphs = elites + offspring

        # Save state and (optionally) stop after one generation
        save_state(cur_gen+1, pop_graphs, best_graph, gen_best_curve, overall_best_curve, best_px)
        if RUN_ONE_GENERATION:
            break
        cur_gen += 1

    # Return a dummy genome for compatibility; graph is saved to disk/state
    dummy_best = BodyGenome(
        type_p=np.zeros(GENOTYPE_SIZE, dtype=np.float32),
        conn_p=np.zeros(GENOTYPE_SIZE, dtype=np.float32),
        rot_p =np.zeros(GENOTYPE_SIZE, dtype=np.float32),
        fitness=best_graph.fitness,
        progress_x=best_graph.progress_x,
    )
    return dummy_best, gen_best_curve, overall_best_curve

# =========================
# Plotting / viz
# =========================
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

# =========================
# MuJoCo warning silencer (stderr filter)
# =========================
class _FilterStderr(io.TextIOBase):
    def __init__(self, real_stderr, log_path: Path):
        self._real = real_stderr
        self._log = open(log_path, "a", buffering=1, encoding="utf-8")
        self._buf = ""

    def write(self, s):
        self._buf += s
        out = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if "WARNING: " in line:
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

# =========================
# Main
# =========================
def main() -> None:
    with silence_mujoco_warnings():
        best_body, gen_best_curve, overall_best_curve = evolve()

        # Load the best graph from disk (saved in save_state) and launch a demo
        if not BEST_JSON_PATH.exists():
            try:
                _, _, best_graph, _, _, _ = load_state()
                save_graph_as_json(best_graph.graph, BEST_JSON_PATH)
            except Exception:
                console.log("No saved graph found; skipping demo.")
                return

        with open(BEST_JSON_PATH, "r", encoding="utf-8") as f:
            gjson = json.load(f)
        graph = json_graph.node_link_graph(gjson)
        core = construct_mjspec_from_graph(graph)

        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        if USE_CPG:
            world = OlympicArena()
            world.spawn(core.spec, spawn_position=SPAWN_START)
            model = world.spec.compile()
            data = mj.MjData(model)
            cpg_cb = cpg_controller_factory(model)
            ctrl = Controller(controller_callback_function=lambda m,d: cpg_cb(m,d,best_body), tracker=tracker)
        else:
            ctrl = Controller(controller_callback_function=lambda m,d: nn_controller(m,d,best_body), tracker=tracker)

        try:
            _, _, _, _, _, best_px = load_state()
        except Exception:
            best_px = 0.0
        demo_duration = schedule_duration(best_px)

        # Launch viewer demo
        experiment(robot=core, controller=ctrl, duration=demo_duration, mode="launcher", spawn_pos=SPAWN_START)

        # Plot curves
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

        console.log(f"Best fitness so far: {overall_best_curve[-1]:.4f}" if overall_best_curve else "Run complete.")
        console.log(f"MuJoCo warnings (if any) logged at: {WARN_LOG}")

if __name__ == "__main__":
    try:
        import multiprocessing as _mp
        if hasattr(_mp, "set_start_method"):
            _mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
