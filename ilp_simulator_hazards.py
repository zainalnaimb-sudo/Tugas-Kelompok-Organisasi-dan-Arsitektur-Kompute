# Simulator dasar: https://github.com/harshitasingh4863/instruction-level-parallelism-simulator.git
# Pengembangan tugas: implementasi RAW, WAR, WAW + dependency graph + register renaming
#
# Cara pakai:
#   python ilp_simulator_hazards.py          -> jalankan semua skenario
#   python ilp_simulator_hazards.py war      -> satu skenario (raw / war / waw / campuran / sample)
# Di Google Colab: tempel seluruh isi file ini ke satu cell lalu jalankan.

from dataclasses import dataclass
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


@dataclass
class Instruction:
    name: str
    destination: str
    sources: list


ALL_HAZARDS = ("RAW", "WAR", "WAW")
GRAPH_DIR = "graphs"

# -------------------------------------------------
# Instruction Sets
# -------------------------------------------------

# Set asli dari simulator (hanya mengandung RAW)
SAMPLE_SET = [
    Instruction("I1", "R1", ["R2", "R3"]),
    Instruction("I2", "R4", ["R5", "R6"]),
    Instruction("I3", "R7", ["R1", "R4"]),
    Instruction("I4", "R8", ["R9", "R10"]),
    Instruction("I5", "R11", ["R7", "R8"]),
    Instruction("I6", "R12", ["R13", "R14"]),
]

# RAW: I2 membaca R1 yang ditulis I1, I3 membaca R4 yang ditulis I2
RAW_SET = [
    Instruction("I1", "R1", ["R2", "R3"]),
    Instruction("I2", "R4", ["R1", "R5"]),
    Instruction("I3", "R6", ["R4", "R7"]),
]

# WAR: I2 menulis R2 yang dibaca I1, I3 menulis R4 yang dibaca I2
WAR_SET = [
    Instruction("I1", "R1", ["R2", "R3"]),
    Instruction("I2", "R2", ["R4", "R5"]),
    Instruction("I3", "R4", ["R6", "R7"]),
]

# WAW: I1, I2, I3 sama-sama menulis R1
WAW_SET = [
    Instruction("I1", "R1", ["R2", "R3"]),
    Instruction("I2", "R1", ["R4", "R5"]),
    Instruction("I3", "R1", ["R6", "R7"]),
]

# Campuran: RAW (I1->I2), WAR (I1->I3 dan I2->I4), WAW (I1->I4)
MIXED_SET = [
    Instruction("I1", "R1", ["R2", "R3"]),
    Instruction("I2", "R4", ["R1", "R5"]),
    Instruction("I3", "R2", ["R6", "R7"]),
    Instruction("I4", "R1", ["R8", "R9"]),
]

# -------------------------------------------------
# Function to check dependency
# -------------------------------------------------

def get_hazards(inst1, inst2, kinds=ALL_HAZARDS):
    """
    inst1 dieksekusi lebih dulu dari inst2.
    Mengembalikan list (jenis, register) untuk hazard yang membuat inst2 bergantung pada inst1.
    """
    found = []

    # RAW: inst2 membaca register yang ditulis inst1
    if "RAW" in kinds and inst1.destination in inst2.sources:
        found.append(("RAW", inst1.destination))

    # WAR: inst2 menulis register yang dibaca inst1
    if "WAR" in kinds and inst2.destination in inst1.sources:
        found.append(("WAR", inst2.destination))

    # WAW: inst1 dan inst2 menulis register yang sama
    if "WAW" in kinds and inst1.destination == inst2.destination:
        found.append(("WAW", inst2.destination))

    return found


def has_dependency(inst1, inst2, kinds=ALL_HAZARDS):
    """
    Returns True if inst2 depends on inst1.
    kinds = jenis hazard yang diperiksa (default: RAW, WAR, WAW).
    Pakai kinds=("RAW",) untuk perilaku simulator asli.
    """
    return len(get_hazards(inst1, inst2, kinds)) > 0


def get_edges(instructions, kinds=ALL_HAZARDS):
    """Semua ketergantungan sebagai list (indeks_asal, indeks_tujuan, jenis, register)."""
    edges = []
    for j in range(len(instructions)):
        for i in range(j):
            for kind, reg in get_hazards(instructions[i], instructions[j], kinds):
                edges.append((i, j, kind, reg))
    return edges


def fmt(instruction):
    return f"{instruction.destination} = {' + '.join(instruction.sources)}"


# -------------------------------------------------
# Sequential Execution
# -------------------------------------------------

def sequential_execution(instructions, verbose=True):
    if verbose:
        print("\n" + "=" * 50)
        print("SEQUENTIAL EXECUTION")
        print("=" * 50)

    cycle = 0
    for instruction in instructions:
        cycle += 1
        if verbose:
            print(f"Cycle {cycle}: Executing {instruction.name} ({fmt(instruction)})")

    if verbose:
        print(f"\nTotal Sequential Cycles: {cycle}")
    return cycle


# -------------------------------------------------
# ILP / Parallel Execution
# -------------------------------------------------

def parallel_execution(instructions, kinds=ALL_HAZARDS, verbose=True):
    if verbose:
        print("\n" + "=" * 50)
        print("INSTRUCTION-LEVEL PARALLEL EXECUTION  (hazard: " + ", ".join(kinds) + ")")
        print("=" * 50)

    completed = set()
    remaining = instructions.copy()
    cycle = 0

    while remaining:
        cycle += 1
        current_cycle = []

        for instruction in remaining:
            # Cek apakah semua dependensi sebelumnya sudah selesai
            dependencies_satisfied = True
            for previous in instructions:
                if previous.name == instruction.name:
                    break
                if has_dependency(previous, instruction, kinds):
                    if previous.name not in completed:
                        dependencies_satisfied = False
                        break
            if dependencies_satisfied:
                current_cycle.append(instruction)

        if verbose:
            print(f"\nCycle {cycle}:")
            for instruction in current_cycle:
                print(f"  -> {instruction.name}: {fmt(instruction)}")

        for instruction in current_cycle:
            completed.add(instruction.name)
            remaining.remove(instruction)

    if verbose:
        print(f"\nTotal ILP Cycles: {cycle}")
    return cycle


# -------------------------------------------------
# Dependency Analysis
# -------------------------------------------------

def show_dependencies(instructions, kinds=ALL_HAZARDS):
    print("\n" + "=" * 50)
    print("DEPENDENCY ANALYSIS")
    print("=" * 50)

    edges = get_edges(instructions, kinds)
    for i, j, kind, reg in edges:
        print(f"{instructions[j].name} depends on {instructions[i].name} ({kind} on {reg})")
    if not edges:
        print("No dependencies found.")

    counts = {k: sum(1 for e in edges if e[2] == k) for k in ALL_HAZARDS}
    print("Ringkasan: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return edges


# -------------------------------------------------
# Register Renaming (menghilangkan WAR dan WAW)
# -------------------------------------------------

def rename_registers(instructions):
    """
    Jika register tujuan sudah pernah muncul sebelumnya (dibaca/ditulis), ganti dengan
    register baru (nomor terbesar + 1). Pembacaan berikutnya memakai nama baru.
    """
    all_regs = [r for ins in instructions for r in [ins.destination] + ins.sources]
    next_num = max(int(r[1:]) for r in all_regs) + 1
    mapping, seen, renamed = {}, set(), []

    for ins in instructions:
        sources = [mapping.get(s, s) for s in ins.sources]
        dest = mapping.get(ins.destination, ins.destination)
        if dest in seen:
            dest = f"R{next_num}"
            next_num += 1
            mapping[ins.destination] = dest
        seen.update(sources)
        seen.add(dest)
        renamed.append(Instruction(ins.name, dest, sources))
    return renamed


# -------------------------------------------------
# Dependency Graph
# -------------------------------------------------

EDGE_COLOR = {"RAW": "#C0392B", "WAR": "#2471A3", "WAW": "#CA6F1E"}
EDGE_STYLE = {"RAW": "-", "WAR": "--", "WAW": ":"}


def compute_levels(instructions, kinds=ALL_HAZARDS):
    """Cycle paling awal (mulai 1) tiap instruksi; sama dengan hasil parallel_execution."""
    levels = [1] * len(instructions)
    for i, j, _, _ in get_edges(instructions, kinds):
        levels[j] = max(levels[j], levels[i] + 1)
    return levels


def draw_dependency_graph(instructions, title, filename, kinds=ALL_HAZARDS):
    """Node = instruksi (kolom = cycle ILP), panah = ketergantungan, label = jenis (register)."""
    os.makedirs(GRAPH_DIR, exist_ok=True)
    levels = compute_levels(instructions, kinds)
    n_levels = max(levels)
    columns = {}
    for idx, lv in enumerate(levels):
        columns.setdefault(lv, []).append(idx)
    max_rows = max(len(v) for v in columns.values())

    x_gap, y_gap = 2.8, 1.5
    pos = {}
    for lv, members in columns.items():
        for k, idx in enumerate(members):
            pos[idx] = ((lv - 1) * x_gap, ((len(members) - 1) / 2 - k) * y_gap)

    has_skip = any(levels[j] - levels[i] > 1 for i, j, _, _ in get_edges(instructions, kinds))
    extra = 1.3 if has_skip else 0.0   # ruang untuk panah lengkung yang melewati satu cycle
    fig, ax = plt.subplots(figsize=(max(5.5, 2.9 * n_levels + 1), max(3.2, 1.35 * max_rows + 1.6) + extra))

    for idx, ins in enumerate(instructions):
        x, y = pos[idx]
        ax.text(x, y, f"{ins.name}\n{fmt(ins)}", ha="center", va="center", fontsize=9.5,
                family="monospace", zorder=3,
                bbox=dict(boxstyle="round,pad=0.45", fc="#E1F5EE", ec="#0F6E56"))

    groups = {}
    for i, j, kind, reg in get_edges(instructions, kinds):
        groups.setdefault((i, j), []).append((kind, reg))

    label_specs = []
    for (i, j), items in groups.items():
        m = len(items)
        skip = levels[j] - levels[i] > 1
        for k, (kind, reg) in enumerate(items):
            rad = (k - (m - 1) / 2) * 0.35 + (0.6 if skip else 0.0)
            ax.add_patch(FancyArrowPatch(pos[i], pos[j], arrowstyle="-|>", mutation_scale=14,
                                         connectionstyle=f"arc3,rad={rad}", color=EDGE_COLOR[kind],
                                         linestyle=EDGE_STYLE[kind], linewidth=1.6,
                                         shrinkA=48, shrinkB=48, zorder=2))
            label_specs.append((i, j, kind, reg, rad))

    for lv in range(1, n_levels + 1):
        ax.text((lv - 1) * x_gap, max(p[1] for p in pos.values()) + 1.15, f"Cycle {lv}",
                ha="center", fontsize=9, color="#555555")

    handles = [plt.Line2D([0], [0], color=EDGE_COLOR[k], ls=EDGE_STYLE[k], lw=1.6, label=k) for k in kinds]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=False)
    ax.set_xlim(-1.5, (n_levels - 1) * x_gap + 1.5)
    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 1.0 - (1.9 if has_skip else 0.0), max(ys) + 1.6)
    ax.set_title(title, fontsize=12, pad=10)
    ax.axis("off")
    fig.tight_layout()
    for i, j, kind, reg, rad in label_specs:
        p1 = np.array(ax.transData.transform(pos[i]))
        p2 = np.array(ax.transData.transform(pos[j]))
        dx, dy = p2 - p1
        mid = (p1 + p2) / 2 + 0.5 * rad * np.array([dy, -dx])   # titik tengah kurva arc3
        lx, ly = ax.transData.inverted().transform(mid)
        ax.text(lx, ly, f"{kind} ({reg})", color=EDGE_COLOR[kind], fontsize=8.5,
                ha="center", va="center", zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
    path = os.path.join(GRAPH_DIR, filename)
    fig.savefig(path, dpi=150)
    if "ipykernel" in sys.modules:   # di Colab/Jupyter, tampilkan juga di notebook
        plt.show()
    plt.close(fig)
    print(f"Dependency graph disimpan: {path}")
    return path


# -------------------------------------------------
# Skenario
# -------------------------------------------------

SCENARIOS = {
    "raw":      ("RAW (Read After Write)", RAW_SET, False),
    "war":      ("WAR (Write After Read)", WAR_SET, True),
    "waw":      ("WAW (Write After Write)", WAW_SET, True),
    "campuran": ("Campuran RAW + WAR + WAW", MIXED_SET, True),
    "sample":   ("Contoh asli simulator", SAMPLE_SET, False),
}


def performance(sequential_cycles, parallel_cycles, n):
    speedup = sequential_cycles / parallel_cycles
    efficiency = speedup / n * 100
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"Sequential Cycles : {sequential_cycles}")
    print(f"ILP Cycles        : {parallel_cycles}")
    print(f"Speedup           : {speedup:.2f}x")
    print(f"ILP Efficiency    : {efficiency:.2f}%")


def run_scenario(key):
    title, instrs, with_renaming = SCENARIOS[key]
    print("\n" + "#" * 60)
    print(f"  SKENARIO: {title}")
    print("#" * 60)
    print("\nInstruction Set:")
    for ins in instrs:
        print(f"{ins.name}: {fmt(ins)}")

    show_dependencies(instrs)
    seq = sequential_execution(instrs)
    par = parallel_execution(instrs)
    performance(seq, par, len(instrs))
    draw_dependency_graph(instrs, f"Dependency graph {title}", f"{key}_graph.png")

    renamed_cycles = par
    if with_renaming:
        renamed = rename_registers(instrs)
        print("\n" + "-" * 50)
        print("SETELAH REGISTER RENAMING")
        print("-" * 50)
        for ins in renamed:
            print(f"{ins.name}: {fmt(ins)}")
        show_dependencies(renamed)
        renamed_cycles = parallel_execution(renamed)
        performance(seq, renamed_cycles, len(renamed))
        draw_dependency_graph(renamed, f"{title} setelah renaming", f"{key}_renamed_graph.png")

    return seq, par, renamed_cycles


# -------------------------------------------------
# Main Program
# -------------------------------------------------

def main(selected=None):
    print("\n" + "*" * 60)
    print("      INSTRUCTION-LEVEL PARALLELISM SIMULATOR")
    print("      Deteksi RAW, WAR, WAW + Dependency Graph")
    print("*" * 60)

    keys = selected or list(SCENARIOS)
    summary = []
    for key in keys:
        seq, par, ren = run_scenario(key)
        raw_only = parallel_execution(SCENARIOS[key][1], kinds=("RAW",), verbose=False)
        summary.append((SCENARIOS[key][0], seq, raw_only, par, ren))

    print("\n" + "=" * 78)
    print("RINGKASAN (jumlah cycle)")
    print("=" * 78)
    print(f"{'Skenario':<28}{'Sequential':>11}{'RAW saja':>10}{'Semua hazard':>14}{'Renaming':>10}")
    for name, seq, raw_only, par, ren in summary:
        print(f"{name:<28}{seq:>11}{raw_only:>10}{par:>14}{ren:>10}")
    print("\nCatatan: kolom 'RAW saja' = simulator asli (mengabaikan WAR dan WAW), sehingga")
    print("terlalu optimis untuk program yang punya WAR/WAW. Register renaming menghilangkan")
    print("WAR dan WAW, tetapi tidak bisa menghilangkan RAW.")


if __name__ == "__main__":
    chosen = [a for a in sys.argv[1:] if a in SCENARIOS]
    main(chosen or None)
