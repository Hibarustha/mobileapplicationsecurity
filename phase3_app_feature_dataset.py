print("🔥 FEATURE ENGINE V5 — LARGER CHUNKS + APP-LEVEL AGGREGATION 🔥")

import os
import numpy as np
import pandas as pd
from collections import Counter

CLEAN_FOLDER = "CLEANED_LOGS"
MIN_SYSCALL_THRESHOLD = 30

# Larger chunk = more context per sample, more chance of catching sparse network syscalls
CHUNK_SIZE = 300

# ─────────────────────────────────────────────
# LOAD & CHUNK
# ─────────────────────────────────────────────
def load_syscalls(path):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]

def chunk_sequence(seq, size, stride=150):
    # Overlapping chunks prevent behaviors from getting split at boundaries
    return [seq[i:i+size] for i in range(0, max(1, len(seq) - size + 1), stride)]


# ─────────────────────────────────────────────
# SYSCALL GROUPS — from diagnosis
# ─────────────────────────────────────────────
NET_ACTIVE = {
    "socket", "connect", "setsockopt", "getsockopt",
    "sendmsg", "sendto", "recvfrom", "recvmsg",
    "shutdown", "getsockname", "getpeername", "bind", "listen",
}
BENIGN_LEAN = {
    "fstat", "fcntl", "sched_setscheduler", "dup",
    "pread64", "pwrite64", "writev", "fdatasync",
    "faccessat", "getpriority", "geteuid", "ioctl",
}
MALWARE_LEAN = {
    "epoll_pwait", "read", "mmap", "write", "close",
    "munmap", "mprotect", "rt_sigprocmask",
    "socket", "connect", "setsockopt", "sendmsg",
    "shutdown", "getsockname", "getsockopt", "sigaltstack",
}
FILE_IO = {
    "openat", "read", "write", "close", "fstat",
    "pread64", "pwrite64", "writev", "fdatasync",
    "faccessat", "newfstatat",
}
MEMORY = {"mmap", "munmap", "mprotect", "madvise"}
SCHED  = {
    "futex", "sched_setscheduler", "getpriority",
    "prctl", "rt_sigprocmask", "sigaltstack",
}

ALL_SYSCALLS = [
    "epoll_pwait", "read", "write", "ioctl", "getuid",
    "recvfrom", "futex", "writev", "prctl", "fstat",
    "sched_setscheduler", "sendto", "mmap", "close",
    "munmap", "openat", "getpriority", "pread64", "dup",
    "newfstatat", "mprotect", "fcntl", "madvise", "faccessat",
    "rt_sigprocmask", "sigaltstack", "geteuid", "fdatasync",
    "pwrite64", "socket", "connect", "setsockopt", "getsockopt",
    "sendmsg", "shutdown", "getsockname", "getpeername",
    "recvmsg", "bind", "listen", "setresuid",
]


# ─────────────────────────────────────────────
# SLIDING WINDOW STATS
# ─────────────────────────────────────────────
def window_stats(seq, group_set, window_size=100):
    ratios = []
    step = window_size // 2
    for i in range(0, len(seq) - window_size + 1, step):
        window = seq[i:i + window_size]
        ratio = sum(1 for s in window if s in group_set) / window_size
        ratios.append(ratio)
    if not ratios:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(ratios)
    return float(arr.mean()), float(arr.max()), float(arr.std()), float(arr.min())


# ─────────────────────────────────────────────
# CHUNK-LEVEL FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(seq, label, name):
    total   = len(seq)
    counter = Counter(seq)
    present = set(seq)
    probs   = np.array(list(counter.values())) / total

    f = {
        "app_name":       name,
        "label":          label,
        "length":         total,
        "unique":         len(counter),
        "entropy":        -np.sum(probs * np.log2(probs + 1e-12)),
        "dominant_ratio": max(counter.values()) / total,
        "unique_ratio":   len(counter) / total,
    }

    def ratio(sc):
        return counter.get(sc, 0) / total

    def grp_ratio(grp):
        return sum(counter.get(s, 0) for s in grp) / total

    # ── Individual syscall ratios ──────────────
    for sc in ALL_SYSCALLS:
        f[f"sc_{sc}"] = ratio(sc)

    # ── Group ratios ──────────────────────────
    f["grp_net_active"]   = grp_ratio(NET_ACTIVE)
    f["grp_benign_lean"]  = grp_ratio(BENIGN_LEAN)
    f["grp_malware_lean"] = grp_ratio(MALWARE_LEAN)
    f["grp_file_io"]      = grp_ratio(FILE_IO)
    f["grp_memory"]       = grp_ratio(MEMORY)
    f["grp_sched"]        = grp_ratio(SCHED)

    net  = f["grp_net_active"]
    file = f["grp_file_io"]
    mem  = f["grp_memory"]
    sch  = f["grp_sched"]

    # ── Network intensity ──────────────────────
    socket_r  = ratio("socket")
    connect_r = ratio("connect")
    f["net_intensity"]       = net
    f["net_vs_file"]         = net / (file + 1e-6)
    f["net_vs_sched"]        = net / (sch  + 1e-6)
    f["net_vs_total_nonnw"]  = net / (1 - net + 1e-6)
    f["socket_connect_sum"]  = socket_r + connect_r
    f["socket_connect_prod"] = socket_r * connect_r
    f["socket_amplified"]    = socket_r * 10

    send_r = ratio("sendto") + ratio("sendmsg")
    recv_r = ratio("recvfrom") + ratio("recvmsg")
    f["send_recv_ratio"] = send_r / (recv_r + 1e-6)
    f["send_recv_diff"]  = send_r - recv_r
    f["send_recv_sum"]   = send_r + recv_r

    # ── Lean scores ───────────────────────────
    f["malware_lean_score"] = f["grp_malware_lean"]
    f["benign_lean_score"]  = f["grp_benign_lean"]
    f["lean_diff"]          = f["grp_malware_lean"] - f["grp_benign_lean"]
    f["lean_ratio"]         = f["grp_malware_lean"] / (f["grp_benign_lean"] + 1e-6)

    f["benign_penalty"] = (
        ratio("fstat")              * 2.0 +
        ratio("fcntl")              * 1.5 +
        ratio("sched_setscheduler") * 1.0 +
        ratio("pwrite64")           * 2.0 +
        ratio("fdatasync")          * 2.0 +
        ratio("dup")                * 1.0
    )
    f["malware_boost"] = (
        ratio("socket")      * 5.0 +
        ratio("connect")     * 5.0 +
        ratio("setsockopt")  * 4.0 +
        ratio("sendmsg")     * 3.0 +
        ratio("shutdown")    * 4.0 +
        ratio("getsockname") * 3.0 +
        ratio("rt_sigprocmask") * 1.5 +
        ratio("sigaltstack") * 1.5
    )
    f["mal_ben_score"] = f["malware_boost"] - f["benign_penalty"]

    # ── Sliding window burst detection ────────
    for grp_name, grp_set in [
        ("net",    NET_ACTIVE),
        ("memory", MEMORY),
        ("sched",  SCHED),
        ("file",   FILE_IO),
    ]:
        mean_, max_, std_, min_ = window_stats(seq, grp_set, window_size=100)
        f[f"win_{grp_name}_mean"] = mean_
        f[f"win_{grp_name}_max"]  = max_
        f[f"win_{grp_name}_std"]  = std_
        f[f"win_{grp_name}_min"]  = min_

    f["net_window_peak_ratio"] = f["win_net_max"] / (f["win_net_mean"] + 1e-6)

    # ── Statistical sequence features ─────────
    transitions = sum(seq[i] != seq[i-1] for i in range(1, total))
    f["transition_ratio"] = transitions / total

    runs, run = [], 1
    for i in range(1, total):
        if seq[i] == seq[i-1]:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    f["max_burst"]  = max(runs)
    f["avg_burst"]  = float(np.mean(runs))
    f["burst_std"]  = float(np.std(runs))
    f["num_bursts"] = len(runs)

    values = np.array(list(counter.values()), dtype=float)
    f["freq_mean"] = float(np.mean(values))
    f["freq_std"]  = float(np.std(values))
    f["freq_max"]  = float(np.max(values))
    f["freq_skew"] = float(
        np.mean(((values - np.mean(values)) / (np.std(values) + 1e-9)) ** 3)
    )

    for i, (sc, count) in enumerate(counter.most_common(10)):
        f[f"top{i}_freq"] = count / total

    rare_threshold = total * 0.005
    rare_calls = {k: v for k, v in counter.items() if v < rare_threshold}
    f["rare_count"] = len(rare_calls)
    f["rare_ratio"] = sum(rare_calls.values()) / total

    # ── Evasion-Resistant Distance Features ──
    def avg_distance(c1, c2):
        last_idx = -1
        dists = []
        for i, s in enumerate(seq):
            if s == c1: last_idx = i
            elif s == c2 and last_idx != -1:
                dists.append(i - last_idx)
        return float(np.mean(dists)) if dists else -1.0

    f["dist_socket_connect"] = avg_distance("socket", "connect")
    f["dist_openat_read"]    = avg_distance("openat", "read")

    # ── Presence flags ────────────────────────
    for sc in ["socket", "connect", "setsockopt", "sendmsg",
               "shutdown", "getsockname", "bind", "listen",
               "setresuid", "mprotect", "sigaltstack", "getsockopt",
               "getpeername", "recvmsg", "sendto"]:
        f[f"has_{sc}"] = int(sc in present)

    f["net_variety"]     = sum(1 for s in NET_ACTIVE   if s in present)
    f["benign_variety"]  = sum(1 for s in BENIGN_LEAN  if s in present)
    f["malware_variety"] = sum(1 for s in MALWARE_LEAN if s in present)

    # ── Interaction features ───────────────────
    f["entropy_x_net"]     = f["entropy"]      * net
    f["entropy_x_lean"]    = f["entropy"]      * f["lean_diff"]
    f["burst_x_net"]       = f["max_burst"]    * net
    f["net_variety_x_net"] = f["net_variety"]  * net
    f["malboost_x_net"]    = f["malware_boost"] * net
    f["win_net_x_variety"] = f["win_net_mean"] * f["net_variety"]
    f["socket_x_entropy"]  = socket_r          * f["entropy"]

    return f


# ─────────────────────────────────────────────
# APP-LEVEL AGGREGATION
# After all chunks of an app are extracted,
# add cross-chunk statistics as extra features.
# This gives the model app-wide context even at
# the chunk level.
# ─────────────────────────────────────────────
def add_app_level_features(df):
    print("  Computing app-level aggregation features...")

    df["app_base"] = df["app_name"].str.replace(r"_chunk\d+$", "", regex=True)

    # Features to aggregate across chunks of the same app
    agg_cols = [
        "grp_net_active", "net_intensity", "socket_amplified",
        "malware_boost", "benign_penalty", "mal_ben_score",
        "lean_diff", "entropy", "net_variety",
        "win_net_mean", "win_net_max",
        "sc_socket", "sc_connect", "sc_setsockopt", "sc_shutdown",
        "dist_socket_connect", "dist_openat_read",
        "has_socket", "has_setsockopt", "has_shutdown",
    ]

    # Compute per-app stats
    app_stats = df.groupby("app_base")[agg_cols].agg(["mean", "max", "std", "min"])
    app_stats.columns = [f"app_{col}_{stat}" for col, stat in app_stats.columns]
    app_stats = app_stats.reset_index()

    # Merge back onto chunk-level rows
    df = df.merge(app_stats, on="app_base", how="left")
    df = df.drop(columns=["app_base"])

    print(f"  Added {len(app_stats.columns) - 1} app-level features")
    return df


# ─────────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────────
def build_dataset():
    rows = []

    for filename in os.listdir(CLEAN_FOLDER):
        path = os.path.join(CLEAN_FOLDER, filename)
        seq  = load_syscalls(path)

        if len(seq) < MIN_SYSCALL_THRESHOLD:
            continue

        label  = 1 if "malware" in filename.lower() else 0
        chunks = chunk_sequence(seq, CHUNK_SIZE)

        for i, chunk in enumerate(chunks):
            if len(chunk) < 50:
                continue
            rows.append(extract_features(chunk, label, filename + f"_chunk{i}"))

    df = pd.DataFrame(rows).fillna(0)

    # Add app-level aggregated features
    df = add_app_level_features(df)
    df = df.fillna(0)

    df.to_csv("app_dataset.csv", index=False)

    n_malware = df["label"].sum()
    n_benign  = len(df) - n_malware

    print(f"✅ Dataset created  : {len(df)} samples")
    print(f"   Malware chunks  : {n_malware}")
    print(f"   Benign chunks   : {n_benign}")
    print(f"   Features        : {len(df.columns) - 2}")

if __name__ == "__main__":
    build_dataset()