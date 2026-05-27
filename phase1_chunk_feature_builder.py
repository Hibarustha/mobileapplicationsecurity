import os
import pandas as pd
import math
from collections import Counter
import joblib



CLEAN_FOLDER =  "CLEANED_LOGS"
CHUNK_SIZE = 300
TOP_K_SYSCALLS = 80

KEY_SYSCALLS = [
    "read", "write", "ioctl",
    "socket", "connect",
    "recvfrom", "sendto",
    "open", "close",
    "openat", "unlinkat", "renameat",
    "fsync", "fdatasync", "fchmodat",
    "getsockopt", "setsockopt", "getsockname",
    "sendmsg", "recvmsg", "shutdown",
    "getdents64", "pipe2", "wait4",
    "exit", "nanosleep", "ptrace",
    "getpid", "gettid", "uname"
]

INFORMATIVE_PATTERNS = [
    ["read", "write"],
    ["ioctl", "recvfrom"],
    ["socket", "connect"],
    ["recvfrom", "sendto"],
]

PROCESS_SYSCALLS = [
    "wait4", "exit", "ptrace", "getpid", "gettid",
    "uname", "pipe2", "nanosleep", "clone",
    "sched_yield", "set_tid_address"
]

SIGNAL_SYSCALLS = [
    "rt_sigaction", "rt_sigprocmask", "rt_sigtimedwait",
    "rt_tgsigqueueinfo", "tgkill", "sigaltstack",
    "restart_syscall"
]

FILE_MUTATION_SYSCALLS = [
    "unlinkat", "renameat", "fsync", "fdatasync",
    "fchmodat", "ftruncate", "statfs", "getdents64",
    "faccessat", "lseek"
]

MEMORY_SYSCALLS = [
    "mmap", "munmap", "mprotect", "madvise"
]


def load_syscalls(filepath):
    with open(filepath, "r") as f:
        return [line.strip() for line in f.readlines()]


def pattern_count(sequence, pattern):
    count = 0
    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i:i+len(pattern)] == pattern:
            count += 1
    return count


def compute_entropy(counter, total):
    entropy = 0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def build_top_syscall_vocabulary():

    global_counter = Counter()

    for filename in os.listdir(CLEAN_FOLDER):
        filepath = os.path.join(CLEAN_FOLDER, filename)

        if os.path.isfile(filepath):
            syscalls = load_syscalls(filepath)
            global_counter.update(syscalls)

    top_syscalls = [syscall for syscall, _ in global_counter.most_common(TOP_K_SYSCALLS)]
    print("Top syscall vocabulary size:", len(top_syscalls))

    return top_syscalls


def extract_chunk_features(syscalls, label, app_name, top_syscalls):

    rows = []

    for start in range(0, len(syscalls), CHUNK_SIZE):

        chunk = syscalls[start:start + CHUNK_SIZE]

        if len(chunk) < 50:
            continue

        counter = Counter(chunk)
        total = len(chunk)

        features = {}

        selected_syscalls = list(dict.fromkeys(KEY_SYSCALLS + top_syscalls))

        for sc in selected_syscalls:
            features[f"freq_{sc}"] = counter.get(sc, 0)
            features[f"ratio_{sc}"] = counter.get(sc, 0) / total
            features[f"has_{sc}"] = 1 if counter.get(sc, 0) > 0 else 0

        features["total_syscalls"] = total
        features["unique_syscalls"] = len(set(chunk))
        features["unique_ratio"] = len(set(chunk)) / total

        # Network features
        network_count = (
            counter.get("socket", 0) +
            counter.get("connect", 0) +
            counter.get("getsockopt", 0) +
            counter.get("setsockopt", 0) +
            counter.get("getsockname", 0) +
            counter.get("sendmsg", 0) +
            counter.get("recvmsg", 0) +
            counter.get("sendto", 0) +
            counter.get("recvfrom", 0) +
            counter.get("shutdown", 0)
        )
        features["network_activity"] = network_count
        features["network_ratio"] = network_count / (total + 1)

        file_activity = (
            counter.get("open", 0) +
            counter.get("openat", 0) +
            counter.get("close", 0) +
            counter.get("read", 0) +
            counter.get("write", 0) +
            counter.get("unlinkat", 0) +
            counter.get("renameat", 0) +
            counter.get("fsync", 0) +
            counter.get("fdatasync", 0) +
            counter.get("fchmodat", 0) +
            counter.get("getdents64", 0)
        )
        features["file_activity"] = file_activity
        features["file_ratio"] = file_activity / (total + 1)

        process_activity = (
            counter.get("wait4", 0) +
            counter.get("exit", 0) +
            counter.get("ptrace", 0) +
            counter.get("getpid", 0) +
            counter.get("gettid", 0) +
            counter.get("uname", 0) +
            counter.get("pipe2", 0) +
            counter.get("nanosleep", 0)
        )
        features["process_activity"] = process_activity
        features["process_ratio"] = process_activity / (total + 1)

        signal_activity = sum(counter.get(sc, 0) for sc in SIGNAL_SYSCALLS)
        features["signal_activity"] = signal_activity
        features["signal_ratio"] = signal_activity / (total + 1)

        mutation_activity = sum(counter.get(sc, 0) for sc in FILE_MUTATION_SYSCALLS)
        features["mutation_activity"] = mutation_activity
        features["mutation_ratio"] = mutation_activity / (total + 1)

        memory_activity = sum(counter.get(sc, 0) for sc in MEMORY_SYSCALLS)
        features["memory_activity"] = memory_activity
        features["memory_ratio"] = memory_activity / (total + 1)

        # IOCTL dominance
        features["ioctl_intensity"] = counter.get("ioctl", 0) / (total + 1)
        features["read_write_balance"] = abs(counter.get("read", 0) - counter.get("write", 0)) / (total + 1)
        features["open_close_balance"] = abs((counter.get("open", 0) + counter.get("openat", 0)) - counter.get("close", 0)) / (total + 1)
        features["socket_connect_balance"] = abs(counter.get("socket", 0) - counter.get("connect", 0)) / (total + 1)
        features["network_to_file_ratio"] = network_count / (file_activity + 1)
        features["process_to_network_ratio"] = process_activity / (network_count + 1)

        most_common = counter.most_common(3)
        top_counts = [item[1] for item in most_common]
        while len(top_counts) < 3:
            top_counts.append(0)

        features["top1_ratio"] = top_counts[0] / total
        features["top2_ratio"] = top_counts[1] / total
        features["top3_ratio"] = top_counts[2] / total
        features["top3_sum_ratio"] = sum(top_counts) / total
        features["singletons"] = sum(1 for count in counter.values() if count == 1)
        features["singleton_ratio"] = features["singletons"] / (len(counter) + 1)
        features["rare_syscalls"] = sum(1 for count in counter.values() if count <= 2)
        features["rare_ratio"] = features["rare_syscalls"] / (len(counter) + 1)

        # Entropy
        features["syscall_entropy"] = compute_entropy(counter, total)

        for pattern in INFORMATIVE_PATTERNS:
            pattern_name = "_".join(pattern)
            features[f"pattern_{pattern_name}_count"] = pattern_count(chunk, pattern)

        features["label"] = label
        features["app_name"] = app_name

        rows.append(features)

    return rows


def build_chunk_dataset():

    all_rows = []
    top_syscalls = build_top_syscall_vocabulary()

    for filename in os.listdir(CLEAN_FOLDER):

        filepath = os.path.join(CLEAN_FOLDER, filename)

        if os.path.isfile(filepath):

            syscalls = load_syscalls(filepath)

            if len(syscalls) == 0:
                continue

            lowered_name = filename.lower()
            label = 1 if ("infected" in lowered_name or "malware" in lowered_name) else 0

            chunk_rows = extract_chunk_features(syscalls, label, filename, top_syscalls)
            all_rows.extend(chunk_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv("chunk_dataset.csv", index=False)

    print("Chunk dataset created.")
    print("Total chunks:", len(df))
    top_syscalls = build_top_syscall_vocabulary()
    joblib.dump(top_syscalls, "top_syscalls.pkl")


if __name__ == "__main__":
    build_chunk_dataset()