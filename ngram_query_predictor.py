# @title === Imports & Configuration ===

import os
import re
import numpy as np
import pandas as pd
import csv
import time
import random
from collections import Counter
import kenlm
import psutil
import json
import matplotlib.pyplot as plt
import threading
import subprocess

# === Local Paths ===
BASE_PATH = "C:/Users/enesi/Desktop/DSV/DVK-Uppsats/"
INPUT_DIR = BASE_PATH + "aol_processed/processed_files/"
TRAIN_FILE_PATH = BASE_PATH + "data/ngram_train_sequence.txt"
EVAL_FILE_PATH = BASE_PATH + "data/ngram_eval.txt"
VOCAB_DICT_PATH = BASE_PATH + "data/query_vocab_dict.json"
VOCAB_STATS_PATH = BASE_PATH + "data/query_vocab_stats.json"


def clean_query(query):
    if not isinstance(query, str):
        return None
    return query.strip().lower()

def create_query_vocab(input_dir, vocab_size=None):
    query_counts = Counter()
    all_queries = []

    print(" Starting query-level vocabulary creation...")

    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(input_dir, filename)
        print(f" Processing: {filename}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    query = clean_query(parts[1])
                    if query:
                        query_counts.update([query])
                        all_queries.append(query)
                except Exception as e:
                    print(f" Error processing line: {line} - {e}")
                    continue

    random.seed(42)
    split_index = int(0.8 * len(all_queries))
    train_data = all_queries[:split_index]
    eval_data = all_queries[split_index:]

    with open(TRAIN_FILE_PATH, "w", encoding="utf-8") as train_file:
        train_file.write(" ".join(train_data))
    with open(EVAL_FILE_PATH, "w", encoding="utf-8") as eval_file:
        eval_file.write("\n".join(eval_data))

    print(f" Train set saved to: {TRAIN_FILE_PATH}")
    print(f" Eval set saved to: {EVAL_FILE_PATH}")

    most_common_queries = query_counts.most_common(vocab_size) if vocab_size else query_counts.items()
    vocab_dict = {query: idx for idx, (query, _) in enumerate(most_common_queries)}
    actual_vocab_size = len(vocab_dict)
    total_queries = sum(query_counts.values())
    covered = sum(query_counts[q] for q in vocab_dict)
    coverage = (covered / total_queries) * 100 if total_queries > 0 else 0

    vocab_stats = {
        "Requested_Vocabulary_Size": vocab_size if vocab_size else "Full",
        "Actual_Vocabulary_Size": actual_vocab_size,
        "Total_Queries_Found": total_queries,
        "Total_Unique_Queries_Found": len(query_counts),
        "Coverage_Percentage_Of_Top_Queries": round(coverage, 2),
    }

    with open(VOCAB_DICT_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f)
    with open(VOCAB_STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab_stats, f)

    print(" Vocabulary saved to:", VOCAB_DICT_PATH)
    print(" Vocabulary Stats:")
    print(json.dumps(vocab_stats, indent=4))

    return vocab_dict, vocab_stats

def get_query_vocabulary(query_file, vocab_size=None):
    word_counts = Counter()
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            query = line.strip()
            if query:
                word_counts.update([query])
    most_common = word_counts.most_common(vocab_size) if vocab_size else word_counts.items()
    return {query: idx for idx, (query, _) in enumerate(most_common)}

# === Evaluation ===

def query_level_next_prediction(model_path, eval_file, top_k=5, n=3, sample_size=1000):
    vocab_dict = get_query_vocabulary(eval_file)
    vocabulary = list(vocab_dict.keys())
    model = kenlm.Model(model_path)

    with open(eval_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]

    if sample_size is not None and sample_size < len(queries):
        queries = queries[:sample_size]
        # queries = random.sample(queries, sample_size)


    print(f" Evaluating {len(queries)} queries with {n}-gram model...")

    mrr_scores = []
    top1_scores = []
    context_window = n - 1
    start_time = time.time()

    for i in range(len(queries) - context_window):
        context = " ".join(queries[i:i + context_window])
        true_query = queries[i + context_window]

        scores = {q: model.score(f"{context} {q}", bos=False, eos=False) for q in vocabulary}
        sorted_preds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_preds = [q for q, _ in sorted_preds[:top_k]]

        rank = next((r for r, (q, _) in enumerate(sorted_preds, 1) if q == true_query), None)
        mrr = 1.0 / rank if rank else 0.0
        mrr_scores.append(mrr)

        top1_scores.append(1 if rank == 1 else 0)

        if i < 10:
            print(f"Example {i+1}")
            print(f"Context     : {context}")
            print(f"True query  : {true_query}")
            print(f"Top-{top_k} : {top_preds}")
            print(f"Rank        : {rank if rank else 'Not found'}")
            print(f"MRR         : {mrr:.4f}")
            print("-" * 40)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f" Evaluated {i+1} examples in {elapsed:.2f} seconds")

    total = len(mrr_scores)
    avg_mrr = sum(mrr_scores) / total
    accuracy = sum(top1_scores) / total

    print(f"\n Final Results ({n}-gram):")
    print(f"Examples        : {total}")
    print(f"Mean MRR        : {avg_mrr:.4f}")
    print(f"Top-1 Accuracy  : {accuracy:.4f} ({sum(top1_scores)}/{total})")

    return avg_mrr, accuracy


# === Resource Logger ===

class ResourceLogger:
    def __init__(self, output_path, interval=10):
        self.output_path = output_path
        self.interval = interval
        self.stop_event = threading.Event()
        self.logs = []
        self.thread = None
        self.start_time = None
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def _collect_resources(self):
        elapsed = time.time() - self.start_time
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu = psutil.cpu_percent(interval=1)
        try:
            command = [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            gpu_output = result.stdout.strip()
            gpus = []
            for line in gpu_output.split('\n'):
                if not line.strip():
                    continue
                idx, name, util, total, used, free = line.split(',')
                gpus.append({
                    'gpu_id': int(idx.strip()),
                    'gpu_name': name.strip(),
                    'gpu_load': float(util.strip()),
                    'gpu_memory_total': int(total.strip()),
                    'gpu_memory_used': int(used.strip()),
                    'gpu_memory_free': int(free.strip())
                })
        except Exception as e:
            print(f"GPU error: {e}")
            gpus = []

        self.logs.append({
            'elapsed_seconds': elapsed,
            'cpu_percent': cpu,
            'memory': {
                'percent': memory.percent
            },
            'disk': {
                'percent': disk.percent
            },
            'gpus': gpus
        })

    def _logging_thread(self):
        while not self.stop_event.is_set():
            self._collect_resources()
            self.stop_event.wait(self.interval)

    def start(self):
        print(f" Starting resource logger...")
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._logging_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        with open(self.output_path, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f" Resource logs saved to {self.output_path}")

# === Plotting ===

def plot_model_performance(results, output_path):
    ngrams = list(results.keys())

    mrrs = [results[n]['MRR'] * 100 for n in ngrams]
    accs = [results[n]['Accuracy'] * 100 for n in ngrams]

    x = np.arange(len(ngrams))
    width = 0.35

    plt.figure(figsize=(8, 5))
    bars_mrr = plt.bar(x - width/2, mrrs, width, label='MRR')
    bars_acc = plt.bar(x + width/2, accs, width, label='Accuracy')
    plt.xticks(x, [f"{n}-gram" for n in ngrams])

    max_y = max(max(mrrs), max(accs)) + 0.2
    plt.ylim(0, max_y)
    plt.title("Model Performance")
    plt.xlabel("N-gram")
    plt.ylabel("Score (%)")
    plt.legend()

    # Add % labels on top of bars
    for bar in bars_mrr + bars_acc:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f" Saved performance plot to {output_path}")


def plot_resource_utilization_from_json(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    times = [entry['elapsed_seconds'] / 60.0 for entry in data]
    cpu = [entry['cpu_percent'] for entry in data]
    mem = [entry['memory']['percent'] for entry in data]
    disk = [entry['disk']['percent'] for entry in data]
    gpu = [entry['gpus'][0]['gpu_load'] if entry['gpus'] else 0 for entry in data]

    plt.figure(figsize=(10, 6))
    plt.plot(times, cpu, label="CPU %")
    plt.plot(times, mem, label="Memory %")
    plt.plot(times, disk, label="Disk %")
    plt.plot(times, gpu, label="GPU %")
    plt.xlabel("Elapsed Time (min)")
    plt.ylabel("Usage %")
    plt.title("System Resource Usage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f" Saved resource usage plot to {output_path}")

def plot_mrr_vs_context(results, output_path="plots/ngram_mrr_vs_context.pdf"):
    context_lengths = list(results.keys())
    mean_mrrs = [results[n]["MRR"] * 100 for n in context_lengths]  

    plt.figure(figsize=(8, 5))
    plt.plot(context_lengths, mean_mrrs, color='blue', marker='o', linewidth=2)
    plt.title("N-gram Context Length vs Mean MRR")
    plt.xlabel("Context Length (N-gram)")
    plt.ylabel("Mean MRR (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(context_lengths)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f" Saved MRR vs Context plot to {output_path}")
    plt.close()


# === MAIN ===

if __name__ == "__main__":
    create_query_vocab(INPUT_DIR)

    logger = ResourceLogger(output_path=BASE_PATH + "logs/resource_log.json", interval=10)
    logger.start()

    results = {}
    for n in [2, 3, 4, 5]:
        model_path = BASE_PATH + f"data/ngram_{n}.binary"
        print(f"\n Evaluating {n}-gram model...")
        mrr, acc = query_level_next_prediction(
            model_path=model_path,
            eval_file=EVAL_FILE_PATH,
            top_k=5,
            n=n,
            sample_size=1000
        )
        results[n] = {"MRR": mrr, "Accuracy": acc}

    logger.stop()

    plot_model_performance(results, BASE_PATH + "plots/ngram_performance.pdf")
    plot_resource_utilization_from_json(BASE_PATH + "logs/resource_log.json", BASE_PATH + "plots/resource_usage.pdf")
    plot_mrr_vs_context(results, BASE_PATH + "plots/ngram_mrr_vs_context.pdf")


    print("\n All done.")
