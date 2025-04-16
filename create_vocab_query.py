import os
import json
import random
from collections import Counter

# === Paths ===
BASE_PATH = "C:/Users/enesi/Desktop/DSV/DVK-Uppsats/"
INPUT_DIR = BASE_PATH + "aol_processed/processed_files/"
TRAIN_FILE_PATH = BASE_PATH + "data/ngram_train.txt"
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

    print("🔁 Starting query-level vocabulary creation...")

    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(input_dir, filename)
        print(f"📄 Processing: {filename}")

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
                    print(f"⚠️ Error processing line: {line} - {e}")
                    continue

    random.seed(42)
    split_index = int(0.8 * len(all_queries))
    train_data = all_queries[:split_index]
    eval_data = all_queries[split_index:]

    with open(TRAIN_FILE_PATH, "w", encoding="utf-8") as train_file:
        train_file.write("\n".join(train_data))
    with open(EVAL_FILE_PATH, "w", encoding="utf-8") as eval_file:
        eval_file.write("\n".join(eval_data))

    print(f"✅ Train set saved to: {TRAIN_FILE_PATH}")
    print(f"✅ Eval set saved to: {EVAL_FILE_PATH}")

    # === Build Vocabulary ===
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

    print("✅ Query-level vocabulary saved to:", VOCAB_DICT_PATH)
    print("📊 Vocabulary Stats:")
    print(json.dumps(vocab_stats, indent=4))

    return vocab_dict, vocab_stats

def get_query_vocabulary(query_file, vocab_size=None):
    from collections import Counter
    word_counts = Counter()

    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            query = line.strip().lower()
            if query:
                word_counts.update([query])  # Treat whole query as a token

    most_common = word_counts.most_common(vocab_size) if vocab_size else word_counts.items()
    return {query: idx for idx, (query, _) in enumerate(most_common)}


if __name__ == "__main__":
    create_query_vocab(INPUT_DIR)
