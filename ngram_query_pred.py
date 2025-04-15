import kenlm
from create_vocab_query import get_query_vocabulary as get_vocabulary


def query_level_next_prediction(model_path, eval_file, top_k=5, num_examples=10):
    model = kenlm.Model(model_path)

    with open(eval_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]

    print(f"🔍 Running next-query prediction on {num_examples} examples...\n")

    vocab_dict = get_vocabulary(eval_file, vocab_size=None)
    vocabulary = list(vocab_dict.keys())

    mrr_scores = []

    for i in range(len(queries) - 2):
        context = f"{queries[i]} {queries[i+1]}"  # Trigram context: 2 previous queries
        true_query = queries[i+2]

        scores = {
            q: model.score(f"{context} {q}", bos=False, eos=False)
            for q in vocabulary
        }

        sorted_preds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_preds = [q for q, _ in sorted_preds[:top_k]]

        if true_query in top_preds:
            rank = top_preds.index(true_query) + 1
            mrr = 1.0 / rank
        else:
            rank = None
            mrr = 0.0
        mrr_scores.append(mrr)

        print(f"🔹 Example {i+1}")
        print(f"Context     : [{queries[i]}] → [{queries[i+1]}]")
        print(f"True query  : {true_query}")
        print(f"Top-{top_k} : {top_preds}")
        print(f"MRR         : {mrr:.4f}")
        print("-" * 40)

        if i + 1 == num_examples:
            break

    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    print(f"\n📊 Average MRR over {len(mrr_scores)} examples: {avg_mrr:.4f}")

if __name__ == "__main__":
    model_path = "C:/Users/enesi/Desktop/DSV/DVK-Uppsats/3gram_query.binary"
    eval_file = "C:/Users/enesi/Desktop/DSV/DVK-Uppsats/data/ngram_eval.txt"
    query_level_next_prediction(model_path, eval_file)
