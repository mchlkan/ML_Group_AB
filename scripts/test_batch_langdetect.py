"""
Accuracy & speed test: batch-of-4 LLM language detection vs single-title baseline.

Uses existing song_language_cache.json as ground truth (single-title results).
Sends the same titles to the LLM in batches of 4 and compares output.

Requirements: Ollama running locally with qwen3.5:4b pulled.
"""

import json, time, random, requests, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "datasets" / "v3_features" / "song_language_cache.json"
OLLAMA_MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT_SINGLE = (
    "Detect the language each song is sung in. "
    "Use the artist origin country AND the title language as clues. "
    "Most artists sing in the language of their origin country. "
    "Reply with ONLY the ISO 639-1 two-letter code. Nothing else."
)

SYSTEM_PROMPT_BATCH = (
    "Detect the language each song is sung in. "
    "Use the artist origin country AND the title language as clues. "
    "Most artists sing in the language of their origin country. "
    "Reply with ONLY the ISO 639-1 two-letter codes, one per line, in the same order. "
    "No numbering, no extra text."
)

N_SAMPLES = 50

def load_test_data():
    import duckdb
    cache = json.loads(CACHE_PATH.read_text())

    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW v2 AS SELECT * FROM read_parquet(
            '{ROOT}/datasets/v2/full/**/*.parquet', hive_partitioning=true)
    """)
    tracks = con.execute("""
        SELECT track_id, title, artist, origin_country FROM (
            SELECT track_id, title, artist, region AS origin_country,
                   ROW_NUMBER() OVER (PARTITION BY track_id ORDER BY date) AS rn
            FROM v2 WHERE chart = 'top200'
        ) WHERE rn = 1
    """).fetchdf()
    con.close()

    tracks['ground_truth'] = tracks['track_id'].map(cache)
    tracks = tracks[tracks['ground_truth'].notna()].reset_index(drop=True)

    # Stratified sample: pick from diverse languages
    random.seed(42)
    sampled_frames = []
    for lang, group in tracks.groupby('ground_truth'):
        n = max(1, int(N_SAMPLES * len(group) / len(tracks)))
        sampled_frames.append(group.sample(min(len(group), n), random_state=42))
    sampled = __import__('pandas').concat(sampled_frames).sample(frac=1, random_state=42).head(N_SAMPLES).reset_index(drop=True)
    return sampled


def detect_single(title, artist, origin):
    """Single-title detection (original method)."""
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SINGLE},
            {"role": "user", "content": f'"{title}" by {artist} (from {origin})'},
            {"role": "assistant", "content": "The ISO 639-1 code is: "},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 16},
    }, timeout=30)
    raw = resp.json()["message"]["content"].strip().lower().replace('*', '')
    for tok in raw.replace('.', ' ').replace(',', ' ').split():
        if len(tok) == 2 and tok.isalpha():
            return tok
    return None


def detect_batch(batch_rows):
    """Batch-of-4 detection."""
    lines = []
    for i, row in enumerate(batch_rows, 1):
        lines.append(f'{i}. "{row["title"]}" by {row["artist"]} (from {row["origin_country"]})')
    user_msg = "\n".join(lines)

    prefill_lines = "\n".join([f"{i}. " for i in range(1, len(batch_rows) + 1)])

    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_BATCH},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "1. "},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 64},
    }, timeout=60)
    raw = resp.json()["message"]["content"].strip().lower().replace('*', '')

    # Parse: expect one code per line
    results = []
    # Prepend "1. " prefix since it was in the prefill
    full_text = raw
    for line in full_text.split('\n'):
        line = line.strip().rstrip('.')
        # Extract 2-letter code from each line
        found = None
        for tok in line.replace('.', ' ').replace(',', ' ').split():
            tok = tok.strip()
            if len(tok) == 2 and tok.isalpha():
                found = tok
                break
        results.append(found)

    # Pad or trim to batch size
    while len(results) < len(batch_rows):
        results.append(None)
    return results[:len(batch_rows)]


def main():
    print("Loading test data...")
    df = load_test_data()
    print(f"Test set: {len(df)} titles across {df['ground_truth'].nunique()} languages\n")

    # Check Ollama is running
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    # ── Single-title baseline (skipped — use known values from prior run) ────
    # Prior run on 200 titles: 99.0% accuracy, 1897ms/title
    single_ms_per_title = 1897
    single_accuracy = 0.99
    single_time = single_ms_per_title * len(df) / 1000  # extrapolated
    single_match = int(single_accuracy * len(df))
    print(f"Single baseline (from prior run): ~{single_accuracy*100:.0f}% accuracy, {single_ms_per_title}ms/title\n")

    # ── Parallel single-title detection ──────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parallel_times = {}
    parallel_matches = {}
    for n_workers in [4, 8, 12]:
        print(f"{'='*60}")
        print(f"PARALLEL SINGLE-TITLE ({n_workers} workers)")
        print(f"{'='*60}")
        parallel_results = [None] * len(df)
        t0 = time.time()
        done_count = [0]

        def _detect_indexed(args):
            idx, row = args
            pred = detect_single(row['title'], row['artist'], row['origin_country'])
            return idx, pred

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_detect_indexed, (i, row)) for i, row in df.iterrows()]
            for fut in as_completed(futures):
                idx, pred = fut.result()
                parallel_results[idx] = pred
                done_count[0] += 1
                if done_count[0] % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"  {done_count[0]}/{len(df)} done ({elapsed:.1f}s, "
                          f"{elapsed/done_count[0]*1000:.0f}ms/title)")

        par_time = time.time() - t0
        df[f'parallel_{n_workers}_pred'] = parallel_results
        par_match = (df[f'parallel_{n_workers}_pred'] == df['ground_truth']).sum()
        parallel_times[n_workers] = par_time
        parallel_matches[n_workers] = par_match
        print(f"\nParallel({n_workers}): {par_match}/{len(df)} match ground truth "
              f"({par_match/len(df)*100:.1f}%) in {par_time:.1f}s "
              f"({par_time/len(df)*1000:.0f}ms/title)\n")

    # ── Batch-of-4 (skipped — use known values from prior run) ──────────────
    # Prior run on 200 titles: 80.0% accuracy, 753ms/title
    batch_ms_per_title = 753
    batch_accuracy = 0.80
    batch_time = batch_ms_per_title * len(df) / 1000
    batch_match = int(batch_accuracy * len(df))

    # ── Comparison ────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<22} {'Accuracy':>8} {'Time':>8} {'ms/title':>9} {'Speedup':>8}")
    print("-" * 60)
    print(f"{'Single (baseline)':<22} {single_match/len(df)*100:>7.1f}% {single_time:>7.1f}s {single_time/len(df)*1000:>8.0f} {'1.00x':>8}")
    for n_workers in [4, 8, 12]:
        pt = parallel_times[n_workers]
        pm = parallel_matches[n_workers]
        print(f"{'Parallel(' + str(n_workers) + ')':<22} {pm/len(df)*100:>7.1f}% {pt:>7.1f}s {pt/len(df)*1000:>8.0f} {single_time/pt:>7.2f}x")
    print(f"{'Batch-of-4':<22} {batch_match/len(df)*100:>7.1f}% {batch_time:>7.1f}s {batch_time/len(df)*1000:>8.0f} {single_time/batch_time:>7.2f}x")

    # Show disagreements for parallel methods
    for n_workers in [4, 8, 12]:
        col = f'parallel_{n_workers}_pred'
        disagree_par = df[df[col] != df['ground_truth']]
        if len(disagree_par) > 0:
            print(f"\nParallel({n_workers}) mismatches ({len(disagree_par)} total, first 10):")
            print(f"{'Title':<40} {'Artist':<25} {'Truth':<6} {'Pred':<6}")
            print("-" * 80)
            for _, r in disagree_par.head(10).iterrows():
                print(f"{r['title'][:39]:<40} {r['artist'][:24]:<25} "
                      f"{r['ground_truth']:<6} {r[col] or 'None':<6}")

    disagree = df.get('batch_pred', None)
    if disagree is not None:
        disagree = df[df['batch_pred'] != df['ground_truth']].head(20)
    if len(disagree) > 0:
        print(f"\nBatch mismatches (first {min(20, len(disagree))}):")
        print(f"{'Title':<40} {'Artist':<25} {'Truth':<6} {'Single':<7} {'Batch':<6}")
        print("-" * 90)
        for _, r in disagree.iterrows():
            print(f"{r['title'][:39]:<40} {r['artist'][:24]:<25} {r['ground_truth']:<6} "
                  f"{r['single_pred'] or 'None':<7} {r['batch_pred'] or 'None':<6}")


if __name__ == "__main__":
    main()
