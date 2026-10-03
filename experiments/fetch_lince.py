"""
Fetch LinCE MSA-Egyptian Arabic data from the Hugging Face parquet mirror
(the original ritual.uh.edu server is unreachable) and print statistics.

Usage (from the project folder, environment active):
    python fetch_lince.py

Output: data/raw/lince_msaea/{lid_msaea,ner_msaea}/{train,validation,test}.parquet
        + a CoNLL copy of each split (token<TAB>label) for easy reading.
"""
import os
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]  # repository root
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface"))
OUT = PROJECT_DIR / "data" / "raw" / "lince_msaea"

import pandas as pd  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

CONFIGS = ["lid_msaea", "ner_msaea"]


def to_list(x):
    return list(x) if x is not None else []


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = snapshot_download(
        repo_id="lince-benchmark/lince",
        repo_type="dataset",
        revision="refs/convert/parquet",
        allow_patterns=[f"{c}/*" for c in CONFIGS],
        local_dir=str(OUT / "_parquet"),
    )

    for cfg in CONFIGS:
        print("\n" + "=" * 70 + f"\n{cfg}\n" + "=" * 70)
        cfg_dir = Path(tmp) / cfg
        if not cfg_dir.exists():
            print("  not found in mirror")
            continue
        (OUT / cfg).mkdir(exist_ok=True)
        for split in ["train", "validation", "test"]:
            files = sorted((cfg_dir / split).glob("*.parquet"))
            if not files:
                continue
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            df.to_parquet(OUT / cfg / f"{split}.parquet")
            label_col = "lid" if "lid" in df.columns else "ner"
            print(f"\n[{split}] rows={len(df)} columns={list(df.columns)}")

            # CoNLL copy
            with open(OUT / cfg / f"{split}.conll", "w", encoding="utf-8") as fh:
                for _, r in df.iterrows():
                    words, labs = to_list(r["words"]), to_list(r.get(label_col))
                    labs = labs if len(labs) == len(words) else [""] * len(words)
                    for w, l in zip(words, labs):
                        fh.write(f"{w}\t{l}\n")
                    fh.write("\n")

            labels = Counter(l for x in df[label_col] for l in to_list(x))
            print("  label counts:", dict(labels.most_common()))
            if cfg == "lid_msaea":
                cs = switches = 0
                example = None
                for _, r in df.iterrows():
                    labs = to_list(r["lid"])
                    langs = [l for l in labs if l in ("lang1", "lang2")]
                    sw = sum(1 for a, b in zip(langs, langs[1:]) if a != b)
                    if {"lang1", "lang2"} <= set(langs):
                        cs += 1
                        if example is None and 8 <= len(labs) <= 20:
                            example = list(zip(to_list(r["words"]), labs))
                    switches += sw
                print(f"  code-switched sentences={cs} ({cs/len(df):.1%})  switch-points={switches}")
                if example:
                    print("  example:", " ".join(f"{w}/{l}" for w, l in example))

    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
