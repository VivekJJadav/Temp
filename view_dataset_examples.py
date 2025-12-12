import pickle
import os

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed"

FILES = {
    "train_normal.pkl":      "train_normal.txt",
    "train_dedup.pkl":       "train_dedup.txt",
    "train_duplicated.pkl":  "train_duplicated.txt",
    "val.pkl":               "val.txt",
}

def export_pkl_to_txt(pkl_path, txt_path):
    print(f"Loading {pkl_path}...")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    raw = data.get("raw", [])
    labels = data.get("labels", [])

    if not raw or not labels:
        print(f"⚠️  Warning: No raw or labels in {pkl_path}")
        return

    print(f"Writing {len(raw):,} lines to {txt_path}...")

    with open(txt_path, "w", encoding="utf-8") as out:
        for t, lab in zip(raw, labels):
            out.write(f"[{lab}] {t}\n")

    print(f"✔ Done: {txt_path}")

def main():
    for pkl_name, txt_name in FILES.items():
        pkl_path = os.path.join(INPUT_DIR, pkl_name)
        txt_path = os.path.join(OUTPUT_DIR, txt_name)

        if not os.path.exists(pkl_path):
            print(f"❌ Missing: {pkl_path}")
            continue

        export_pkl_to_txt(pkl_path, txt_path)

    print("\nAll exports completed.")

if __name__ == "__main__":
    main()
