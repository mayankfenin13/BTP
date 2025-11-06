import argparse, os, json, glob
import matplotlib.pyplot as plt

def plot_unlearn_summary(run_dir, title=None):
    p = os.path.join(run_dir,"metrics","unlearn_summary.json")
    if not os.path.exists(p):
        print("No metrics yet:", p); return
    with open(p) as f:
        m = json.load(f)
    # bar chart speed-up
    fig = plt.figure()
    keys = [k for k in m.keys() if "time_s" in k]
    vals = [m[k] for k in keys]
    plt.bar(keys, vals)
    plt.ylabel("Seconds")
    plt.title(title or "Retraining time")
    out = os.path.join(run_dir,"plots")
    os.makedirs(out, exist_ok=True)
    plt.savefig(os.path.join(out,"retrain_time.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    # accuracy pair if present
    acc_keys = [k for k in m.keys() if "acc" in k]
    if acc_keys:
        fig = plt.figure()
        vals = [m[k] for k in acc_keys]
        plt.bar(acc_keys, vals)
        plt.ylabel("Accuracy")
        plt.title("Accuracy after unlearning")
        plt.savefig(os.path.join(out,"accuracy.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    plot_unlearn_summary(args.run_dir, args.title)
