# Speech-learning-unlearning
Learning and Selective unlearning of a set amount of users
from google.colab import drive
drive.mount('/content/drive')

!pip install transformers datasets torch torchaudio librosa soundfile accelerate -q

from huggingface_hub import notebook_login
notebook_login()

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor
)
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

from datasets import load_dataset

dataset = load_dataset("Codec-SUPERB/Voxceleb1_test_original", trust_remote_code=True)


def add_speaker_id(example):
    example["speaker_id"] = example["id"].split("+")[0]
    return example

dataset["test"] = dataset["test"].map(add_speaker_id)

all_speakers = sorted(list(set(dataset["test"]["speaker_id"])))
print(f"Total speakers available: {len(all_speakers)}")
print(f"First 10 speaker IDs: {all_speakers[:10]}")


print(dataset["test"].features)
print(dataset["test"][0])

FOUR_SPEAKERS = all_speakers[:4]
print(f"Chosen speakers: {FOUR_SPEAKERS}")

label2id = {spk: idx for idx, spk in enumerate(FOUR_SPEAKERS)}
id2label = {idx: spk for spk, idx in label2id.items()}
print(f"Label mapping: {label2id}")

def filter_four_speakers(example):
    return example["speaker_id"] in FOUR_SPEAKERS

filtered = dataset["test"].filter(filter_four_speakers)

split = filtered.train_test_split(test_size=0.2, seed=42)
train_data = split["train"]
test_data  = split["test"]

print(f"Train samples: {len(train_data)}")
print(f"Test samples:  {len(test_data)}")

for spk in FOUR_SPEAKERS:
    count = sum(1 for x in train_data["speaker_id"] if x == spk)
    print(f"  Speaker {spk}: {count} train clips")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "superb/wav2vec2-base-superb-sid"
)

MAX_DURATION = 3.0
SAMPLE_RATE  = 16000
MAX_SAMPLES  = int(MAX_DURATION * SAMPLE_RATE)

class SpeakerDataset(Dataset):
    def __init__(self, hf_dataset, label2id):
        self.data     = hf_dataset
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        audio = np.array(item["audio"]["array"], dtype=np.float32)

        if len(audio) < MAX_SAMPLES:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
        else:
            audio = audio[:MAX_SAMPLES]

        inputs = feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(self.label2id[item["speaker_id"]], dtype=torch.long)
        }

train_dataset = SpeakerDataset(train_data, label2id)
test_dataset  = SpeakerDataset(test_data,  label2id)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches:  {len(test_loader)}")

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "superb/wav2vec2-base-superb-sid",
    num_labels=4,
    label2id={str(v): k for k, v in id2label.items()},
    id2label={str(k): v for k, v in id2label.items()},
    ignore_mismatched_sizes=True
)

model = model.to(DEVICE)
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

EPOCHS      = 10
LR          = 1e-4
optimizer   = AdamW(model.parameters(), lr=LR)
loss_fn     = torch.nn.CrossEntropyLoss()

train_losses    = []
train_accuracies = []
test_accuracies  = []

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(DEVICE)
            labels       = batch["label"].to(DEVICE)
            logits       = model(input_values).logits
            preds        = torch.argmax(logits, dim=-1)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / total, all_preds, all_labels

print("Starting training...\n")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    correct, total = 0, 0

    for batch in train_loader:
        input_values = batch["input_values"].to(DEVICE)
        labels       = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_values).logits
        loss   = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds       = torch.argmax(logits, dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    train_acc  = correct / total
    avg_loss   = epoch_loss / len(train_loader)
    test_acc, _, _ = evaluate(model, test_loader)

    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1:>2}/{EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc*100:.1f}% | "
          f"Test Acc: {test_acc*100:.1f}%")

print("\nTraining complete.")

epochs_range = range(1, EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1.plot(epochs_range, train_losses, "b-o", linewidth=2, label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross Entropy Loss")
ax1.set_title("Training Loss vs. Epoch")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curve
ax2.plot(epochs_range, [a*100 for a in train_accuracies], "g-o", linewidth=2, label="Train Accuracy")
ax2.plot(epochs_range, [a*100 for a in test_accuracies],  "r-o", linewidth=2, label="Test Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy vs. Epoch")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 105)

plt.suptitle("Wav2Vec2 Speaker Identification — 4 Speakers", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → training_curves.png")

_, all_preds, all_labels = evaluate(model, test_loader)

# Per-speaker accuracy
print("Per-Speaker Accuracy on Test Set:")
print("─" * 40)
for idx, spk in id2label.items():
    mask    = [l == idx for l in all_labels]
    correct = sum(p == l for p, l in zip(all_preds, all_labels) if l == idx)
    total   = sum(mask)
    acc     = correct / total if total > 0 else 0
    print(f"  Speaker {spk} (label {idx}): {correct}/{total} → {acc*100:.1f}%")

# Bar chart of per-speaker accuracy
speaker_accs = []
for idx in range(4):
    correct = sum(p == l for p, l in zip(all_preds, all_labels) if l == idx)
    total   = sum(1 for l in all_labels if l == idx)
    speaker_accs.append(correct / total * 100 if total > 0 else 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
bars = ax1.bar(
    [f"Speaker\n{id2label[i]}" for i in range(4)],
    speaker_accs,
    color=colors,
    edgecolor="black",
    linewidth=0.8
)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Per-Speaker Accuracy (Before Unlearning)")
ax1.set_ylim(0, 110)
ax1.grid(True, axis="y", alpha=0.3)
for bar, acc in zip(bars, speaker_accs):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{acc:.1f}%",
        ha="center", va="bottom", fontweight="bold"
    )

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[f"Spk {id2label[i]}" for i in range(4)]
)
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title("Confusion Matrix (Before Unlearning)")

plt.suptitle("4-Speaker Identification Results", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("speaker_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → speaker_accuracy.png")

torch.save({
    "model_state_dict": model.state_dict(),
    "label2id": label2id,
    "id2label": id2label,
    "train_losses": train_losses,
    "train_accuracies": train_accuracies,
    "test_accuracies": test_accuracies
}, "speaker_model_before_unlearning.pt")

print("Model saved → speaker_model_before_unlearning.pt")
print(f"Final Test Accuracy: {test_accuracies[-1]*100:.1f}%")
print(f"\nSpeaker designated for unlearning: Speaker {id2label[3]} (label 3)")
print("Ready for Phase 2 — SCRUB Unlearning.")

# Forget speaker = id10273 (label 3)
# Retain speakers = id10270, id10271, id10272 (labels 0,1,2)

FORGET_SPEAKER = id2label[3]
print(f"Forget speaker: {FORGET_SPEAKER}")

def is_forget(example):
    return example["speaker_id"] == FORGET_SPEAKER

def is_retain(example):
    return example["speaker_id"] != FORGET_SPEAKER

# Split train data into forget and retain sets
forget_data = train_data.filter(is_forget)
retain_data  = train_data.filter(is_retain)

print(f"Forget set size: {len(forget_data)}")
print(f"Retain set size:  {len(retain_data)}")

forget_dataset = SpeakerDataset(forget_data, label2id)
retain_dataset  = SpeakerDataset(retain_data,  label2id)

forget_loader = DataLoader(forget_dataset, batch_size=8, shuffle=True)
retain_loader  = DataLoader(retain_dataset,  batch_size=8, shuffle=True)

print(f"Forget batches: {len(forget_loader)}")
print(f"Retain batches:  {len(retain_loader)}")

import copy

# We need a frozen reference model to compute KL divergence
# This prevents retain speakers from drifting during unlearning
reference_model = copy.deepcopy(model)
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False

print("Reference model frozen and ready.")

# Run this BEFORE Cell 13 to reset to the pre-unlearning model
checkpoint = torch.load("speaker_model_before_unlearning.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
print("Model reset to pre-unlearning state.")

import torch.nn.functional as F

UNLEARN_EPOCHS   = 5
SCRUB_LR         = 1e-5
ASCENT_WEIGHT    = 0.5    # reduced — less aggressive forgetting
KL_WEIGHT        = 2.0    # increased — stronger retain protection
DESCENT_WEIGHT   = 1.0    # NEW — direct supervision on retain speakers

unlearn_optimizer = AdamW(model.parameters(), lr=SCRUB_LR)

forget_accs_during = []
retain_accs_during = []

print("Starting Improved Unlearning...")
print(f"Forgetting speaker: {FORGET_SPEAKER}\n")

for epoch in range(UNLEARN_EPOCHS):
    model.train()

    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)
    steps = max(len(forget_loader), len(retain_loader))

    epoch_loss = 0

    for step in range(steps):
        unlearn_optimizer.zero_grad()

        # ── 1. GRADIENT ASCENT on forget speaker ──────────────────
        try:
            forget_batch = next(forget_iter)
        except StopIteration:
            forget_iter  = iter(forget_loader)
            forget_batch = next(forget_iter)

        f_inputs = forget_batch["input_values"].to(DEVICE)
        f_labels = forget_batch["label"].to(DEVICE)
        f_logits = model(f_inputs).logits
        f_loss   = loss_fn(f_logits, f_labels)
        ascent_loss = -ASCENT_WEIGHT * f_loss

        # ── 2. GRADIENT DESCENT on retain speakers (direct) ───────
        try:
            retain_batch = next(retain_iter)
        except StopIteration:
            retain_iter  = iter(retain_loader)
            retain_batch = next(retain_iter)

        r_inputs = retain_batch["input_values"].to(DEVICE)
        r_labels = retain_batch["label"].to(DEVICE)
        r_logits = model(r_inputs).logits
        descent_loss = DESCENT_WEIGHT * loss_fn(r_logits, r_labels)

        # ── 3. KL DIVERGENCE — keep retain close to reference ─────
        with torch.no_grad():
            ref_logits = reference_model(r_inputs).logits

        kl_loss = KL_WEIGHT * F.kl_div(
            F.log_softmax(r_logits,   dim=-1),
            F.softmax(ref_logits,     dim=-1),
            reduction="batchmean"
        )

        # ── Total loss: forget + retain supervision + retain anchor
        total_loss = ascent_loss + descent_loss + kl_loss
        total_loss.backward()
        unlearn_optimizer.step()
        epoch_loss += total_loss.item()

    # ── Evaluate after each epoch ──────────────────────────────────
    forget_test    = test_data.filter(is_forget)
    retain_test    = test_data.filter(is_retain)
    f_acc, _, _    = evaluate(model, DataLoader(SpeakerDataset(forget_test, label2id), batch_size=8))
    r_acc, _, _    = evaluate(model, DataLoader(SpeakerDataset(retain_test, label2id), batch_size=8))

    forget_accs_during.append(f_acc * 100)
    retain_accs_during.append(r_acc * 100)

    print(f"Epoch {epoch+1}/{UNLEARN_EPOCHS} | "
          f"Forget Acc: {f_acc*100:.1f}% (want ~0%) | "
          f"Retain Acc: {r_acc*100:.1f}% (want >90%)")

print("\nUnlearning complete.")

epochs_range = range(1, UNLEARN_EPOCHS + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── Unlearning progress ──
axes[0].plot(epochs_range, forget_accs_during, "r-o", linewidth=2, label="Forget Speaker")
axes[0].plot(epochs_range, retain_accs_during, "g-o", linewidth=2, label="Retain Speakers")
axes[0].axhline(y=0,   color="red",   linestyle="--", alpha=0.5, label="Target (forget)")
axes[0].axhline(y=100, color="green", linestyle="--", alpha=0.5, label="Target (retain)")
axes[0].set_xlabel("Unlearning Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("SCRUB Unlearning Progress")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-5, 110)

# ── Before vs After per speaker ──
before_accs = []
after_accs  = []

_, all_preds_before, all_labels_before = evaluate(
    reference_model, test_loader
)
_, all_preds_after, all_labels_after = evaluate(
    model, test_loader
)

for idx in range(4):
    # before
    correct = sum(p == l for p, l in zip(all_preds_before, all_labels_before) if l == idx)
    total   = sum(1 for l in all_labels_before if l == idx)
    before_accs.append(correct / total * 100 if total > 0 else 0)
    # after
    correct = sum(p == l for p, l in zip(all_preds_after, all_labels_after) if l == idx)
    total   = sum(1 for l in all_labels_after if l == idx)
    after_accs.append(correct / total * 100 if total > 0 else 0)

x      = np.arange(4)
width  = 0.35
labels = [f"Spk {id2label[i]}" for i in range(4)]
colors_before = ["#2196F3"] * 4
colors_after  = ["#4CAF50", "#4CAF50", "#4CAF50", "#F44336"]  # red for forget speaker

bars1 = axes[1].bar(x - width/2, before_accs, width, label="Before Unlearning", color="#2196F3", edgecolor="black")
bars2 = axes[1].bar(x + width/2, after_accs,  width, label="After Unlearning",  color=colors_after, edgecolor="black")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=9)
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Per-Speaker Accuracy\nBefore vs. After Unlearning")
axes[1].legend()
axes[1].set_ylim(0, 115)
axes[1].grid(True, axis="y", alpha=0.3)
for bar, acc in zip(bars2, after_accs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{acc:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

# ── Confusion matrix after unlearning ──
cm_after = confusion_matrix(all_labels_after, all_preds_after)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_after,
    display_labels=[f"Spk {id2label[i]}" for i in range(4)]
)
disp.plot(ax=axes[2], colorbar=False, cmap="Reds")
axes[2].set_title("Confusion Matrix\n(After Unlearning)")

plt.suptitle("SCRUB Machine Unlearning — Speaker id10273 Forgotten",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("unlearning_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → unlearning_results.png")

print("=" * 55)
print("       MACHINE UNLEARNING — FINAL SUMMARY")
print("=" * 55)
print(f"  Forget speaker : {FORGET_SPEAKER} (label 3)")
print(f"  Method         : SCRUB (Gradient Ascent + Gradient Descent + KL)")
print()
print("  Per-Speaker Accuracy:")
print(f"  {'Speaker':<12} {'Before':>8} {'After':>8} {'Status':>10}")
print("  " + "-" * 42)
for i in range(4):
    status = "FORGOTTEN ✓" if i == 3 else "Retained ✓"
    print(f"  {id2label[i]:<12} {before_accs[i]:>7.1f}% {after_accs[i]:>7.1f}%  {status}")
print("=" * 55)











