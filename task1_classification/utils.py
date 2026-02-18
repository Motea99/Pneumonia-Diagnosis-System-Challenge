import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =====================================
# 1️⃣ تثبيت العشوائية (لإعادة إنتاج النتائج)
# =====================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================
# 2️⃣ اختيار الجهاز
# =====================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================
# 3️⃣ حساب المقاييس
# =====================================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


# =====================================
# 4️⃣ طباعة النتائج بشكل منظم
# =====================================
def print_metrics(metrics_dict):
    print("\n📊 Evaluation Metrics")
    print("-" * 30)
    for key, value in metrics_dict.items():
        print(f"{key.capitalize()}: {value:.4f}")
    print("-" * 30)


# =====================================
# 5️⃣ حفظ المودل
# =====================================
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved at {path}")


# =====================================
# 6️⃣ تحميل المودل
# =====================================
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    return model


# =====================================
# 7️⃣ Early Stopping
# =====================================
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):

        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss > self.best_loss:
            self.counter += 1
            print(f"⚠️ EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0
