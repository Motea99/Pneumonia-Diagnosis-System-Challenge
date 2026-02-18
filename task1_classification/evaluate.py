import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================
# 1. التحقق من البيئة والمتطلبات الأساسية
# ============================================

def check_requirements():
    """التحقق من وجود جميع المتطلبات"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    print(f"✅ PyTorch version: {torch.__version__}")

    # التحقق من وجود model و test_loader
    global model, test_loader
    try:
        print(f"✅ Model type: {type(model).__name__}")
        print(f"✅ Test loader batches: {len(test_loader)}")
    except NameError:
        print("❌ Error: model or test_loader not defined!")
        return False

    return True

# ============================================
# 2. دالة لفحص وتعديل حجم الصور
# ============================================

def get_image_size_from_loader(test_loader):
    """فحص حجم الصور في test_loader"""
    for images, _ in test_loader:
        return images.shape[2:]  # (height, width)
    return None

def fix_image_size(image, target_size=(224, 224)):
    """تعديل حجم الصورة إلى الحجم المطلوب"""
    if image.shape[1:] != target_size:
        resize = transforms.Resize(target_size)
        return resize(image)
    return image

# ============================================
# 3. الدالة الرئيسية للتقييم (بدون تكرار الأرقام)
# ============================================

def evaluate_model_safe(model, test_loader):
    """
    دالة تقييم متكاملة مع معالجة تلقائية لأحجام الصور
    """
    # إعداد الجهاز
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # التحقق من حجم الصور
    sample_shape = None
    for images, _ in test_loader:
        sample_shape = images.shape
        print(f"📊 Detected image size: {sample_shape[2]}x{sample_shape[3]}")
        break

    # الحجم المتوقع للنموذج
    expected_size = 224  # ViT expects 224x224

    # إذا كان الحجم غير مناسب، نقوم بتعديله تلقائياً
    if sample_shape and (sample_shape[2] != expected_size or sample_shape[3] != expected_size):
        print(f"⚠️  Image size mismatch: Got {sample_shape[2]}x{sample_shape[3]}, expected {expected_size}x{expected_size}")
        print("🔄 Automatically resizing images during evaluation...")

        # إضافة resize transform
        resize_transform = transforms.Resize((expected_size, expected_size))

        # تعديل test_loader مؤقتاً
        original_collate = test_loader.collate_fn

        def collate_with_resize(batch):
            images = []
            labels = []
            for img, label in batch:
                if img.shape[1:] != (expected_size, expected_size):
                    img = resize_transform(img)
                images.append(img)
                labels.append(label)
            return torch.stack(images), torch.tensor(labels)

        test_loader.collate_fn = collate_with_resize
        print("✅ Automatic resizing enabled!")

    model.eval()
    all_preds = []
    all_labels = []

    print("\n🚀 Starting evaluation process...")
    print(f"📦 Number of batches: {len(test_loader)}")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            try:
                images = images.to(device)

                # التحقق من حجم الصور قبل الإدخال
                if images.shape[2:] != (expected_size, expected_size):
                    print(f"⚠️  Batch {batch_idx}: Unexpected shape {images.shape}, resizing...")
                    images = torch.stack([fix_image_size(img, (expected_size, expected_size)) for img in images])
                    images = images.to(device)

                # Forward pass
                outputs = model(images)

                # Handle Hugging Face output format
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                # Get predictions
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Progress update
                if (batch_idx + 1) % max(1, len(test_loader)//5) == 0:
                    print(f"✅ Processed {batch_idx + 1}/{len(test_loader)} batches")

            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                continue

    # تحويل إلى numpy
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()

    print(f"\n📊 Total samples evaluated: {len(all_labels)}")
    print(f"📊 Class distribution: {np.unique(all_labels, return_counts=True)}")

    # حساب المقاييس
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # التقرير الكامل
    target_names = ['Normal', 'Pneumonia']
    report = classification_report(all_labels, all_preds, target_names=target_names)

    # طباعة النتائج
    print("\n" + "="*50)
    print(f"🎯 OVERALL ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print("="*50)
    print("\n📋 CLASSIFICATION REPORT:")
    print(report)

    print("\n📊 CONFUSION MATRIX:")
    print(cm)

    # ============================================
    # رسم Confusion Matrix بشكل صحيح (بدون تكرار)
    # ============================================
    plt.figure(figsize=(12, 10))

    # الطريقة الأولى: استخدام annot=True فقط (بدون إضافة نصوص يدوية)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=target_names,
                     yticklabels=target_names,
                     annot_kws={'size': 16, 'weight': 'bold'},
                     cbar_kws={'label': 'Count'})

    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix - ViT Model', fontsize=16, fontweight='bold')

    # إضافة النسب المئوية في خانة منفصلة (اختياري)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                percentage = cm[i, j] / np.sum(cm[i, :]) * 100
                # إضافة النسبة المئوية أسفل الرقم الرئيسي
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                       ha='center', va='center', color='black', fontsize=10)

    plt.tight_layout()
    plt.show()

    # إحصائيات إضافية
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()

        print("\n📈 DETAILED METRICS:")
        print(f"   • True Positives: {tp}")
        print(f"   • True Negatives: {tn}")
        print(f"   • False Positives: {fp}")
        print(f"   • False Negatives: {fn}")

        if tp + fn > 0:
            sensitivity = tp / (tp + fn)
            print(f"   • Sensitivity (Recall): {sensitivity:.4f}")

        if tn + fp > 0:
            specificity = tn / (tn + fp)
            print(f"   • Specificity: {specificity:.4f}")

        # Precision and F1-score
        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"   • Precision: {precision:.4f}")

        if precision + sensitivity > 0:
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
            print(f"   • F1-Score: {f1:.4f}")

    return acc, cm

# ============================================
# 4. التحقق من النموذج والبيانات
# ============================================

def validate_model_and_data():
    """التحقق من صحة النموذج والبيانات قبل التقييم"""

    # التأكد من وجود model
    try:
        model
    except NameError:
        print("❌ ERROR: 'model' is not defined!")
        print("Please define your model first.")
        return False

    # التأكد من وجود test_loader
    try:
        test_loader
    except NameError:
        print("❌ ERROR: 'test_loader' is not defined!")
        print("Please define your test_loader first.")
        return False

    # التأكد من أن النموذج في وضع التقييم
    model.eval()

    # فحص عينة من البيانات
    try:
        for images, labels in test_loader:
            print(f"✅ Sample batch - Images: {images.shape}, Labels: {labels.shape}")
            print(f"✅ Label values: {torch.unique(labels)}")
            break
    except Exception as e:
        print(f"❌ Error accessing test_loader: {e}")
        return False

    return True

# ============================================
# 5. التنفيذ النهائي
# ============================================

print("="*60)
print("🔍 VIT MODEL EVALUATION SYSTEM")
print("="*60)

# التحقق من المتطلبات
if validate_model_and_data():
    print("\n✅ Validation passed. Starting evaluation...\n")

    # تنفيذ التقييم
    try:
        acc, cm = evaluate_model_safe(model, test_loader)
        print("\n✨ Evaluation completed successfully!")
    except Exception as e:
        print(f"\n❌ Unexpected error during evaluation: {e}")
        print("\n🛠️  Troubleshooting tips:")
        print("   1. Check if model is correctly loaded")
        print("   2. Verify test_loader contains valid data")
        print("   3. Ensure GPU memory is sufficient")
        print("   4. Try reducing batch size if memory issues")
else:
    print("\n❌ Validation failed. Please check the errors above.")
