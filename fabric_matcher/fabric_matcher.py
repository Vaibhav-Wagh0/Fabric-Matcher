import os
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Load Pretrained VGG16 Model
# -----------------------------
print("🔄 Loading VGG16 model...")
model = VGG16(weights="imagenet", include_top=False, pooling="avg")
print("✅ VGG16 model loaded successfully\n")

# -----------------------------
# 2. Feature Extraction
# -----------------------------
def extract_features_from_path(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        arr = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
        return model.predict(arr, verbose=0).flatten()
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return None

def extract_features_from_array(img_array):
    try:
        img = cv2.resize(img_array, (224, 224))
        arr = preprocess_input(np.expand_dims(img, axis=0))
        return model.predict(arr, verbose=0).flatten()
    except Exception as e:
        print(f"❌ Error processing rotated image: {e}")
        return None

# -----------------------------
# 3. Load Fabric Database
# -----------------------------
fabric_dir = "data/"
fabric_embeddings = {}
print("🔎 Scanning data folder...")

if not os.path.exists(fabric_dir):
    print(f"❌ Folder not found: {fabric_dir}")
    exit()

for file in os.listdir(fabric_dir):
    path = os.path.join(fabric_dir, file)
    print(f"Found file: {file}")
    if path.lower().endswith((".jpg", ".png", ".jpeg")):
        feats = extract_features_from_path(path)
        if feats is not None:
            fabric_embeddings[file] = feats
            print(f"   ✔ Extracted features for {file}")
        else:
            print(f"   ⚠ Skipped {file} (no features)")
    else:
        print(f"   ⚠ Skipped {file} (not an image)")

if not fabric_embeddings:
    print("❌ No valid images found in data/ folder. Exiting.")
    exit()

print(f"\n✅ Total fabrics loaded: {len(fabric_embeddings)}\n")

# -----------------------------
# 4. Load Test Images
# -----------------------------
test_dir = "test/"
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

if not test_files:
    print("❌ No test images found in test/ folder")
    exit()

print(f"🔎 Found {len(test_files)} test images in {test_dir}\n")

# -----------------------------
# Helper: Rotate Image
# -----------------------------
def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# -----------------------------
# Helper: Compute Best Match for One Test Image
# -----------------------------
def process_test_image(test_img_path):
    print(f"🔎 Using test image: {test_img_path}")
    img_cv = cv2.imread(test_img_path)
    if img_cv is None:
        print("❌ Failed to read test image. Skipping.\n")
        return

    print("✅ Test image loaded\n")
    angles = range(-30, 31, 10)
    rotated_features = [(angle, extract_features_from_array(rotate_image(img_cv, angle))) 
                        for angle in angles]
    rotated_features = [(a,f) for a,f in rotated_features if f is not None]

    if not rotated_features:
        print("❌ Could not extract features for any rotation. Skipping.\n")
        return

    print("🔄 Calculating similarities...")
    similarities = {}
    for fname, feat in fabric_embeddings.items():
        max_sim, best_angle = max(
            ((cosine_similarity([f], [feat])[0][0], a) for a,f in rotated_features),
            key=lambda x: x[0]
        )
        similarities[fname] = (max_sim, best_angle)
        print(f"   → {fname}: best {max_sim:.4f} at {best_angle}° rotation")

    best_match = max(similarities, key=lambda k: similarities[k][0])
    best_score, best_angle = similarities[best_match]

    print("\n🎯 Best Match:")
    print(f"   Fabric: {best_match}")
    print(f"   Similarity Score: {best_score:.4f}")
    print(f"   Best Rotation Angle: {best_angle}°\n")
    print("-------------------------------------------------------------\n")

# -----------------------------
# 5. Process All Test Images
# -----------------------------
for test_file in test_files:
    process_test_image(os.path.join(test_dir, test_file))
