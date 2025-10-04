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
# 2. Feature Extraction Function
# -----------------------------
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return None

def extract_features_from_array(img_array):
    """Helper: extract features directly from numpy image array"""
    try:
        img = cv2.resize(img_array, (224, 224))
        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"❌ Error processing rotated image: {e}")
        return None

# -----------------------------
# 3. Load Fabric Database (10 known fabrics)
# -----------------------------
fabric_dir = "data/"
fabric_embeddings = {}

print("🔎 Scanning data folder...")
if not os.path.exists(fabric_dir):
    print(f"❌ Folder not found: {fabric_dir}")
else:
    for file in os.listdir(fabric_dir):
        path = os.path.join(fabric_dir, file)
        print(f"Found file: {file}")
        if path.lower().endswith((".jpg", ".png", ".jpeg")):
            features = extract_features(path)
            if features is not None:
                fabric_embeddings[file] = features
                print(f"   ✔ Extracted features for {file}")
            else:
                print(f"   ⚠ Skipped {file} (no features)")
        else:
            print(f"   ⚠ Skipped {file} (not an image)")

if len(fabric_embeddings) == 0:
    print("❌ No valid images found in data/ folder. Exiting.")
    exit()

print(f"\n✅ Total fabrics loaded: {len(fabric_embeddings)}\n")

# -----------------------------
# 4. Load Test Image (auto-detect)
# -----------------------------
test_dir = "test/"
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

if len(test_files) == 0:
    print("❌ No test images found in test/ folder")
    exit()

test_img_path = os.path.join(test_dir, test_files[0])
print(f"🔎 Using test image: {test_img_path}")

# Load test image (OpenCV for rotation)
test_img_cv = cv2.imread(test_img_path)
if test_img_cv is None:
    print("❌ Failed to read test image. Exiting.")
    exit()
print("✅ Test image loaded\n")

# -----------------------------
# 5. Generate Rotated Versions of Test Image
# -----------------------------
def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

rotated_features = []
angles = range(-30, 31, 10)  # -30, -20, -10, 0, 10, 20, 30

print("🔄 Extracting features for rotated test images...")
for angle in angles:
    rotated_img = rotate_image(test_img_cv, angle)
    feats = extract_features_from_array(rotated_img)
    if feats is not None:
        rotated_features.append((angle, feats))
        print(f"   ✔ Angle {angle}° processed")
    else:
        print(f"   ⚠ Skipped angle {angle}°")

if len(rotated_features) == 0:
    print("❌ Could not extract features for any rotation. Exiting.")
    exit()

# -----------------------------
# 6. Compute Similarities
# -----------------------------
print("\n🔄 Calculating similarities...")
similarities = {}

for fname, feat in fabric_embeddings.items():
    max_sim = -1
    best_angle = 0
    for angle, test_feat in rotated_features:
        sim = cosine_similarity([test_feat], [feat])[0][0]
        if sim > max_sim:
            max_sim = sim
            best_angle = angle
    similarities[fname] = (max_sim, best_angle)
    print(f"   → {fname}: best {max_sim:.4f} at {best_angle}° rotation")

# -----------------------------
# 7. Find Best Match
# -----------------------------
best_match = max(similarities, key=lambda k: similarities[k][0])
best_score, best_angle = similarities[best_match]

print("\n🎯 Best Match:")
print(f"   Fabric: {best_match}")
print(f"   Similarity Score: {best_score:.4f}")
print(f"   Best Rotation Angle: {best_angle}°")
