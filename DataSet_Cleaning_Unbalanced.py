import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

# === 1. Load original dataset ===
df = pd.read_csv("FULL_ALLIN_MERGED.csv")

# === 2. Convert "Time" to seconds BEFORE dropping it ===
df['Duration_sec'] = pd.to_timedelta("00:" + df['Time']).dt.total_seconds()

# === 3. Drop unnecessary columns ===
drop_cols = ['Song', 'Artist', 'Popularity', 'Album', 'Album Date', 'Added At', 'Spotify Track Id', 'ISRC', 'Time']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
df = df.dropna()

# === 4. Assign expanded genre classes ===
def assign_genre_group(genres):
    g = str(genres).lower()
    if any(x in g for x in ["hard techno", "acid techno", "industrial", "tekno", "rave", "warehouse", "schranz"]):
        return "Hard Techno"
    elif any(x in g for x in ["melodic techno", "progressive", "deep", "emotional", "organic", "tech house", "afro", "indie", "house"]):
        return "Melodic Techno"
    elif any(x in g for x in ["trance", "psytrance", "goa"]):
        return "Trance"
    elif any(x in g for x in ["dark techno", "edm"]):
        return "Industrial/Dark"
    else:
        return "Other"

df['Genre'] = df['Genres'].apply(assign_genre_group)

# === Genre distribution check ===
print("🎵 Genre distribution (UNBALANCED):")
print(df['Genre'].value_counts())

# === Label encode genre ===
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['Genre'])
joblib.dump(le, 'electrinity_labelencoder_unbalanced.pkl')

# === 5. Encode Key and Camelot ===
key_encoder = LabelEncoder()
camelot_encoder = LabelEncoder()
df['Key_encoded'] = key_encoder.fit_transform(df['Key'].astype(str))
df['Camelot_encoded'] = camelot_encoder.fit_transform(df['Camelot'].astype(str))
joblib.dump(key_encoder, 'electrinity_key_encoder_unbalanced.pkl')
joblib.dump(camelot_encoder, 'electrinity_camelot_encoder_unbalanced.pkl')

# === 6. Create subgenre binary features ===
subgenres = ['tekno', 'acid', 'industrial', 'rave', 'warehouse', 'minimal', 'trance', 'house', 'bass', 'dark']
for sub in subgenres:
    df[f'Has_{sub.capitalize()}'] = df['Genres'].str.contains(sub, case=False, na=False).astype(int)

# === 6.5 Remove genres with <2 samples to allow stratified split ===
genre_counts = df['Genre'].value_counts()
valid_genres = genre_counts[genre_counts >= 2].index
df = df[df['Genre'].isin(valid_genres)]

# === 7. Select features ===
selected_features = [
    'BPM', 'Dance', 'Energy', 'Acoustic', 'Instrumental',
    'Happy', 'Speech', 'Live', 'Loud (Db)', 'Duration_sec',
    'Key_encoded', 'Camelot_encoded'
] + [f'Has_{sub.capitalize()}' for sub in subgenres]

# === ✅ SKIP Step 8: Do NOT downsample ===

# 💾 Save the unbalanced dataset
df.to_csv("Unbalanced_FULL_ALLIN_MERGED.csv", index=False)
print("📂 Saved: 'Unbalanced_FULL_ALLIN_MERGED.csv'")

# === 9. Normalize ===
scaler = MinMaxScaler()
df[selected_features] = scaler.fit_transform(df[selected_features])
joblib.dump(scaler, 'electrinity_scaler_unbalanced.pkl')

# === 10. Prepare training data ===
X = df[selected_features]
y = df['genre_encoded']
joblib.dump(X.columns.tolist(), 'electrinity_features_unbalanced.pkl')

# === 11. Train RandomForest (Main Model) ===
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
joblib.dump(rf, 'electrinity_model_rf_unbalanced.pkl')

# === 12. Train SGD (Optional fallback) ===
classes = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))

sgd = SGDClassifier(loss='log_loss', class_weight=class_weight_dict, random_state=42)
sgd.partial_fit(X, y, classes=classes)
joblib.dump(sgd, 'electrinity_model_sgd_unbalanced.pkl')

# === 13. Save feature means ===
feature_means = X.mean().to_dict()
joblib.dump(feature_means, 'electrinity_feature_means_unbalanced.pkl')

# === 14. Evaluate both models ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"📊 Total samples: {len(X)}")
print(f"🧪 Test set size: {len(X_test)} samples ({len(X_test)/len(X)*100:.2f}%)")
print(f"🧠 Train set size: {len(X_train)} samples ({len(X_train)/len(X)*100:.2f}%)")


print("\n🎯 Testing RandomForest (Unbalanced):")
y_pred_rf = rf.predict(X_test)
labels_in_test = np.unique(y_test)
print(classification_report(y_test, y_pred_rf, labels=labels_in_test, target_names=le.inverse_transform(labels_in_test)))
print("Confusion Matrix (RF):")
print(confusion_matrix(y_test, y_pred_rf, labels=labels_in_test))

print("\n🧠 Testing SGDClassifier (Unbalanced):")
y_pred_sgd = sgd.predict(X_test)
print(classification_report(y_test, y_pred_sgd, labels=labels_in_test, target_names=le.inverse_transform(labels_in_test)))
print("Confusion Matrix (SGD):")
print(confusion_matrix(y_test, y_pred_sgd, labels=labels_in_test))

print("✅ Dual models trained on FULL unbalanced dataset!")

# === Function to plot confusion matrix ===
def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

# === Plot for RandomForest ===
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=labels_in_test)
plot_confusion_matrix(cm_rf, le.inverse_transform(labels_in_test), "🎯 Confusion Matrix - RandomForest")

print("📊 RandomForest F1 Score:", f1_score(y_test, y_pred_rf, average='weighted'))
print("📊 RandomForest Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("📊 RandomForest Recall:", recall_score(y_test, y_pred_rf, average='weighted'))

# === Plot for SGDClassifier ===
cm_sgd = confusion_matrix(y_test, y_pred_sgd, labels=labels_in_test)
plot_confusion_matrix(cm_sgd, le.inverse_transform(labels_in_test), "🧠 Confusion Matrix - SGDClassifier")

print("📊 SGD F1 Score:", f1_score(y_test, y_pred_sgd, average='weighted'))
print("📊 SGD Precision:", precision_score(y_test, y_pred_sgd, average='weighted'))
print("📊 SGD Recall:", recall_score(y_test, y_pred_sgd, average='weighted'))