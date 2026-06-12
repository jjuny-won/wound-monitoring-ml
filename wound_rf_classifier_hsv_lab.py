import pandas as pd
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 0. 설정 (파일 경로만 여기서 수정)
# ──────────────────────────────────────────
FILE_PATH = "pH_색상변화_ver2.xlsx"   # 실행 환경에 맞게 수정
OUTPUT_IMG = "rf_ph_result_hsv_lab.png"

FEATURE_SETS = {
    "RGB":         ['R', 'G', 'B'],
    "RGB+HSV":     ['R', 'G', 'B', 'H', 'S', 'V'],
    "RGB+LAB":     ['R', 'G', 'B', 'L', 'A', 'B_lab'],
    "RGB+HSV+LAB": ['R', 'G', 'B', 'H', 'S', 'V', 'L', 'A', 'B_lab'],
}

# ──────────────────────────────────────────
# 1. 데이터 로드 및 전처리
# ──────────────────────────────────────────
df_raw = pd.read_excel(FILE_PATH, header=None)

pH_labels = ['pH5', 'pH6', 'pH7', 'pH8', 'pH9']
col_starts = [0, 3, 6, 9, 12]

records = []
for label, start in zip(pH_labels, col_starts):
    sub = df_raw.iloc[2:, start:start+3].copy()
    sub.columns = ['R', 'G', 'B']
    sub = sub.dropna()
    sub = sub[sub['R'].apply(lambda x: str(x).replace('.','').isdigit())]
    sub = sub.astype(float).astype(int)
    sub['label'] = label
    records.append(sub)

df = pd.concat(records, ignore_index=True)

print("=" * 55)
print("  pH 색상 데이터셋 요약")
print("=" * 55)
print(f"총 샘플 수: {len(df)}")
print("\n클래스별 샘플 수:")
print(df['label'].value_counts().sort_index().to_string())

# ──────────────────────────────────────────
# 2. HSV / LAB feature 추가
# ──────────────────────────────────────────
def rgb_to_extra_features(r, g, b):
    """RGB → HSV, LAB 변환 후 6개 채널 반환"""
    pixel = np.array([[[int(b), int(g), int(r)]]], dtype=np.uint8)  # BGR
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0][0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2]), \
           float(lab[0]), float(lab[1]), float(lab[2])

df[['H', 'S', 'V', 'L', 'A', 'B_lab']] = df.apply(
    lambda row: rgb_to_extra_features(row['R'], row['G'], row['B']),
    axis=1, result_type='expand'
)

print(f"\nFeature 확장 완료: RGB(3) → RGB+HSV+LAB(9)")
print(df[['R','G','B','H','S','V','L','A','B_lab']].describe().round(1).to_string())

# ──────────────────────────────────────────
# 3. Feature Set 별 RF 비교 실험
# ──────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(df['label'])

results = {}

print("\n" + "=" * 55)
print("  Feature Set 별 성능 비교")
print("=" * 55)

for name, features in FEATURE_SETS.items():
    X = df[features].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

    results[name] = {
        'model': rf,
        'features': features,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
    }

    print(f"\n[{name}]")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  5-Fold CV     : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ──────────────────────────────────────────
# 4. 최종 모델 (RGB+HSV+LAB) 상세 리포트
# ──────────────────────────────────────────
best_name = "RGB+HSV+LAB"
best = results[best_name]

print("\n" + "=" * 55)
print(f"  [{best_name}] 클래스별 분류 리포트")
print("=" * 55)
print(classification_report(best['y_test'], best['y_pred'], target_names=le.classes_))

print("Feature Importance:")
feat_names = best['features']
for feat, imp in zip(feat_names, best['model'].feature_importances_):
    print(f"  {feat:6s}: {imp*100:.1f}%")

# ──────────────────────────────────────────
# 5. 예측 테스트 (임의 RGB 입력)
# ──────────────────────────────────────────
print("\n" + "=" * 55)
print("  예측 테스트 (임의 RGB 입력)")
print("=" * 55)

test_samples = [
    [100, 29, 67],
    [104, 44, 95],
    [132, 141, 152],
    [20, 56, 75],
    [56, 81, 88],
]

rf_best = best['model']
for rgb in test_samples:
    r, g, b = rgb
    h, s, v, l, a, b_lab = rgb_to_extra_features(r, g, b)
    feat_vec = [r, g, b, h, s, v, l, a, b_lab]
    pred = rf_best.predict([feat_vec])[0]
    proba = rf_best.predict_proba([feat_vec])[0]
    label = le.inverse_transform([pred])[0]
    conf = proba.max() * 100
    print(f"  RGB{rgb} → {label}  (신뢰도: {conf:.1f}%)")

# ──────────────────────────────────────────
# 6. 시각화 저장
# ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('pH RF Classification: Feature Set Comparison', fontsize=14, fontweight='bold')

# (1) Feature Set별 Accuracy 비교
ax = axes[0][0]
names = list(results.keys())
accs = [results[n]['test_acc'] * 100 for n in names]
cv_means = [results[n]['cv_mean'] * 100 for n in names]
cv_stds = [results[n]['cv_std'] * 100 for n in names]
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, accs, w, label='Test Acc', color='#3498db', edgecolor='black')
ax.bar(x + w/2, cv_means, w, yerr=cv_stds, label='CV Mean', color='#e67e22',
       edgecolor='black', capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, fontsize=9)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Feature Set별 Accuracy 비교')
ax.set_ylim([80, 105])
ax.legend()
for i, (a_val, c_val) in enumerate(zip(accs, cv_means)):
    ax.text(i - w/2, a_val + 0.3, f'{a_val:.1f}', ha='center', fontsize=8)
    ax.text(i + w/2, c_val + 0.3, f'{c_val:.1f}', ha='center', fontsize=8)

# (2) RGB-only Confusion Matrix
ax = axes[0][1]
cm = confusion_matrix(results['RGB']['y_test'], results['RGB']['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title(f"Confusion Matrix: RGB only\nAcc: {results['RGB']['test_acc']*100:.1f}%")
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# (3) RGB+HSV+LAB Confusion Matrix
ax = axes[1][0]
cm2 = confusion_matrix(best['y_test'], best['y_pred'])
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title(f"Confusion Matrix: RGB+HSV+LAB\nAcc: {best['test_acc']*100:.1f}%")
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# (4) Feature Importance (RGB+HSV+LAB)
ax = axes[1][1]
feat_imp = best['model'].feature_importances_
colors_map = {
    'R': '#e74c3c', 'G': '#2ecc71', 'B': '#3498db',
    'H': '#f39c12', 'S': '#e67e22', 'V': '#d35400',
    'L': '#9b59b6', 'A': '#8e44ad', 'B_lab': '#6c3483'
}
bar_colors = [colors_map[f] for f in feat_names]
bars = ax.bar(feat_names, feat_imp * 100, color=bar_colors, edgecolor='black', linewidth=0.8)
ax.set_title(f'Feature Importance: {best_name}')
ax.set_ylabel('Importance (%)')
for bar, val in zip(bars, feat_imp * 100):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
print(f"\n결과 그래프 저장 완료: {OUTPUT_IMG}")
print("\n모든 분석 완료!")
