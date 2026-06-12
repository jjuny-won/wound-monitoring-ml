import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 1. 데이터 로드 및 전처리
# ──────────────────────────────────────────
FILE_PATH = "/mnt/user-data/uploads/pH_색상변화_ver2.xlsx"

df_raw = pd.read_excel(FILE_PATH, header=None)

# 컬럼 구조: pH5(R,G,B) | pH6(R,G,B) | pH7(R,G,B) | pH8(R,G,B) | pH9(R,G,B)
# Row 0: pH 라벨, Row 1: R/G/B 헤더, Row 2~: 데이터

pH_labels = ['pH5', 'pH6', 'pH7', 'pH8', 'pH9']
col_starts = [0, 3, 6, 9, 12]  # 각 pH 시작 컬럼 인덱스

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
print(f"\nRGB 범위: R({df['R'].min()}~{df['R'].max()}), G({df['G'].min()}~{df['G'].max()}), B({df['B'].min()}~{df['B'].max()})")

# ──────────────────────────────────────────
# 2. 학습 준비
# ──────────────────────────────────────────
X = df[['R', 'G', 'B']].values
le = LabelEncoder()
y = le.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ──────────────────────────────────────────
# 3. Random Forest 학습
# ──────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ──────────────────────────────────────────
# 4. 평가
# ──────────────────────────────────────────
y_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

print("\n" + "=" * 55)
print("  모델 성능 결과")
print("=" * 55)
print(f"Test Accuracy  : {test_acc*100:.2f}%")
print(f"5-Fold CV      : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

print("\n클래스별 분류 리포트:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nFeature Importance (R / G / B):")
for feat, imp in zip(['R', 'G', 'B'], rf.feature_importances_):
    print(f"  {feat}: {imp*100:.1f}%")

# ──────────────────────────────────────────
# 5. 예측 테스트 (임의 RGB 입력)
# ──────────────────────────────────────────
print("\n" + "=" * 55)
print("  예측 테스트 (임의 RGB 입력)")
print("=" * 55)

test_samples = [
    [100, 29, 67],   # pH5 근처
    [104, 44, 95],   # pH6 근처
    [132, 141, 152], # pH7 근처
    [20, 56, 75],    # pH8 근처
    [56, 81, 88],    # pH9 근처
]

for rgb in test_samples:
    pred = rf.predict([rgb])[0]
    proba = rf.predict_proba([rgb])[0]
    label = le.inverse_transform([pred])[0]
    conf = proba.max() * 100
    print(f"  RGB{rgb} → {label}  (신뢰도: {conf:.1f}%)")

# ──────────────────────────────────────────
# 6. 시각화 저장
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('pH Random Forest Classification Results', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title(f'Confusion Matrix\nAcc: {test_acc*100:.1f}%')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Feature Importance
feat_names = ['R', 'G', 'B']
colors = ['#e74c3c', '#2ecc71', '#3498db']
bars = axes[1].bar(feat_names, rf.feature_importances_ * 100, color=colors, edgecolor='black', linewidth=0.8)
axes[1].set_title('Feature Importance (%)')
axes[1].set_ylabel('Importance (%)')
for bar, val in zip(bars, rf.feature_importances_ * 100):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', fontweight='bold')

# CV Scores
fold_labels = [f'Fold {i+1}' for i in range(5)]
axes[2].bar(fold_labels, cv_scores * 100, color='#9b59b6', edgecolor='black', linewidth=0.8)
axes[2].axhline(cv_scores.mean() * 100, color='red', linestyle='--', label=f'Mean: {cv_scores.mean()*100:.1f}%')
axes[2].set_title('5-Fold Cross Validation')
axes[2].set_ylabel('Accuracy (%)')
axes[2].set_ylim([0, 105])
axes[2].legend()

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/rf_ph_result.png', dpi=150, bbox_inches='tight')
print("\n결과 그래프 저장 완료!")
print("\n모든 분석 완료!")
