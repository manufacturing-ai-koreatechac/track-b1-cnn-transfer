# Part 1-2: AI 기초와 데이터 이해 (품질/FA 엔지니어용)

> **v12 Enhanced**: AI Hub 비전 데이터 + 이미지 처리 + 객체 탐지 기초

---

## 🎯 학습 목표

- ✅ AI Hub 공공 비전 데이터 활용
- ✅ 이미지 데이터 로드 및 전처리
- ✅ OpenCV 기본 이미지 처리
- ✅ 간단한 객체 탐지 실습
- ✅ 불량 이미지 분류 기초

---

## 📚 실습 구성

| 순서 | 실습 | 파일 | 소요 시간 | 난이도 |
|:----:|------|------|:---------:|:------:|
| 1 | AI Hub 데이터 탐색 | `01_aihub_vision_data.ipynb` | 30분 | ⭐ |
| 2 | 이미지 전처리 | `02_image_preprocessing.ipynb` | 45분 | ⭐⭐ |
| 3 | 불량 분류 기초 | `03_defect_classification.ipynb` | 45분 | ⭐⭐ |

**총 소요 시간**: 약 2시간

---

## 🚀 시작하기

### 1️⃣ 환경 설정

```bash
# Part 1-2 폴더로 이동
cd practice-v12-enhanced/part1-2

# 추가 패키지 설치 (이미지 처리)
pip install pillow opencv-python scikit-image
```

### 2️⃣ Jupyter Lab 실행

```bash
jupyter lab
```

---

## 📊 사용 데이터

### AI Hub 반도체 불량 이미지

| 항목 | 내용 |
|------|------|
| **출처** | [AI Hub](https://www.aihub.or.kr/) |
| **설명** | 반도체 웨이퍼 불량 이미지 |
| **카테고리** | 정상, Scratch, Edge, Center |

### AI Hub OHT/AGV 이미지

| 항목 | 내용 |
|------|------|
| **출처** | [AI Hub](https://www.aihub.or.kr/) |
| **설명** | 자율주행 물류 로봇 이미지 |
| **용도** | 장애물 탐지, 경로 인식 |

---

## 🔧 실습 상세 내용

### 실습 1: AI Hub 데이터 탐색 (30분)

**학습 내용**:
- AI Hub 데이터셋 구조 이해
- 이미지 파일 로드 및 시각화
- 레이블 데이터 분석
- 데이터 분포 확인

**주요 코드**:
```python
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 로드
img = Image.open('defect_001.jpg')
plt.imshow(img)
plt.title('반도체 불량 이미지')
plt.show()
```

### 실습 2: 이미지 전처리 (45분)

**학습 내용**:
- 이미지 리사이징 및 정규화
- 색상 공간 변환 (RGB, HSV, Grayscale)
- 노이즈 제거 및 필터링
- 데이터 증강 (Augmentation)

**주요 코드**:
```python
import cv2
import numpy as np

# 이미지 전처리 파이프라인
def preprocess_image(img_path):
    # 로드
    img = cv2.imread(img_path)

    # 리사이즈
    img = cv2.resize(img, (224, 224))

    # 정규화
    img = img / 255.0

    return img
```

### 실습 3: 불량 분류 기초 (45분)

**학습 내용**:
- 전통적 머신러닝 분류기
- 이미지 피처 추출 (HOG, SIFT)
- 간단한 CNN 모델
- 성능 평가 및 시각화

**주요 코드**:
```python
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog

# HOG 피처 추출
features = []
for img_path in image_paths:
    img = cv2.imread(img_path, 0)  # Grayscale
    hog_feat = hog(img, pixels_per_cell=(16, 16))
    features.append(hog_feat)

# 분류
clf = RandomForestClassifier()
clf.fit(features, labels)
```

---

## 💡 학습 팁

### 이미지 크기 확인
```python
from PIL import Image
img = Image.open('image.jpg')
print(f"크기: {img.size}, 모드: {img.mode}")
```

### 여러 이미지 한 번에 시각화
```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, img_path in enumerate(image_paths[:6]):
    img = Image.open(img_path)
    axes[idx//3, idx%3].imshow(img)
    axes[idx//3, idx%3].set_title(f'Image {idx+1}')
plt.tight_layout()
plt.show()
```

---

## 관련 공개 데이터셋

| # | 데이터셋 | 설명 | 규모 | 링크 |
|:-:|---------|------|:----:|------|
| 1 | **MVTec Anomaly Detection Dataset** | 산업용 물체·텍스처 15종 고해상도 이미지. 정상 학습 이미지 + 70가지 이상 유형. CNN 전이학습 벤치마크 표준. | 5,354장 | [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| 2 | **NEU Surface Defect Database** | 열연강판 6가지 결함 유형(롤마크·크랙·스케일 등). ResNet/CNN 분류 연구에 광범위하게 사용. 불균형 클래스 처리 실습 적합. | 1,800장 | [NEU](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) |
| 3 | **Severstal Steel Defect Detection** | 세베르스탈 철강 제조 실제 라인 데이터. 4종 결함 유형 픽셀 단위 마스크 포함. Kaggle 대회 데이터로 완성도 높음. | 18,000장 | [Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection/data) |

## 📚 참고 자료

### 데이터 소스
- [AI Hub](https://www.aihub.or.kr/)
- [공공데이터포털](https://www.data.go.kr/)

### 라이브러리 문서
- [OpenCV-Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Pillow](https://pillow.readthedocs.io/)
- [scikit-image](https://scikit-image.org/)

---

## 🎓 학습 체크리스트

- [ ] AI Hub 데이터를 성공적으로 로드했다
- [ ] 이미지 전처리 파이프라인을 구현했다
- [ ] 불량 이미지를 분류하는 모델을 만들었다
- [ ] 모델 성능을 평가하고 시각화했다

---

*제조AI 교육 v12 Enhanced | Part 1-2 | 2025.02*
