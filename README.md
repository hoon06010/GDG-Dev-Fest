# GDG-Dev-Fest

Epic 1. 코드 리뷰 & 베이스라인 확립
 1.1 Feature Engineering 이유 분석
  - name에서 Title 추출 -> age 결측치 처리에서 사용
  - fare 중앙값으로 결측치 처리 & 로그 변환 -> 데이터 평준화
  - embarked 최빈값으로 결측치 처리 -> 범주형 데이터이므로 최빈값으로 처리
  - family_size, sibsp, parch를 Family_Group(범주형 데이터)으로 통합 -> 데이터 단순화로 효율적인 학습

 1.2 Baseline 설정
  - macro avg의 f1-score: 0.82

 1.3 평가 지표(f1-score) 이해
  - f1_score: 정확도와 정밀도의 조화평균
  - 데이터가 불균형할 때 모델의 성능을 더 합리적으로 평가할 수 있는 지표

Epic 2. ML 모델 튜닝 & 교차 검증
 2.1 GridSearchCV 튜닝 준비

 2.2 LogisticRegression(lr) Hyperparameter Tuning
  - lr 튜닝 파라미터 정의 (ex. param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']})
  - GridSearchCV 객체 생성
  - X_train_processed, y_train 데이터로 grid_search.fit()을 실행
  - grid_search.best_score(최고 F1)와 grid_search.best_params(최적 파라미터)를 확인
  - 검증 결과: 0.7972

 2.3 SVC & KNeighborsClassifier Tuning
  - SVC & KNeighborsClassifier 모델 임포트, GridSearchCV 실행
  - (ex. param_grid_svc = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}, param_grid_knn = {'n_neighbors': [3, 5, 7, 9, 11]})
  - f1_score 검증
  - 검증 결과: (SVC: 0.7925, KNN: 0.7911)

Epic 3. DNN 모델 도입
 3.1 TensorFlow/Keras 라이브러리 준비
  `import tensorflow as tf`
  `from tensorflow.keras.models import Sequential`
  `from tensorflow.keras.layers import Dense, Dropout, Input`
  `from tensorflow.keras.callbacks import EarlyStopping`

 3.2 DNN 모델 구축
  - model.add(Input(shape=(X_train_processed.shape[1],))) (Sequential 모델을 정의)
  - model.add(Dense(32, activation='relu')) (1층)
  - model.add(Dropout(0.2)) (과적합 방지)
  - model.add(Dense(16, activation='relu')) (2층)
  - model.add(Dense(1, activation='sigmoid')) (이진 분류를 위한 최종 출력)

 3.3 모델 컴파일, 학습
  - tf.keras.metrics.Precision과 tf.keras.metrics.Recall을 metrics로 추가 -> fl_score 측정
  - EarlyStopping 콜백 정의 (ex. monitor='val_loss', patience=10)
  - model.fit()
  - 검증 결과: 0.80


