# 마비말 진단 모델에서의 PnPXAI 프레임워크 활용

최종 업데이트: 2024-12-26

이 문서는 마비말 진단 모델에 PnPXAI를 활용하여 (1) 특징 중요도 산출 (2)피쳐별 우수/개선 영역 분석을 수행하는 방법을 소개합니다. 

## 마비말 진단 모델 개요

**DDK(Task)란?**
* **DDK**(Diadochokinetic) Task는 구어운동조절능력 평가용 작업이며, 신경학적 말장애나 운동조절 문제 진단에 자주 활용됩니다.
* /퍼/, /터/, /커/ 그리고 /퍼터커/와 같은 반복 음절을 일정 속도·리듬으로 발화하도록 하여, 발음 규칙성·속도·리듬·쉼 간격 등을 평가합니다.
* 본 튜토리얼에 사용된 마비말 모델은 DDK task에서 마비말의 심각도를 예측합니다.

**모델 입력**
1. Mel-spectrogram
2. Wav2Vec2
3. DDK (발음 반복 횟수, 규칙성, 쉼 간격 등)

**모델 구조**
* 다층 퍼셉트론(MLP), ResNet, Wav2Vec2 layer 등을 결합한 형태

**모델 출력**
* 심각도: 0 (정상), 1 (경증), 2 (중증)

## 전체 파이프라인 요약
**마비말 진단 모델에서**, 주요 특징(Mel-spectrogram, Wav2Vec2, DDK)을 이용해 예측을 수행합니다. 그리고 예측 과정에서 pnpxai 라이브러리에 포함된 **설명알고리즘**을 적용해, 모델이 어느 특징에 크게 의존했는지 정량적으로 확인합니다 (본 튜토리얼에서는 여러 설명 알고리즘 중 Integrated Gradients (IG) 알고리즘 사용). 그리고 IG 결과를 바탕으로 각 피처가 **우수(0)/보통(1)/개선(2)** 중 어디에 해당하는지를 추론하고, 이를 **언어치료사가 직접 레이블링한 데이터**와 비교함으로써 모델의 해석 가능성과 예측 신뢰도를 함께 검증하는 과정을 거치게 됩니다.

1. 환경 세팅 및 데이터 다운로드
* 합성음 + 레이블(언어치료사 레이블) + 모델 파라미터(ckpt) 다운로드
2. 특징 추출 및 CSV 저장
* 마비말 진단 모델에 사용되는 feature들을 추출후 CSV 생성(DDK, Wav2Vec, Mel-spectrogram)
2. 모델 Inference
* CSV로부터 특징을 로드하고, 예측을 수행하여 (id, task_id → 심각도)를 출력
3. PnPXAI 기반 특징 중요도 산출
* `pnpxai.explainers.IntegratedGradients`를 사용하여 특징별 중요도(IG) 계산
4. 우수/개선 영역 분석
* 언어치료사가 레이블링한 6개 DDK feature(예: ddk_rate, ddk_pause_rate 등)에 대해 우수(0)/보통(1)/개선(2) 영역 추출

## 환경 세팅 및 데이터 다운로드
마비말 진단 모델관련 패키지를 설치하고 모델 파라미터와 DDK 테스트셋 음성 데이터를 다운로드합니다.
```bash
# 마비말 진단 모델 설치 (github 리포지토리)
git clone git@github.com:sogang-isds/xai_ddk_multi.git

# 합성 데이터 다운로드
cd xai_ddk_multi
mkdir data

# 합성음 (synthesized 음성) 다운로드 및 압축 해제
gdown --id 1ukq2RNBeh2Rfwde04wVUUlN3lya1LT1q
unzip ddk_test_synthesized.zip -d data

# 정답지 다운로드 (환자별 중증도 + 언어치료사판단한 피쳐별 우수/개선 영역 레이블)
gdown --id 1w8G7txx4ArvLPyRXNgJ8E6KBPPqxc0Ab
mv test_labels.csv data

# 모델 파라미터 (checkpoint 등)
gdown "https://drive.google.com/drive/folders/1Jx5nYAj9NLMjANRd8b5_iQKZvYug2yiq" --folder
```
위 과정을 수행하면, xai_ddk_multi 디렉토리 내부가 아래 예시와 같이 구성됩니다:
```bash
xai_ddk_multi/
├── data/
│   ├── ddk_test_synthesized/   # 합성된 DDK 음성 데이터
│   └── test_labels.csv         # 환자별 중증도 + 언어치료사 레이블 (우수/개선 영역)
├── checkpoints/
│   ├── intelligibility_model.ckpt 
│   ├── vad_model.ckpt
│   ├── multi_input_model.ckpt  # 다운받은 모델 파라미터
│   └── ...
└── (... 기타 코드 및 라이브러리 ...)
```

## 특징 추출 및 CSV 저장
아래 코드는 DDK 모델에서 사용하는 특징을 추출하여, `test_data_1.csv`와 `test_data_2.csv` 형태로 저장합니다.

```python
from common import APP_ROOT
import os
import glob
import pandas as pd
from math import isnan
from argparse import Namespace

import torch
import torchaudio

from multi_input_model import DDKWav2VecModel
from intelligibility_model import PTKModel
from vad_model import VADModel
from get_features import (
    prepare_data, my_padding, get_speech_interval,
    cal_features_from_speech_interval
)
from ddk.features import DDK

MODEL_DIR = os.path.join(APP_ROOT, 'checkpoints')

def prepare_audio(wav):
    """
    First time setup of the dataset
    """

    y, sr = torchaudio.load(wav)  # type:ignore
    
    if sr != 16000:
        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=16000)
    
    y = y.mean(0)
    
    y = my_padding(y, 16000*15)

    return y.unsqueeze(0)


class FeaturesGenerator(object):
    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
        self.ddk_path = os.path.join(APP_ROOT, 'ddk/')

        intel_args = {
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 100,
            'n_gpu': 1,
        }

        intel_args = Namespace(**intel_args)
        
        intel_path = os.path.join(MODEL_DIR, "intelligibility_model.ckpt")
        self.intel_model = PTKModel.load_from_checkpoint(intel_path,n_classes=5,args=intel_args)
        self.intel_model.to(self.device)
        self.intel_model.eval()

        vad_args = {
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 100,
            'n_gpu': 1,
            'hidden_size': 128,
            'num_layers': 16
        }
        
        vad_args = Namespace(**vad_args)
        vad_path = os.path.join(MODEL_DIR, "vad_model.ckpt")
        self.vad_model = VADModel.load_from_checkpoint(vad_path, args=vad_args)
        self.vad_model.to(self.device)

        self.model = model

    def get_intelligibility(self, path):       
        y = prepare_data(path)

        y = my_padding(y, 16000 * 15).to(self.device)
        y = self.intel_model.mel_converter(y)
        y = y.unsqueeze(0)
        y = y.unsqueeze(0)
        y = self.intel_model(y)
        
        intelligibility = torch.argmax(y, dim=-1).cpu().numpy()
        
        return int(intelligibility[0])       

    def get_phonation_features(self, path):        
        ddk_features = DDK(path, self.ddk_path)
        ddk_features = [0 if isnan(x) else x for x in ddk_features]
        return ddk_features

    def get_prosody_respiration_features(self, path, threshold, min_dur):
        audio_tensor, sample_rate  = torchaudio.load(path)
        if sample_rate != 16000:
            audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
            sample_rate = 16000
        audio_tensor = audio_tensor.mean(0)
        
        dur = len(audio_tensor) / sample_rate
        
        pad_size = int(16000*0.75)
        padding = torch.zeros((pad_size))
        
        audio_tensor = torch.concat([padding, audio_tensor])
        
        audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        mel = self.vad_model.spec_converter(audio_tensor)
        
        out = self.vad_model(mel)
        out = self.vad_model.sigmoid(out)

        out = (out > threshold).float().cpu()

        out = get_speech_interval(out, min_dur)
        
        ddk_rate, ddk_avg, ddk_std, ddk_pause_rate, pause_avg, ddk_pause_std = cal_features_from_speech_interval(out, dur)
        
        return ddk_rate, ddk_avg, ddk_std, ddk_pause_rate, pause_avg, ddk_pause_std

    def get_features(self, audio_path, gender):
        intelligibility = self.get_intelligibility(audio_path)
        phonation_features = self.get_phonation_features(audio_path)

        threshold = 0.7
        min_dur = 0.08
        prosody_respiration_features = self.get_prosody_respiration_features(audio_path, threshold, min_dur)
        
        features = [gender, intelligibility]
        features += phonation_features
        features += prosody_respiration_features

        audio = prepare_audio(audio_path).to(self.device)
        mel_x = self.model.mel_spectrogram(audio.unsqueeze(0))
        mel_x = self.model.db_converter(mel_x)

        spec_x = mel_x
        w2v_x = audio
        char_x = features

        # CNN(ResNet)
        spec_x, _ = self.model.resnet_model(spec_x)
        
        # CNN projection
        spec_x = self.model.post_spec_layer(spec_x)
        spec_x = self.model.relu(spec_x)
        spec_x = self.model.post_spec_dropout(spec_x)

        spec_attn_x = spec_x.reshape(spec_x.shape[0], 1, -1)
        
        # wav2vec 2.0 
        w2v_x = self.model.wav2vec(w2v_x)[0]
        w2v_x = torch.matmul(spec_attn_x, w2v_x)
        w2v_x = w2v_x.reshape(w2v_x.shape[0], -1)
        
        # wav2vec projection
        w2v_x = self.model.post_w2v_layer(w2v_x)
        w2v_x = self.model.relu(w2v_x)
        w2v_x = self.model.post_wv2_droput(w2v_x)
        
        # CNN + wav2vec concat and projection
        spec_w2v_x = torch.cat([spec_x, w2v_x], dim=-1)
        spec_w2v_x = self.model.post_attn_layer(spec_w2v_x)
        spec_w2v_x = self.model.relu(spec_w2v_x)
        spec_w2v_x = self.model.post_attn_dropout(spec_w2v_x)

        spec_w2v_x = spec_w2v_x.squeeze()  # (128,) 모양으로 차원 제거
        spec_w2v_x = spec_w2v_x.tolist()  # 리스트로 변환

        return spec_w2v_x, features 

df_labels = pd.read_csv(os.path.join(APP_ROOT, 'tutorial_logs/test_labels.csv'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDKWav2VecModel.load_from_checkpoint(os.path.join(MODEL_DIR, "multi_input_model.ckpt")).to(device).eval()

features_generator = FeaturesGenerator(model)

column_names = [
    "id",  
    "task_id", 
    # ==================== #
    "gender",
    "intelligibility",
    "var_f0_semitones",
    "var_f0_hz",
    "avg_energy",
    "var_energy",
    "max_energy",
    "ddk_rate",
    "ddk_average",
    "ddk_std",
    "ddk_pause_rate",
    "ddk_pause_average",
    "ddk_pause_std"
]

audio_paths = glob.glob(f'{APP_ROOT}/data/ddk_test_synthesized/*/*/*.wav')

# Initialize an empty list to hold all features data
data_1 = []
data_2 = []

for i, audio_path in enumerate(audio_paths):
    print(i, len(audio_paths))
    filename = os.path.basename(audio_path)[:-4]
    if 'nia' in filename:
        id = filename.split('_')[0] + '_' + filename.split('_')[1]
        task_id = filename.split('_')[6]
        gender = filename.split('_')[3]
    else:
        id = filename.split('_')[0]    
        task_id = filename.split('_')[1]
        gender = filename.split('_')[2]

    if id in df_labels['id'].values:
        spec_w2v_x, features = features_generator.get_features(audio_path, gender)
        data_1.append([id] + [task_id] + spec_w2v_x)
        data_2.append([id] + [task_id] + features)
    else:
        print(f"{id} is NOT present in df_labels['id']")

# spec_w2v_x에 대한 컬럼 이름을 자동으로 생성 (spec_w2v_x_1, spec_w2v_x_2, ..., spec_w2v_x_128)
spec_w2v_columns = [f"spec_w2v_x_{i+1}" for i in range(128)]

# Create a DataFrame with the collected features for df_1
df_1 = pd.DataFrame(data_1, columns=["id", "task_id"] + spec_w2v_columns)

# Save the DataFrame to a CSV file
df_1.to_csv(os.path.join(APP_ROOT, 'logs/test_data_1.csv'), index=False)

# Create a DataFrame with the collected features
df_2 = pd.DataFrame(data_2, columns=column_names)

# Save the DataFrame to a CSV file
df_2.to_csv(os.path.join(APP_ROOT, 'logs/test_data_2.csv'), index=False)
```

## 모델 Inference 및 예측 결과 저장

위에서 생성된 `test_data_1.csv`와 `test_data_2.csv`를 로드하여 마비말 모델에 입력하고, id, task_id별 **심각도 예측**(0/1/2)을 CSV(test_preds.csv)에 저장합니다.
```python
import joblib
import numpy as np

scaler = joblib.load(os.path.join(APP_ROOT, "scaler.save"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDKWav2VecModel.load_from_checkpoint(os.path.join(MODEL_DIR, "multi_input_model.ckpt")).to(device).eval()

df1 = pd.read_csv(os.path.join(APP_ROOT, 'logs/test_data_1.csv'), dtype={'task_id': str})
df2 = pd.read_csv(os.path.join(APP_ROOT, 'logs/test_data_2.csv'), dtype={'task_id': str})

results = []

for (idx1, row1), (idx2, row2) in zip(df1.iterrows(), df2.iterrows()):
    id = row1['id']
    task_id = row1['task_id']

    # 첫 두 컬럼(id, task_id)를 제외하고 나머지를 선택
    row1_filtered = row1.iloc[2:]  # iloc[2:]를 사용하여 2번째 이후의 값들만 선택
    
    # 나머지 값을 PyTorch 텐서로 변환
    spec_w2v_x = torch.tensor(row1_filtered.values.astype(np.float32), dtype=torch.float32, device=device)
    spec_w2v_x = spec_w2v_x.unsqueeze(dim=0)

    # 첫 두 컬럼(id, task_id)를 제외하고 나머지를 선택
    row2_filtered = row2.iloc[2:]  # iloc[2:]를 사용하여 2번째 이후의 값들만 선택
    char_x = scaler.transform([row2_filtered.values])
    char_x = torch.tensor(char_x).float().to(device)
    
    # Characteristics projection
    char_x_pro = model.char_layer(char_x)

    # CNN + wav2vec + characteristics concat
    total_x = torch.cat([spec_w2v_x, char_x_pro], dim=-1)
    
    # Final output
    out = model.final_layer(total_x)

    severity = torch.argmax(out, dim=-1)[0]
    severity = int(severity)

    # 결과 저장
    results.append({'id': id, 'task_id': task_id, 'severity': severity})

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)

# 결과 DataFrame을 CSV 파일로 저장
results_df.to_csv(os.path.join(APP_ROOT, 'logs/test_preds.csv'), index=False)
```

**마비말 진단 모델 예측 성능 출력 결과 예시**
```python

from sklearn.metrics import classification_report

# Load the two CSV files
test_labels = pd.read_csv(os.path.join(APP_ROOT, 'data/test_labels.csv'))
test_preds = pd.read_csv(os.path.join(APP_ROOT, 'logs/test_preds.csv'))

# Merge the files on 'id'
merged_df = pd.merge(test_labels, test_preds, on='id', suffixes=('_true', '_pred'))

# Calculate the accuracy for each task
task_ids = merged_df['task_id'].unique()
task_accuracies = {}

for task_id in task_ids:
    task_df = merged_df[merged_df['task_id'] == task_id]
    accuracy = (task_df['severity_true'] == task_df['severity_pred']).mean()
    task_accuracies[task_id] = accuracy

# Overall accuracy
overall_accuracy = (merged_df['severity_true'] == merged_df['severity_pred']).mean()

# Classification report (Precision, Recall, F1-score for each class)
class_report = classification_report(merged_df['severity_true'], merged_df['severity_pred'], target_names=['Class 0', 'Class 1', 'Class 2'])

# Display results
print("Task-wise accuracies:")
for task_id, accuracy in task_accuracies.items():
    print(f"Task {task_id}: {accuracy:.2%}")

print(f"\nOverall accuracy: {overall_accuracy:.2%}")

print("\nClassification Report (Per-Class Performance):")
print(class_report)
```

```
Task-wise accuracies:
Task 2: 69.64%
Task 3: 75.00%
Task 4: 69.64%
Task 5: 71.43%

Overall accuracy: 71.43%

Classification Report (Per-Class Performance):
              precision    recall  f1-score   support

     Class 0       0.10      0.06      0.08        32
     Class 1       0.79      0.88      0.83       180
     Class 2       0.00      0.00      0.00        12

    accuracy                           0.71       224
   macro avg       0.30      0.31      0.30       224
weighted avg       0.65      0.71      0.68       224
```

> **Note**: task_id는 /퍼/, /터/, /커/, /퍼터커/ 발음과 대응됩니다. 예를 들어:
> * Task 2: /퍼/ 발음 반복
> * Task 3: /터/ 발음 반복
> * Task 4: /커/ 발음 반복
> * Task 5: /퍼터커/ 연속 발음

## PnPXAI를 활용한 특징 중요도산출
본 튜토리얼에서는 `pnpxai.explainers.IntegratedGradients`를 사용하여 특징별 기여도를 산출합니다. 아래 예시에서는 `test_data_1.csv` + `test_data_2.csv`를 concat한 뒤 IG를 구하고, 그 결과를 `test_ig_pnpxai.csv`에 저장합니다.

```python
import torch.nn as nn
from pnpxai.explainers import IntegratedGradients


# CustomNet 정의 (concat된 입력을 받아 spec_w2v_x와 char_x로 분리)
class CustomNet(nn.Module):
    def __init__(self, model):
        super(CustomNet, self).__init__()
        # model의 char_layer와 final_layer를 사용
        self.char_layer = model.char_layer
        self.final_layer = model.final_layer
    
    def forward(self, concat_x):
        # 입력 텐서를 spec_w2v_x와 char_x로 분리
        spec_w2v_x = concat_x[:, :128]  # 첫 128차원은 spec_w2v_x로 할당
        char_x = concat_x[:, 128:]      # 나머지는 char_x로 할당
        
        # char_layer 통과
        char_x_pro = self.char_layer(char_x)
        
        # CNN + wav2vec + characteristics concat
        total_x = torch.cat([spec_w2v_x, char_x_pro], dim=-1)
        
        # final_layer 통과
        out = self.final_layer(total_x)
        
        return out


def generate_column_dict(column_names, values):
    value_dict = {}
    for column, value in zip(column_names, values):
        if isinstance(value, float):
            value_dict[column] = round(value, 3)
        else:
            value_dict[column] = value

    return value_dict


df1 = pd.read_csv(os.path.join(APP_ROOT, 'logs/test_data_1.csv'), dtype={'task_id': str})
df2 = pd.read_csv(os.path.join(APP_ROOT, 'logs/test_data_2.csv'), dtype={'task_id': str})

scaler = joblib.load(os.path.join(APP_ROOT, "scaler.save"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDKWav2VecModel.load_from_checkpoint(os.path.join(MODEL_DIR, "multi_input_model.ckpt")).to(device).eval()

# CustomNet 인스턴스화
custom_model = CustomNet(model)

df1_features = df1.iloc[:, 2:].values.astype(np.float32)
df2_features = scaler.transform(df2.iloc[:, 2:].values.astype(np.float32))
df1_mean = np.mean(df1.iloc[:, 2:].values.astype(np.float32), axis=0)
df2_mean = np.mean(scaler.transform(df2.iloc[:, 2:].values.astype(np.float32)), axis=0)
baseline_data_df1 = torch.tensor(df1_mean).float().to(device).unsqueeze(0)
baseline_data_df2 = torch.tensor(df2_mean).float().to(device).unsqueeze(0)
baseline = torch.cat([baseline_data_df1, baseline_data_df2], dim=1)

def my_baseline_fn(x):
    return baseline

# explainer = shap.DeepExplainer(custom_model, background_data)
explainer = IntegratedGradients(
    model=custom_model,
    n_steps=50,
    baseline_fn=my_baseline_fn
)

results = []

for (idx1, row1), (idx2, row2) in zip(df1.iterrows(), df2.iterrows()):
    id = row1['id']
    task_id = row1['task_id']

    # 첫 두 컬럼(id, task_id)를 제외하고 나머지를 선택
    row1_filtered = row1.iloc[2:]  # iloc[2:]를 사용하여 2번째 이후의 값들만 선택
    
    # 나머지 값을 PyTorch 텐서로 변환
    spec_w2v_x = torch.tensor(row1_filtered.values.astype(np.float32), dtype=torch.float32, device=device)
    spec_w2v_x = spec_w2v_x.unsqueeze(dim=0)

    # 첫 두 컬럼(id, task_id)를 제외하고 나머지를 선택
    row2_filtered = row2.iloc[2:]  # iloc[2:]를 사용하여 2번째 이후의 값들만 선택
    char_x = scaler.transform([row2_filtered.values])
    char_x = torch.tensor(char_x).float().to(device)

    # spec_w2v_x와 char_x를 concat하여 입력 준비
    concat_x = torch.cat([spec_w2v_x, char_x], dim=-1)  

    # 각 클래스(0, 1, 2)에 대해 Integrated Gradients 계산
    for shap_class in [0, 1, 2]:
        attributions = explainer.attribute(inputs=concat_x, targets=torch.tensor([shap_class]))
        # 첫 128개 값은 제거 후 저장
        ig_values = attributions.squeeze().cpu().detach().numpy()[128:]

        # 결과 저장용 딕셔너리 생성
        result = {
            'id': id,
            'task_id': task_id,
            'shap_class': shap_class  # 클래스 인덱스
        }

        # Integrated Gradients 중요도 값을 지정된 컬럼 이름으로 저장
        column_names = [
            "gender", "intelligibility", "var_f0_semitones", "var_f0_hz", "avg_energy", 
            "var_energy", "max_energy", "ddk_rate", "ddk_average", "ddk_std", 
            "ddk_pause_rate", "ddk_pause_average", "ddk_pause_std"
        ]
        result.update({column: ig_values[i] for i, column in enumerate(column_names[:ig_values.shape[0]])})

        # 결과 리스트에 추가
        results.append(result)

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)

# 결과 DataFrame을 CSV 파일로 저장
results_df.to_csv(os.path.join(APP_ROOT, 'logs/test_ig_pnpxai.csv'), index=False)
```

**피쳐 중요도 출력 예시**
아래는 환자번호 `021`이 `/퍼/` 발음 (task_id = 002)을 수행했을 때, 중증(클래스=2) 판정에 기여한 `DDK 피처 중요도`를 확인하는 예시입니다:

```python
print(results_df[
    (results_df['id'] == '021') & (results_df['task_id'] == '002') & (results_df['shap_class'] == 2)
][['ddk_rate', 'ddk_average', 'ddk_std', 'ddk_pause_rate', 'ddk_pause_average', 'ddk_pause_std']])
```

```
   ddk_rate  ddk_average   ddk_std  ddk_pause_rate  ddk_pause_average  \
5  -0.20948     0.009415 -0.009526        0.015898           0.000584   

   ddk_pause_std  
5       0.015935  
```

## 우수/개선 영역 분석

언어치료사가 레이블링한 6가지 피쳐(`ddk_rate`, `ddk_average`, `ddk_std`, `ddk_pause_rate`, `ddk_pause_average`, `ddk_pause_std`)에 대해, IG값을 우수(0) / 보통(1) / 개선(2)으로 매핑하는 예시입니다.

**중증 환자에 기여하는 피쳐 → 0~100 점수화**
* IG값이 정상군 평균 vs 중증군 평균 중 어디에 더 가까운가?” 를 살펴보고 정상군(0) vs 중증군(2)의 평균값(normal_mean[feature], severe_mean[feature]) 사이에서 환자의 실제값(또는 IG값)이 어느 정도 위치하는지를 0~100 범위로 스케일링하고,
    * 70점 이상이면 우수(0)
    * 30점 이하이면 개선(2)
    * 그 사이면 보통(1) 로 범주화합니다.

```python
from sklearn.metrics import mean_squared_error

def cal_scores(df, normal_mean, severe_mean):
    ascending = [feature for feature in normal_mean if normal_mean[feature] < severe_mean[feature]]
    descending = [feature for feature in normal_mean if normal_mean[feature] >= severe_mean[feature]]

    prosody = ['ddk_rate', 'ddk_average', 'ddk_std']
    respiration = ['ddk_pause_rate', 'ddk_pause_average', 'ddk_pause_std']
    vocalization = []

    scores = {}

    # Ascending 방향 피쳐 계산
    for a in ascending:
        a_max = severe_mean[a]
        if df.loc[0, a] < normal_mean[a]:
            scores[a] = 100
        elif df.loc[0, a] > a_max:
            scores[a] = 0
        else:
            score = (df.loc[0, a] - normal_mean[a]) / (a_max - normal_mean[a])
            scores[a] = round((1 - score) * 100, 2)

    # Descending 방향 피쳐 계산
    for d in descending:
        d_min = severe_mean[d]
        if df.loc[0, d] > normal_mean[d]:
            scores[d] = 100
        elif df.loc[0, d] < d_min:
            scores[d] = 0
        else:
            score = (df.loc[0, d] - d_min) / (normal_mean[d] - d_min)
            scores[d] = round(score * 100, 2)

    # Subsystem Scores
    prosody_scores = np.mean([scores[p] for p in prosody if p in scores])
    respiration_scores = np.mean([scores[r] for r in respiration if r in scores])
    vocalization_scores = 0 if not vocalization else np.mean([scores[v] for v in vocalization if v in scores])

    subsystem_score = {'prosody': prosody_scores, 'respiration': respiration_scores, 'vocalization': vocalization_scores}

    return scores, subsystem_score


# 데이터 불러오기
algo = 'ig'
df_labels = pd.read_csv(os.path.join(APP_ROOT, 'data/test_labels.csv'))
shap_df = pd.read_csv(os.path.join(APP_ROOT, f'logs/test_{algo}_pnpxai.csv'))

# 분석할 피쳐 리스트
features = ['ddk_rate', 'ddk_average', 'ddk_std', 'ddk_pause_rate', 'ddk_pause_average', 'ddk_pause_std']

# id를 기준으로 severity 정보를 shap_df에 병합
shap_df = shap_df.merge(df_labels[['id', 'severity']], on='id', how='left')

# shap_class가 2인 데이터 필터링
shap_class_2_df = shap_df[shap_df['shap_class'] == 2]

# normal_mean과 severe_mean 계산
normal_mean = {}
severe_mean = {}

for feature in features:
    # severity가 0인 경우의 shap_class 2 saliency 평균 (Normal)
    normal_mean[feature] = shap_class_2_df[shap_class_2_df['severity'] == 0][feature].mean()
    
    # severity가 2인 경우의 shap_class 2 saliency 평균 (Severe)
    severe_mean[feature] = shap_class_2_df[shap_class_2_df['severity'] == 2][feature].mean()

shap_df_class_2 = shap_df[shap_df['shap_class'] == 2]
df = shap_df_class_2
labels_df = df_labels

# 결과 저장 리스트
predicted_classes = []

predicted_scores = []
for idx, row in df.iterrows():
    id = row['id']
    task_id = row['task_id']
    
    # 각 row에 대해 feature 점수 계산
    sample_df = pd.DataFrame([row], columns=df.columns, index=[0])
    feature_score, _ = cal_scores(sample_df, normal_mean, severe_mean)
    
    # id, task_id별로 feature 점수를 저장
    score_entry = {'id': id, 'task_id': task_id}
    score_entry.update(feature_score)
    predicted_scores.append(score_entry)

# task_id별로 계산된 feature 점수를 DataFrame으로 변환
predicted_df = pd.DataFrame(predicted_scores)

# id별로 feature 점수의 평균을 계산하여 예측값(0, 1, 2)으로 변환
mean_predicted_df = predicted_df.groupby('id', as_index=False)[features].mean()

# 평균 점수를 기준으로 우수/보통/개선 예측
for feature in features:
    mean_predicted_df[f'{feature}_predicted'] = mean_predicted_df[feature].apply(
        lambda score: 0 if score >= 70 else (2 if score <= 30 else 1)
    )

# 결과 확인
# print(mean_predicted_df[[f'{feature}_predicted' for feature in features]])

# 각 feature별로 평균 점수를 기반으로 예측한 `mean_predicted_df`를 `test_labels.csv`와 병합
comparison_df = labels_df.merge(mean_predicted_df[['id'] + [f'{feature}_predicted' for feature in features]], on='id')

# 정확도 계산
accuracy = {}
for feature in features:
    # 실제 값과 예측된 값을 비교하여 정확도 계산
    comparison_df[f'{feature}_correct'] = comparison_df[feature] == comparison_df[f'{feature}_predicted']
    accuracy[feature] = comparison_df[f'{feature}_correct'].mean()
```

**예측 신뢰도 출력 예시**
```python
# 정확도 출력
print("Feature-wise Accuracy compared to test_labels.csv:")
for feature, acc in accuracy.items():
    print(f"Accuracy for {feature}: {acc:.2f}")

# 각 feature별로 MSE 계산
mse = {}
for feature in features:
    mse[feature] = mean_squared_error(comparison_df[feature], comparison_df[f'{feature}_predicted'])

# MSE 출력
print("\nMean Squared Error (MSE) compared to test_labels.csv:")
for feature, error in mse.items():
    print(f"MSE for {feature}: {error:.2f}")
```

```
Feature-wise Accuracy compared to test_labels.csv:
Accuracy for ddk_rate: 0.77
Accuracy for ddk_average: 0.68
Accuracy for ddk_std: 0.61
Accuracy for ddk_pause_rate: 0.59
Accuracy for ddk_pause_average: 0.57
Accuracy for ddk_pause_std: 0.48

Mean Squared Error (MSE) compared to test_labels.csv:
MSE for ddk_rate: 0.23
MSE for ddk_average: 0.32
MSE for ddk_std: 0.45
MSE for ddk_pause_rate: 0.52
MSE for ddk_pause_average: 0.43
MSE for ddk_pause_std: 0.84
```

> **Note**:
> * 평가지표 1: Accuracy
>   * 언어치료사가 지정한 우수/개선 영역과 설명 알고리즘을 통해 계산된 우수/개선 영역 간 일치율
> * 평가지표 2: Mean Squared Error (MSE)
>   * 우수(0) 영역을 개선(2)으로, 또는 개선(2)을 우수(0)으로 잘못 판단한 경우 큰 패널티를 부여
>   * 심각도가 상반되는 영역에서 발생하는 오차의 영향을 강조하여 반영
