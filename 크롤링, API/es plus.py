import requests
import csv
import torch
from torch import nn
import requests
from PIL import Image
from torchvision import models, transforms
import os
import logging
import urllib.request
from urllib.error import HTTPError
import csv
import torch.optim as optim
import warnings
import numpy as np
from tensorflow.keras.models import load_model, model_from_json  # noqa
from tensorflow.keras.applications.resnet50 import preprocess_input  # noqa

# 클래스 레이블 및 번역
classLabels = ["study", "date", "time", "meeting", "emotional", "modern", "cozy", "nature_friendly", "takeout", "retro"]
label_mapping = {
    "study": "공부", 
    "date": "소개팅", 
    "time": "시간",
    "meeting": "회의",
    "emotional": "감성",
    "modern": "현대",
    "cozy": "포근",
    "nature_friendly": "자연친화",
    "takeout": "미니멀",
    "retro": "레트로"
}

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 카페 내부 분류 모델 로드 (Keras 모델)
try:
    # architecture and weights from HDF5
    cafe_ox_model = load_model('model.h5')
except Exception as e:
    warnings.warn(f"에러 HDF5: {e}")
try:
    # architecture from JSON, weights from HDF5
    with open('architecture.json') as f:
        cafe_ox_model = model_from_json(f.read())
    cafe_ox_model.load_weights('weights.h5')
except Exception as e:
    warnings.warn(f"HDF5 에러: {e}")


# 카페 분위기 분류 모델 로드 (PyTorch 모델)
model = models.resnet50(weights='IMAGENET1K_V1')  # Load the pretrained model
num_features = model.fc.in_features

def create_head(num_features, number_classes, dropout_prob=0.5, activation_func=nn.ReLU):
    features_lst = [num_features, num_features//2, num_features//4]
    layers = []
    for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activation_func())
        layers.append(nn.BatchNorm1d(out_f))
        if dropout_prob != 0: layers.append(nn.Dropout(dropout_prob))
    layers.append(nn.Linear(features_lst[-1], number_classes))
    return nn.Sequential(*layers)

model = model.to(device)
top_head = create_head(num_features, len(classLabels)) 
top_head = top_head.to(device)
model.fc = top_head

# 모델 바닥 일부 freezing
for name, child in model.named_children():
    if name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        for param in child.parameters():
            param.requires_grad = False
    else:
        break

# 모델 로드 (PyTorch 체크포인트)
checkpoint = torch.load("./LatestCheckpoint.pt", map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.eval()

# 이미지 전처리 함수
def process_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')  # 이미지가 RGB로 열리는지 확인
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 모델에 맞는 크기로 리사이징
            transforms.ToTensor(),  # 텐서로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)  # 배치 차원 추가
        return img_tensor
    except Exception as e:
        logging.error(f"이미지 처리 실패: {e}")
        return None
    
# 카페 내부 분류 예측 함수 (PyTorch 텐서를 Keras 모델에 맞게 변환)
def predict_cafe(image_path):
    # 이미지 로드 및 전처리
    img = Image.open(image_path).resize((224, 224))
    img_array = preprocess_input(np.array(img)[np.newaxis, :])
    # 모델 예측
    pred_probs = cafe_ox_model.predict(img_array)
    # 예측 결과 해석 파트
    predicted_label = "Cafe" if pred_probs[0][0] > 0.8 else "Non-Cafe"
    #카페인지 아닌지
    return predicted_label

# 카페 분위기 예측 함수
def predict_cafe_list(image_path):
    img_tensor = process_image(image_path)
    
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(output)  # 각 클래스의 확률
        prob_values, class_indices = torch.sort(probs, descending=True)

        predicted_labels = []
        for prob, idx in zip(prob_values[0], class_indices[0]):
            if prob >= 0.7:
                predicted_labels.append(classLabels[idx])
            else:
                break
    
    return predicted_labels, prob_values[0].tolist()

# 태그 번역
def translate_label(labels):
    return [label_mapping.get(label, label) for label in labels]

# 이미지 다운로드 및 예외 처리
def download_image(image_url, img_path):
    try:
        if not os.path.exists(img_path):
            urllib.request.urlretrieve(image_url, img_path)
            logging.info(f"이미지 다운로드 성공: {img_path}")
        else:
            logging.info(f"이미지 이미 존재: {img_path}")
    except HTTPError as e:
        logging.error(f"이미지 다운로드 실패: {e}")

# 서버로 태그 전송
def cafe_tag_save():
    csv_file_path = r'C:\capstone\크롤링, API\naver_cafe.csv'
    path = r"C:\capstone\cafeImg2"

    if not os.path.exists(path):
        os.makedirs(path)

    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            cafeNum = row['카페 번호']
            url = f'http://localhost:8000/get_cafe_images/?cafe_number={cafeNum}'

            try:
                response = requests.get(url)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logging.error(f"카페 {cafeNum} 이미지 요청 실패: {e}")
                continue

            images_data = response.json()
            all_tags = []
            cafe_images = []  # "카페"로 분류된 이미지들

            images_processed = 0
            for image_info in images_data:
                # 이미지 다운로드
                image_number = image_info.get('number')
                image_url = image_info.get('img')
                img_path = os.path.join(path, f"{cafeNum}_{image_number}.jpg")
                
                download_image(image_url, img_path)

                # "카페"로 분류되는지 확인
                predicted_cafe = predict_cafe(img_path)
                if predicted_cafe == 'Cafe':
                    cafe_images.append(img_path)  # "카페"로 분류된 이미지 저장

                images_processed += 1  # 처리된 이미지 수 증가

                # 2개 이미지까지 다운로드 후 예측 시작
                if len(cafe_images) >= 1:  # 최소 1개 이상 "카페"로 분류되면 예측
                    if len(cafe_images) == 5:
                        break  # 5개의 이미지를 다운로드하면 멈춤

            # "카페"로 분류된 이미지가 1개 이상일 경우에만 태그 예측 후 서버에 전송
            if len(cafe_images) >= 1:
                all_tags = []
                for img_path in cafe_images:
                    labels, probs = predict_cafe_list(img_path)
                    translated_tags = translate_label(labels)
                    all_tags.extend(translated_tags)

                # 중복 제거 및 빈 값 필터링
                unique_tags = list(set(all_tags))
                unique_tags = [tag for tag in unique_tags if tag]  # 빈 값은 제외
                print(unique_tags)

                if unique_tags:
                    """
                    post_url = f'http://localhost:8000/cafe/tags/{cafeNum}'
                    try:
                        response = requests.post(post_url, json=unique_tags)
                        response.raise_for_status()
                        logging.info(f"카페 {cafeNum}의 태그가 성공적으로 전송되었습니다.")
                    except requests.exceptions.RequestException as e:
                        logging.error(f"카페 {cafeNum}의 태그 전송 실패: {e}")
                    """
            else:
                logging.info(f"카페 {cafeNum}의 이미지는 '카페'로 분류된 이미지가 없어서 태그를 전송하지 않았습니다.")
