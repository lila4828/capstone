import requests
import csv

def cafe_info_save():
    # FastAPI 서버 URL
    url = 'http://localhost:8000/cafe_save/'

    # CSV 파일 경로
    csv_file_path = r'C:\Users\djdj4\vscode\capstone\크롤링, API\naver_cafe3.csv'

    # CSV 파일 읽기
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 각 행에서 cafe_info 추출
            cafe_info = {
                "cafeNumber": int(float(row['카페 번호'])),
                "cafeName": row['카페 이름'],
                "cafeUrl": row['URL'],
                "cafeAddress": row['도로명 주소'],
                "cafeLat": float(row['Lat']),  # 위도
                "cafeLon": float(row['Lon']),  # 경도
            }
            
            # FastAPI 서버에 POST 요청 전송
            response = requests.post(url, json=cafe_info)

    # 응답 확인
    if response.status_code == 200:
        print(f"카페 정보가 성공적으로 서버에 전송되었습니다.")
    else:
        print(f"서버에 카페 정보를 저장하는데 실패했습니다. 응답 코드: {response.status_code}")

def cafe_review_save():
    # CSV 파일 경로
    csv_file_path = r'C:\Users\djdj4\vscode\capstone\크롤링, API\naver_review3.csv'

    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        review_list = []
        review_number = 1
        cafeNum = None

        for row in reader:
            if cafeNum is None:
                cafeNum = row['cafeNumber']

            if cafeNum == row['cafeNumber']:
                review_info = {
                    "number": review_number,                # 리뷰 번호
                    "user": row['nickName'],                # 리뷰자
                    "userImg": row['nameImg'],              # 리뷰자 이미지 주소
                    "text": row['revisit'],                 # 리뷰 내용
                    "imgAddress": row['reviewImg'],         # 리뷰 이미지 주소
                    "date": row['date']                     # 리뷰 날짜
                }
                review_number += 1
                review_list.append(review_info)
            else:
                if len(review_list) > 0:  # review_list에 데이터가 있는 경우에만 요청을 보냅니다.
                    url = f'http://localhost:8000/cafe/reviews/{cafeNum}'    # FastAPI 서버 URL
                    response = requests.post(url, json=review_list)

                    # 응답 확인
                    if response.status_code == 200:
                        print(f"카페 번호 {cafeNum}의 리뷰 정보가 성공적으로 서버에 전송되었습니다.")
                    else:
                        print(f"서버에 카페 번호 {cafeNum}의 리뷰 정보를 전송하는데 실패했습니다. 응답 코드: {response.status_code}")

                # 마지막 카페 번호로 이동하고, 현재 리뷰 정보를 새로운 리스트에 담습니다.
                cafeNum = row['cafeNumber']
                review_list = [review_info]
                review_number = 1

            # 마지막 남은 리뷰 정보 전송
        if len(review_list) > 0:
            url = f'http://localhost:8000/cafe/reviews/{cafeNum}'    # FastAPI 서버 URL
            response = requests.post(url, json=review_list)

            # 응답 확인
            if response.status_code == 200:
                print(f"카페 번호 {cafeNum}의 리뷰 정보가 성공적으로 서버에 전송되었습니다.")
            else:
                print(f"서버에 카페 번호 {cafeNum}의 리뷰 정보를 전송하는데 실패했습니다. 응답 코드: {response.status_code}")

def add_cafe_images():
    # CSV 파일 경로
    csv_file_path = r'C:\Users\djdj4\vscode\capstone\크롤링, API\naver_img3.csv'

    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        image_list = []
        image_num = 1
        cafe_number = None

        for row in reader:
            if cafe_number is None:
                cafe_number = row['cafeNumber']
                
            if cafe_number == row['cafeNumber']:
                image_info = {
                    "number": image_num,                     # 이미지 번호
                    "imgAddress": row['cafeImg']        # 이미지 주소
                }
                image_num += 1
                image_list.append(image_info)
            else:
                if len(image_list) > 0:
                    url = f'http://localhost:8000/cafe/images/{cafe_number}'    # FastAPI 서버 URL
                    response = requests.post(url, json=image_list)

                    # 응답 확인
                    if response.status_code == 200:
                        print(f"카페 번호 {cafe_number}의 이미지 정보가 성공적으로 서버에 전송되었습니다.")
                    else:
                        print(f"서버에 카페 번호 {cafe_number}의 이미지 정보를 전송하는데 실패했습니다. 응답 코드: {response.status_code}")

                # 다음 카페 번호로 이동
                cafe_number = row['cafeNumber']
                image_list = []
                image_num = 1

        # 마지막 남은 이미지 정보 전송
        if len(image_list) > 0:
            url = f'http://localhost:8000/cafe/images/{cafe_number}'    # FastAPI 서버 URL
            response = requests.post(url, json=image_list)

            # 응답 확인
            if response.status_code == 200:
                print(f"카페 번호 {cafe_number}의 이미지 정보가 성공적으로 서버에 전송되었습니다.")
            else:
                print(f"서버에 카페 번호 {cafe_number}의 이미지 정보를 전송하는데 실패했습니다. 응답 코드: {response.status_code}")

        return response.text  # 또는 다른 값으로 변경 가능

#cafe_info_save()
#add_cafe_images()
#cafe_review_save()

#--------------------------------------------------------------------------------------------------------------------------

import torch
import requests
from PIL import Image
from torchvision import models, transforms
import os
import logging
import urllib.request
from urllib.error import HTTPError
import csv

# 클래스 레이블 및 번역
classLabels = ["study", "date", "time","meeting","emotional","modern","cozy","nature_freindly", "takeout", "retro"]
label_mapping = {
    "study" : "공부", 
    "date" : "소개팅", 
    "time" : "시간",
    "meeting" : "회의",
    "emotional" : "감성",
    "modern" : "현대",
    "cozy" : "포근",
    "nature_freindly" : "자연친화",
    "takeout" : "미니멀",
    "retro": "레트로"
}

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 카페 내부 분류 모델 로드
import warnings
from tensorflow.keras.models import load_model, model_from_json

try:
    # architecture and weights from HDF5
    cafe_ox_model = load_model('model.h5')
except Exception as e:
    warnings.warn(f"HDF5 모델 로드 실패: {e}")

try:
    # architecture from JSON, weights from HDF5
    with open('architecture.json') as f:
        cafe_ox_model = model_from_json(f.read())
    cafe_ox_model.load_weights('weights.h5')
except Exception as e:
    warnings.warn(f"JSON 아키텍처 또는 HDF5 가중치 로드 실패: {e}")

# 카페 분위기 분류 모델 로드
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(classLabels))  # 출력층 수정
model.load_state_dict(torch.load("./LatestCheckpoint.pt")['model_state_dict'])
model.eval()  # 모델을 평가 모드로 설정

# 이미지 전처리
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

# 카페 예측 함수
def predict_cafe(image_path):
    img_tensor = process_image(image_path)
    
    with torch.no_grad():
        output = cafe_ox_model(img_tensor)
        pred_probs = torch.sigmoid(output)
    
    predicted_label = "Cafe" if pred_probs[0][0] > 0.8 else "Non-Cafe"
    return predicted_label

# 다중 레이블 예측 함수
def predict_cafe_list(image_path):
    img_tensor = process_image(image_path)
    
    with torch.no_grad():
        output = model(img_tensor)
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(output)  # 각 클래스의 확률

        # 확률 값과 클래스 인덱스를 함께 반환
        prob_values, class_indices = torch.sort(probs, descending=True)  # 확률 값 기준으로 내림차순 정렬

        predicted_labels = []
        
        # 임계값을 넘는 클래스들만 선택
        for prob, idx in zip(prob_values[0], class_indices[0]):
            if prob >= 0.8:
                predicted_labels.append(classLabels[idx])
            else:
                break  # 확률이 임계값 이하인 경우 더 이상 선택하지 않음
    
    return predicted_labels

# 태그 번역
def translate_label(labels):
    return [label_mapping.get(label, label) for label in labels]

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
            url = f'http://localhost:8080/get_cafe_images/?cafe_number={cafeNum}'

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
                
                try:
                    urllib.request.urlretrieve(image_url, img_path)
                except HTTPError as e:
                    if e.code == 404:
                        logging.warning(f"HTTP 오류 404: {image_url}을 찾을 수 없습니다.")
                    else:
                        logging.error(f"이미지 다운로드 실패: {e}")
                    continue

                # "카페"로 분류되는지 확인
                predicted_cafe = predict_cafe(img_path)
                if predicted_cafe == 'Cafe':
                    cafe_images.append(img_path)  # "카페"로 분류된 이미지 저장

                images_processed += 1  # 처리된 이미지 수 증가

                # 2개 이미지까지 다운로드 후 예측 시작
                if len(cafe_images) >= 1:  # 최소 1개 이상 "카페"로 분류되면 예측
                    if len(cafe_images) == 2:
                        break  # 2개의 이미지를 다운로드하면 멈춤

            # "카페"로 분류된 이미지가 1개 이상일 경우에만 태그 예측 후 서버에 전송
            if len(cafe_images) >= 1:
                all_tags = []
                for img_path in cafe_images:
                    labels = predict_cafe_list(img_path)
                    translated_tags = translate_label(labels)
                    all_tags.extend(translated_tags)

                # 중복 제거
                unique_tags = list(set(all_tags))

                if unique_tags:
                    post_url = f'http://localhost:8080/cafe/tags/{cafeNum}'
                    try:
                        response = requests.post(post_url, json=unique_tags)
                        response.raise_for_status()
                        logging.info(f"카페 {cafeNum}의 태그가 성공적으로 전송되었습니다.")
                    except requests.exceptions.RequestException as e:
                        logging.error(f"카페 {cafeNum}의 태그 전송 실패: {e}")
            else:
                logging.info(f"카페 {cafeNum}의 이미지는 '카페'로 분류된 이미지가 없어서 태그를 전송하지 않았습니다.")


# 태그 저장 기능 실행
# cafe_tag_save()  # 실제 실행하려면 주석을 해제하세요.
