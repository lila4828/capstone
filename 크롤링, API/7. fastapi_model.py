from fastapi import FastAPI, HTTPException
from typing import List, Optional
import uvicorn
from elasticsearch import Elasticsearch
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms
import logging
from urllib.error import HTTPError
import torch.optim as optim
import warnings
import numpy as np
from tensorflow.keras.models import load_model, model_from_json
from keras.applications.resnet50 import preprocess_input

app = FastAPI()

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

# 모델 로딩 함수
def load_models():
    global cafe_ox_model, model
    try:
        # Keras 모델 로드
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

    # PyTorch 모델 로드 (ResNet50)
    model = models.resnet50(weights='IMAGENET1K_V1')  # Load the pretrained model
    num_features = model.fc.in_features

    def create_head(num_features, number_classes, dropout_prob=0.5, activation_func=nn.ReLU):
        features_lst = [num_features, num_features // 2, num_features // 4]
        layers = []
        for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(activation_func())
            layers.append(nn.BatchNorm1d(out_f))
            if dropout_prob != 0:
                layers.append(nn.Dropout(dropout_prob))
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
    predicted_label = "Cafe" if pred_probs[0][0] > 0.5 else "Non-Cafe"
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

def cafe_tag_search(img_path):
    tags = []
    
    try:
        # "카페"로 분류되는지 확인
        predicted_cafe = predict_cafe(img_path)
        if predicted_cafe == 'Cafe':
            # 카페를 "분위기"로 분류
            labels, prob_values = predict_cafe_list(img_path)
            translated_tags = translate_label(labels)
            tags.extend(translated_tags)
        else:
            tags.append('Non-Cafe')  # 카페가 아닌 경우 추가
        
        # 태그가 없을 경우 빈 문자열로 처리
        if not tags:
            tags.append('No tags found')  # 기본값 설정
    except Exception as e:
        logging.error(f"Error in cafe_tag_search: {e}")
        tags.append('Error processing image')  # 에러 발생 시 기본 태그
    
    # 리스트를 문자열로 변환
    tags_str = ', '.join(tags)  # 태그를 쉼표로 구분하여 하나의 문자열로 합침
    return tags_str


# 서버 시작 시 모델을 로드하는 이벤트 처리
def on_startup():
    load_models()  # 모델 로드 작업
    print("Models are loaded successfully!")

# add_event_handler로 서버 시작 시 실행할 함수 등록
app.add_event_handler("startup", on_startup)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

es = Elasticsearch( hosts = [{'host': 'my-deployment-5d8a0f.es.us-east-1.aws.found.io', 'port': 9243, 'scheme' : "https"}],
                request_timeout=300, max_retries=10, retry_on_timeout=True, 
                basic_auth=('elasticuser', 'hy%^&2022') )

class CafeImage(BaseModel):
    number: int
    imgAddress: str

class Review(BaseModel):
    number: int
    user: str
    userImg: str
    text: Optional[str] = ""
    imgAddress: Optional[str] = ""
    date: str

class CafeInfo(BaseModel):      # 카페 입력 데이터
    cafeNumber: int                             # 카페 번호
    cafeName: str                               # 카페 이름
    cafeUrl: str                                # 카페 URL
    cafeAddress: str                            # 카페 도로명 주소
    cafeLat: float                              # 위도
    cafeLon: float                              # 경도
    cafeImg: Optional[List[CafeImage]] = []     # 카페 전경 이미지들
    review: Optional[List[Review]] = []         # 리뷰 리스트들
    cafeTag: Optional[List[str]] = []           # 형용사들

index_name = "cafe2"

@app.get("/get_cafe_info/")         # 카페 정보를 가져온다. - input : 카페 번호
async def get_cafe_info(cafeNum:int):

    query_dsl = {"bool": {"must": [{"match_all": {}}], "filter": [{"match": {"cafeNumber": cafeNum}}]}}

    res = es.search(index=index_name, query=query_dsl, size=1,
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",    # 카페 번호
                                 "hits.hits._source.cafeName",      # 카페 이름
                                 "hits.hits._source.cafeTag",       # 카페 태그
                                 "hits.hits._source.cafeAddress",   # 카페 주소(도로명)
                                 "hits.hits._source.cafePoint",     # 위경도
                                 "hits.hits._source.cafeUrl",       # 카페 URL
                                 "hits.hits._source.cafeImg",       # 카페 이미지 주소들
                                 "hits.hits._source.review"         # 카페 리뷰들
                                 ])
    return res

@app.get("/get_cafe_point/")   # 위치 중심으로 거리가 가까운거와 형용사 맞는거 가져온다. - intput : 사용자 입력, 위도(lat), 경도(lon)
async def get_cafe_user(search:str, lat: float, lon: float):

    query_dsl = {"bool":
                    {"must": 
                        [
                        { "geo_distance": {
                                "distance": "3000m",  # 3km 거리
                                "cafePoint": {
                                "lat": lat,         # 기준 위도
                                "lon": lon          # 기준 경도
                                }
                            }
                        },
                        {"match": { "cafeTag": search } } # 검색할 태그
                        ]
                    }
                }

    res = es.search(index=index_name, query=query_dsl, size=50,
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",    # 카페 번호
                                 "hits.hits._source.cafeName",      # 카페 이름
                                 "hits.hits._source.cafeTag",       # 카페 태그
                                 "hits.hits._source.cafePoint",     # 위경도
                                 "hits.hits._source.cafeImg",        # 카페 이미지 주소들
                                 "hits.hits._source.cafeAddress"       # 카페 주소들
                                 ])
    return res

@app.get("/get_cafe_point_sort/")   # 태그에 맞는 카페 정보를 거리순으로 정렬하여 가져온다. - intput : 사용자 입력, 위도(lat), 경도(lon)
async def get_cafe_user(search:str, lat: float, lon: float):

    query_dsl = {           # 형용사만 맞으면 가까운거 부터 정렬해서 보내주고
        "query": {
            "match": {
                "cafeTag": search
            }
        },
        "sort": [
            {
                "_geo_distance": {
                    "location": {  # 'location' 필드는 위경도 값이 저장된 필드를 나타냅니다.
                        "lat": lat,
                        "lon": lon
                    },
                    "order": "asc",  # 오름차순 정렬 (가까운 거리부터)
                    "unit": "3000m"  # 거리 단위 (킬로미터)
                }
            }
        ]
    }

    res = es.search(index=index_name, query=query_dsl, size=50,
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",    # 카페 번호
                                 "hits.hits._source.cafeName",      # 카페 이름
                                 "hits.hits._source.cafeTag",       # 카페 태그
                                 "hits.hits._source.cafePoint",     # 위경도
                                 "hits.hits._source.cafeImg"        # 카페 이미지 주소들
                                 ])
    return res

@app.get("/get_cafe_user/")                    # 태그에 맞는 카페 정보 가져온다. - input : 사용자 입력
async def get_cafe_user(search:str):

    query_dsl = {"match": {"cafeTag": search}}
 
    res = es.search(index=index_name, query=query_dsl, size=50,
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",    # 카페 번호
                                 "hits.hits._source.cafeName",      # 카페 이름
                                 "hits.hits._source.cafeTag",       # 카페 태그
                                 "hits.hits._source.cafePoint",     # 위경도
                                 "hits.hits._source.cafeImg",       # 카페 이미지 주소들
                                 "hits.hits._source.cafeAddress"       # 카페 주소들
                                 ])
    return res

@app.get("/get_cafe_user_name/")                    # 이름으로 카페 정보 가져온다. - input : 사용자 입력
async def get_cafe_user_name(search:str):

    query_dsl = {"match": {"cafeName": search}}
 
    res = es.search(index=index_name, query=query_dsl, size=10,
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",    # 카페 번호
                                 "hits.hits._source.cafeName",      # 카페 이름
                                 "hits.hits._source.cafeTag",       # 카페 태그
                                 "hits.hits._source.cafePoint",     # 위경도
                                 "hits.hits._source.cafeImg",        # 카페 이미지 주소들
                                 "hits.hits._source.cafeAddress"       # 카페 주소들
                                 ])
    return res

@app.get("/get_combined_cafe_info/")  # 새로운 API 엔드포인트
async def get_combined_cafe_info(search: str):
    # 첫 번째 검색: 이름으로 검색
    query_name = {"match": {"cafeName": search}}
    res_name = es.search(index=index_name, query=query_name, size=10,
                         filter_path=["hits.total", "hits.hits._score",
                                      "hits.hits._source.cafeNumber",
                                      "hits.hits._source.cafeName",
                                      "hits.hits._source.cafeTag",
                                      "hits.hits._source.cafePoint",
                                      "hits.hits._source.cafeImg",
                                      "hits.hits._source.cafeAddress"])

    # 두 번째 검색: 태그로 검색
    query_tag = {"match": {"cafeTag": search}}
    res_tag = es.search(index=index_name, query=query_tag, size=50,
                        filter_path=["hits.total", "hits.hits._score",
                                     "hits.hits._source.cafeNumber",
                                     "hits.hits._source.cafeName",
                                     "hits.hits._source.cafeTag",
                                     "hits.hits._source.cafePoint",
                                     "hits.hits._source.cafeImg",
                                     "hits.hits._source.cafeAddress"])

    # 기존 형식 유지: 병합된 hits.total과 hits.hits 생성
    combined_hits = []
    seen_cafe_numbers = set()

    # 첫 번째 결과 추가
    for hit in res_name.get("hits", {}).get("hits", []):
        cafe_number = hit["_source"]["cafeNumber"]
        if cafe_number not in seen_cafe_numbers:
            seen_cafe_numbers.add(cafe_number)
            combined_hits.append(hit)

    # 두 번째 결과 추가 (중복 제거)
    for hit in res_tag.get("hits", {}).get("hits", []):
        cafe_number = hit["_source"]["cafeNumber"]
        if cafe_number not in seen_cafe_numbers:
            seen_cafe_numbers.add(cafe_number)
            combined_hits.append(hit)

    # 총 결과 수
    total_hits = len(combined_hits)

    # 결과 반환
    return {
        "hits": {
            "total": {"value": total_hits},
            "hits": combined_hits
        }
    }

@app.get("/get_cafe_images/")          # 카페 이미지를 가져온다. - input : 카페 번호
async def get_cafe_images(cafe_number: int):
    # 카페 번호를 기준으로 해당 카페의 문서를 Elasticsearch에서 가져옴
    cafe_document = es.search(index=index_name, body={"query": {"match": {"cafeNumber": cafe_number}}})
    if not cafe_document['hits']['hits']:
        raise HTTPException(status_code=404, detail="카페가 존재하지 않습니다.")

    # 카페 번호에 해당하는 문서의 이미지 가져오기
    cafe_images = cafe_document['hits']['hits'][0]['_source'].get('cafeImg', [])

    return cafe_images

@app.post("/cafe_save/")            # 카페 정보를 저장한다. - input : 카페 정보(카페 번호, 이름, 주소, URL, 위경도, 이미지 리스트(번호, 이미지), 리뷰 리스트(빈칸), 형용사 리스트(빈칸))
async def cafe_save(cafe_info: CafeInfo):
    # Elasticsearch에 저장할 문서 생성
    cafe_document = {
        "cafeNumber": cafe_info.cafeNumber,
        "cafeName": cafe_info.cafeName,
        "cafeUrl": cafe_info.cafeUrl,
        "cafeAddress": cafe_info.cafeAddress,
        "cafePoint": {
            "lat": cafe_info.cafeLat,  # 위도
            "lon": cafe_info.cafeLon  # 경도
        },
        "cafeImg": [{"number": img.number, "img": img.imgAddress} for img in cafe_info.cafeImg],
        "review": [{"number": review.number, "text":review.text, "img": review.imgAddress} for review in cafe_info.review],
        "cafeTag": [cafe_info.cafeTag]
    }
    # Elasticsearch에 문서 색인
    res = es.index(index=index_name, body=cafe_document)

    return res

@app.post("/cafe/images/{cafe_number}")    # 카페 이미지를 추가한다 - input : 카페 번호, 이미지 리스트(이미지 번호, 이미지주소)
async def add_cafe_images(cafe_number: int, cafe_images: List[CafeImage]):
    # 카페 번호를 기준으로 해당 카페의 문서를 Elasticsearch에서 가져옴
    cafe_document = es.search(index=index_name, body={"query": {"match": {"cafeNumber": cafe_number}}})
    if not cafe_document['hits']['hits']:
        raise HTTPException(status_code=404, detail="카페가 존재하지 않습니다.")

    # 카페 번호에 해당하는 문서의 ID 가져오기
    cafe_id = cafe_document['hits']['hits'][0]['_id']

    for imgs in cafe_images:
        script = {
            "script": {
                "source": "ctx._source.cafeImg.add(params.img)",
                "params": {
                    "img": {
                        "number": imgs.number,
                        "img": imgs.imgAddress
                    }
                }
            }
        }

        # Elasticsearch의 _update API를 사용하여 리뷰 추가
        res = es.update(index=index_name, id=cafe_id, body=script)
    
    return res

@app.post("/cafe/reviews/{cafe_number}")      # 리뷰 리스트를 추가한다. - input : 카페 번호, 리뷰 리스트(번호, 내용, 이미지주소)
async def add_reviews(cafe_number: int, reviews: List[Review]):
    # 카페 번호를 기준으로 해당 카페의 문서를 Elasticsearch에서 가져옴
    cafe_document = es.search(index=index_name, body={"query": {"match": {"cafeNumber": cafe_number}}})
    if not cafe_document['hits']['hits']:
        raise HTTPException(status_code=404, detail="카페가 존재하지 않습니다.")

    # 카페 번호에 해당하는 문서의 ID 가져오기
    cafe_id = cafe_document['hits']['hits'][0]['_id']

    for review in reviews:
        script = {
            "script": {
                "source": "ctx._source.review.add(params.review)",
                "params": {
                    "review": {
                        "number": review.number,
                        "user":review.user,
                        "userImg":review.userImg,
                        "text": review.text,
                        "img": review.imgAddress,
                        "date":review.date
                    }
                }
            }
        }

        # Elasticsearch의 _update API를 사용하여 리뷰 추가
        result = es.update(index=index_name, id=cafe_id, body=script)

    return result
               
@app.post("/cafe/tags/{cafe_number}")             
async def add_tags(cafe_number: int, tag: List[str]):
    # 카페 번호를 기준으로 해당 카페의 문서를 Elasticsearch에서 가져옴
    cafe_document = es.search(index=index_name, body={"query": {"match": {"cafeNumber": cafe_number}}})
    if not cafe_document['hits']['hits']:
        raise HTTPException(status_code=404, detail="카페가 존재하지 않습니다.")

    # 카페 번호에 해당하는 문서의 ID 가져오기
    cafe_id = cafe_document['hits']['hits'][0]['_id']

    res = []

    # 형용사를 추가하는 스크립트 준비
    script = {
        "script": {
            "source": "ctx._source.cafeTag.addAll(params.cafeTag)",
            "params": {
                "cafeTag": tag
            }
        }
    }

    # Elasticsearch의 _update API를 사용하여 태그 추가
    result = es.update(index=index_name, id=cafe_id, body=script)
    res.append(result)  # 결과를 리스트에 추가
    return res

@app.get("/get_cafe_img_user/")                    # 이미지에 맞는 카페 정보 가져온다. - input : 사용자 이미지 위치
async def get_cafe_img_user(img_name:str):
    path = r"C:\capstone\userImg\\" + img_name   # 이미지 이름

    #태그 가져오기
    Tag = cafe_tag_search(path)
    
    return Tag

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
