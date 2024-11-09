from fastapi import FastAPI, HTTPException
from typing import List, Optional
import uvicorn
from elasticsearch import Elasticsearch
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

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

index_name = "cafe"

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
                                 "hits.hits._source.cafeImg"        # 카페 이미지 주소들
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
                                 "hits.hits._source.cafeImg"        # 카페 이미지 주소들
                                 ])
    return res

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

#--------------------------------------------------------------------------------------------------------

import torch
import requests
from PIL import Image
from torchvision import models, transforms
from urllib.error import HTTPError

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

def cafe_tag_search(img_path):
    tags = []
    
    # "카페"로 분류되는지 확인
    predicted_cafe = predict_cafe(img_path)
    if predicted_cafe == 'Cafe':
        # 카페를 "분위기"로 분류
        labels = predict_cafe_list(img_path)
        translated_tags = translate_label(labels)
        tags.extend(translated_tags)
    else :
        tags.extend(None)
    
    # 리스트를 문자열로 변환
    tags_str = ', '.join(tags)  # 태그를 쉼표로 구분하여 하나의 문자열로 합침
    return tags_str


@app.get("/get_cafe_img_user/")                    # 태그에 맞는 카페 정보 가져온다. - input : 사용자 입력
async def get_cafe_img_user(img_path:str):
    path = r"C:\capstone\userImg\1.jpg"

    #Test path
    Tag = cafe_tag_search(path)
    
    #real path
    #Tag = cafe_tag_search(img_path)

    query_dsl = {"match": {"cafeTag": Tag}}
    res = es.search(index=index_name, query=query_dsl, size=50,
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",    # 카페 번호
                                 "hits.hits._source.cafeName",      # 카페 이름
                                 "hits.hits._source.cafeTag",       # 카페 태그
                                 "hits.hits._source.cafePoint",     # 위경도
                                 "hits.hits._source.cafeImg"        # 카페 이미지 주소들
                                 ])
    return res