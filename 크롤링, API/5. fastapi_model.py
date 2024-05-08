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

@app.get("/get_cafe_info/")         # 카페 정보를 가져온다. - input : 카페 번호
async def get_cafe_info(cafeNum: int):

    query_dsl = {"bool": {"must": [{"match_all": {}}], "filter": [{"match": {"cafeNumber": cafeNum}}]}}

    res = es.search(index="cafe2", query=query_dsl, size=1,
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

@app.get("/get_cafe_point/")      # 주변 카페 정보(카페 번호, 카페 이름)를 가져온다.  - intput 위도(lat), 경도(lon)
async def get_cafe_point(lat: float, lon: float):
    query_dsl = { "bool": {
                            "must": [ {
                                "geo_distance": { 
                                    "distance": "30000m",   # 3km 안에
                                    "cafePoint": {
                                        "lat": lat,   # 기준 위도
                                        "lon": lon   # 기준 경도
                                    }
                                }
                            } ] } }

    res = es.search(index="cafe2", query=query_dsl, size=20,
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",    # 카페 번호
                                 "hits.hits._source.cafeName"       # 카페 이름
                                 "hits.hits._source.cafeImg",       # 카페 이미지 주소들
                                 ])
    return res

@app.get("/get_cafe_uear/")                    # 태그에 맞는 카페 정보 가져온다. - input : 사용자 입력
async def get_cafe_uear(search:str):

    query_dsl = {"match": {"cafeTag": search}}

    res = es.search(index="cafe2", query=query_dsl, size=50,
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",    # 카페 번호
                                 "hits.hits._source.cafeName",      # 카페 이름
                                 "hits.hits._source.cafeTag"        # 카페 태그
                                 "hits.hits._source.cafeImg",       # 카페 이미지 주소들
                                 ])
    return res

@app.get("/get_cafe_images/")          # 카페 이미지를 가져온다. - input : 카페 번호
async def get_cafe_images(cafe_number: int):
    # 카페 번호를 기준으로 해당 카페의 문서를 Elasticsearch에서 가져옴
    cafe_document = es.search(index='cafe2', body={"query": {"match": {"cafeNumber": cafe_number}}})
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
    res = es.index(index='cafe2', body=cafe_document)

    return res

@app.post("/cafe/images/{cafe_number}")    # 카페 이미지를 추가한다 - input : 카페 번호, 이미지 리스트(이미지 번호, 이미지주소)
async def add_cafe_images(cafe_number: int, cafe_images: List[CafeImage]):
    # 카페 번호를 기준으로 해당 카페의 문서를 Elasticsearch에서 가져옴
    cafe_document = es.search(index='cafe2', body={"query": {"match": {"cafeNumber": cafe_number}}})
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
        res = es.update(index='cafe2', id=cafe_id, body=script)
    
    return res

@app.post("/cafe/reviews/{cafe_number}")      # 리뷰 리스트를 추가한다. - input : 카페 번호, 리뷰 리스트(번호, 내용, 이미지주소)
async def add_reviews(cafe_number: int, reviews: List[Review]):
    # 카페 번호를 기준으로 해당 카페의 문서를 Elasticsearch에서 가져옴
    cafe_document = es.search(index='cafe2', body={"query": {"match": {"cafeNumber": cafe_number}}})
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
        result = es.update(index='cafe2', id=cafe_id, body=script)

    return result
               
@app.post("/cafe/tags/{cafe_number}")             
async def add_tags(cafe_number: int, tag: List[str]):
    # 카페 번호를 기준으로 해당 카페의 문서를 Elasticsearch에서 가져옴
    cafe_document = es.search(index='cafe2', body={"query": {"match": {"cafeNumber": cafe_number}}})
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
    result = es.update(index='cafe2', id=cafe_id, body=script)
    res.append(result)  # 결과를 리스트에 추가
    return res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
