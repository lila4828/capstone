from fastapi import FastAPI
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

@app.get("/get_cafe_info/")         # 카페 정보를 가져온다. - input : 카페 번호
async def get_cafe_info(cafeNum: int):

    query_dsl = {"bool": {"must": [{"match_all": {}}], "filter": [{"match": {"cafeNumber": cafeNum}}]}}

    res = es.search(index="cafe", query=query_dsl, size=1,
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",    # 카페 번호
                                 "hits.hits._source.cafeName",      # 카페 이름
                                 "hits.hits._source.cafeTag",       # 카페 태그
                                 "hits.hits._source.cafeAddress",   # 카페 주소(도로명)
                                 "hits.hits._source.cafeUrl",       # 카페 URL
                                 "hits.hits._source.cafeImg",       # 카페 이미지 주소들
                                 "hits.hits._source.reView"         # 카페 리뷰들
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

    res = es.search(index="cafe", query=query_dsl, size=100,
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",    # 카페 번호
                                 "hits.hits._source.cafeName"       # 카페 이름
                                 ])
    return res

@app.get("/get_cafe_uear/")                    # 태그에 맞는 카페 정보 가져온다. - input : 사용자 입력
async def get_goods_list(search:str):

    query_dsl = {"match_all" : {}}

    res = es.search(index="cafe", query=query_dsl, size=100,
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",    # 카페 번호
                                 "hits.hits._source.cafeName",      # 카페 이름
                                 "hits.hits._source.cafeTag"        # 카페 태그
                                 ])
    return res

class Item(BaseModel):          # 카페 입력 데이터 
    name: str
    price: float

@app.get("")                    # 카페 정보를 저장한다. - input : 카페 정보(카페 번호, 이름, 위경도, 이미지 리스트(번호, 이미지), 리뷰 리스트(빈칸), 태그 리스트(빈칸))
async def get_goods_list():

    query_dsl = {"match_all" : {}}

    res = es.search(index="cafe", query=query_dsl, size=5, 
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",
                                 "hits.hits._source.cafeName"])
    return res

@app.get("")                    # 리뷰 리스트를 추가한다. - input : 카페 번호, 리뷰 리스트(번호, 내용, 이미지)
async def get_goods_list():

    query_dsl = {"match_all" : {}}

    res = es.search(index="cafe", query=query_dsl, size=5, 
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",
                                 "hits.hits._source.cafeName"])
    return res

@app.get("")                    # 형용사를 추가한다. - input : 카페 번호, 형용사 리스트
async def get_goods_list():

    query_dsl = {"match_all" : {}}

    res = es.search(index="cafe", query=query_dsl, size=5, 
                    filter_path=["hits.total,hits.hits._score",
                                 "hits.hits._source.cafeNumber",
                                 "hits.hits._source.cafeName"])
    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
