import pandas as pd
from geopy.geocoders import Nominatim

# 입력된 주소를 위도와 경도로 변환하는 함수
def address_to_latlon(address: str) -> tuple:
    geolocator = Nominatim(user_agent="geoapiuse")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# CSV 파일에서 카페 정보를 읽어와서 위도와 경도를 추가하여 새로운 CSV 파일에 저장하는 함수
def add_latlon_to_cafe_info(input_csv_path: str, output_csv_path: str):
    # CSV 파일 읽기
    df = pd.read_csv(input_csv_path)
    
    # 기존 주소 열에 복사
    df['기존 주소'] = df['도로명 주소']
    
    # 도로명 주소 단순화
    df['도로명 주소'] = df['도로명 주소'].apply(lambda x: " ".join(x.split()[:4]))
    
    # 주소를 이용하여 위도와 경도 추출하여 데이터프레임에 추가
    df['위도'], df['경도'] = zip(*df['도로명 주소'].apply(address_to_latlon))
    
    # 새로운 CSV 파일로 저장
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print("카페 정보가 성공적으로 CSV 파일에 저장되었습니다.")

# 함수 호출하여 카페 정보를 처리하고 새로운 CSV 파일에 저장
input_csv_path = 'naver_crawling.csv'
output_csv_path = 'naver_cafe_geocode.csv'
add_latlon_to_cafe_info(input_csv_path, output_csv_path)