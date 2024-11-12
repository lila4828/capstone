from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from openpyxl import Workbook
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import requests
import urllib.request
import pandas as pd

# BS4 설정을 위한 세션 생성
session = requests.Session()
headers = {"User-Agent": "user value"}

retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504])

session.mount('http://', HTTPAdapter(max_retries=retries))

# 새로운 엑셀 파일 생성
xlsx = Workbook()
list_sheet = xlsx.active
list_sheet.append(['cafeNumber','cafeImg'])

path = r"C:\capstone\크롤링, API"

# 각 카페의 고유 번호를 CSV에서 불러오기
csv_file = r'C:\capstone\크롤링, API\naver_cafe.csv'
data = pd.read_csv(csv_file)

try:
    driver = webdriver.Chrome()
    driver.implicitly_wait(30)

    for index, row in data.iterrows():
        value = row.iloc[0]
        img_list = []

        # 첫 번째 URL 설정
        if not pd.isnull(value):
            first_url = f'https://m.place.naver.com/restaurant/{int(value)}/photo?entry=ple&reviewSort=recent&filterType=%EB%82%B4%EB%B6%80'
        else:
            continue

        # 크롤링 시작 - 첫 번째 URL
        try:
            driver.get(first_url)  # 첫 번째 URL로 이동
            time.sleep(5)
            html = driver.page_source
            soup = BeautifulSoup(html, 'lxml')

            img_element = soup.find_all('div', class_='wzrbN')

            # 이미지가 없을 경우 두 번째 URL로 이동
            if not img_element:
                print(f"첫 번째 URL에 이미지가 없습니다: {first_url}")
                second_url = f'https://m.place.naver.com/restaurant/{int(value)}/photo'
                driver.get(second_url)  # 두 번째 URL로 이동
                time.sleep(5)
                html = driver.page_source
                soup = BeautifulSoup(html, 'lxml')
                img_element = soup.find_all('div', class_='wzrbN')

                if not img_element:
                    print(f"이미지가 존재하지 않습니다: {value}")
                    continue  # 이미지가 없으면 다음 카페로 이동

            # 이미지 처리
            for i, img in enumerate(img_element, 1):
                img_select = img.select_one('a > img')
                img_src = img_select.get("src")
                
                if img_src:
                    img_path = path + r"\cafeImg\\" + str(value) + '_' + str(i) +'.jpg'   # 이미지 경로 생성
                    # urllib.request.urlretrieve(img_src, img_path)                 # 이미지 다운로드
                else:
                    img_path = ''

                list_sheet.append([value, img_src])
        except Exception as e:
            print(e)
            # 임시 파일 저장
            file_name = 'naver_img_exception_' + str(value) + '.xlsx'
            xlsx.save(file_name)

finally:
    driver.quit()
    # 파일 저장
    file_name = './naver_img1.xlsx'
    xlsx.save(file_name)


excel_file = 'naver_img1.xlsx'  # 엑셀 파일 불러오기
csv_file = 'naver_img1.csv'     # CSV 파일로 변환

df = pd.read_excel(excel_file)

# DataFrame을 CSV 파일로 저장 (UTF-8 인코딩)
df.to_csv(csv_file, index=False, encoding='utf-8-sig')

print("파일 변환이 끝났습니다.")