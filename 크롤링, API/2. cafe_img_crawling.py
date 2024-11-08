from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from openpyxl import Workbook
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import requests
import urllib.request
import pandas as pd

# BS4 setting for secondary access
session = requests.Session()
headers = {"User-Agent": "user value"}

retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504])

session.mount('http://', HTTPAdapter(max_retries=retries))

# New xlsx file
xlsx = Workbook()
list_sheet = xlsx.active
list_sheet.append(['cafeNumber','cafeImg'])

path = r"C:\capstone\크롤링, API"

# 각각의 카페의 고유 번호 csv에서 가져오기
csv_file = r'C:\capstone\크롤링, API\naver_cafe.csv'
data = pd.read_csv(csv_file)

try:
    driver = webdriver.Chrome()
    driver.implicitly_wait(30)

    for index, row in data.iterrows():
        value = row.iloc[0]
        img_list = []

        # url
        if not pd.isnull(value):
            url = f'https://m.place.naver.com/restaurant/{int(value)}/photo?entry=ple&reviewSort=recent&filterType=%EB%82%B4%EB%B6%80'
        else:
            continue
        
        res = driver.get(url)
        
        # Start crawling
        try:
            time.sleep(5)
            html = driver.page_source
            soup = BeautifulSoup(html, 'lxml')

            try:
                img_element = soup.find_all('div', class_='wzrbN')                         
                imgs = ''
            
                for i, img in enumerate(img_element, 1):  # 두 번째 매개변수를 1로 설정하여 인덱스를 1부터 시작
                    img_select = img.select_one('a > img')
                    img_src = img_select.get("src")
                    
                    if img_src:
                        img_path = path + r"\cafeImg\\" + str(value) + '_' + str(i) +'.jpg'   # 이미지 경로 생성
                        #urllib.request.urlretrieve(img_src, img_path)                 # 이미지 다운로드
                    else:
                        img_path = ''

                    list_sheet.append([value, img_src])
            except Exception as e:
                print(e)

            time.sleep(0.06)
            
        except Exception as e:
            print(e)
            # Save the file(temp)
            file_name = 'naver_img_exception_' + str(value) + '.xlsx'
            xlsx.save(file_name)

finally:
    driver.quit()
    # Save the file
    file_name = './naver_img.xlsx'
    xlsx.save(file_name)


excel_file = 'naver_img.xlsx'  #xlsx 파일 불러와서
csv_file = 'naver_img.csv'     #csv 파일 변환

df = pd.read_excel(excel_file)

# DataFrame을 CSV 파일로 저장하기 (UTF-8 인코딩)
df.to_csv(csv_file, index=False, encoding='utf-8-sig')

print("파일 변환이 끝났습니다.")