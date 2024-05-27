from selenium.webdriver.common.by import By
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from openpyxl import Workbook
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import requests
import urllib.request
import re
import pandas as pd

def remove_special_characters(text):
    # 정규 표현식을 사용하여 특수 문자 제거
    return re.sub(r'[\\/:*?"<>|]', '', text)

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
list_sheet.append(['cafeNumber','nickName', 'nameImg', 'date', 'revisit', 'reviewImg'])

path = r"C:\Users\djdj4\vscode\capstone"

# 각각의 카페의 고유 번호 csv에서 가져오기
csv_file = 'naver_cafe2.csv'
data = pd.read_csv(csv_file)

try:
    driver = webdriver.Chrome()
    driver.implicitly_wait(30)

    for index, row in data.iterrows():
        value = row.iloc[1]

        # url
        if not pd.isnull(value):
            url = f'https://m.place.naver.com/restaurant/{int(value)}/review/visitor?entry=ple&reviewSort=recent'
        else:
            continue
        
        res = driver.get(url)

        # Pagedown
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
        # Start crawling/scraping!
        try:
            #리뷰 개수도 가져와서 300개 이상이면 스크롤을 제한적으로 넘기기 해야겠다
            #일단은 리뷰의 개수는 한번만 실행하는 것으로 최대 20개 까지 가져오는 것으로 실행
            i = 0
            try:
                while (i<1):        # 1번만 실행
                    driver.find_element(By.XPATH, '//*[@id="app-root"]/div/div/div/div[6]/div[2]/div[3]/div[2]/div/a').click()
                    time.sleep(0.4)
                    i += 1
            except Exception as e:
                print('finish')

            time.sleep(20)
            html = driver.page_source
            bs = BeautifulSoup(html, 'lxml')
            reviews = bs.select('li.owAeM')

            for r in reviews:
                nickName = r.select_one('div > div.RKXdJ > a.j1rOp > div.qgLL3 > span')                         # 리뷰 사용자
                nameImg = r.select_one('div > div.RKXdJ > a.RJ26d > div > img')                                 # 리뷰 사용자 사진
                date = r.select_one('div > div.jxc2b > div.D40bm > span:nth-child(1) > span:nth-child(3)')      # 리뷰 날짜
                revisit = r.select_one('div > div.vg7Fp > a > span.zPfVt')                                      # 리뷰 내용
                reviewImg = r.select_one('div > div.VAvOk > div > div > div > div > a > img')                   # 리뷰 이미지

                # exception handling
                # 가져온 리뷰자 이름을 text만 가져와서 저장
                nickName = remove_special_characters(nickName.text) if nickName else ''

                # 가져온 리뷰자 사진의 url에서 다운받아 저장
                if nameImg:
                    nameImgSrc = nameImg.get("src")                                            # img에서 다운받을 주소 src
                    if nameImgSrc:
                        nameImgPath = path + r"\nameImg\\" + nickName +'.jpg'
                        urllib.request.urlretrieve(nameImgSrc, nameImgPath)
                    else:
                        nameImgPath = ''
                else:
                    nameImgPath = ''

                # 리뷰 날짜 text 가져와 저장
                date = date.text if date else ''

                # 리뷰 글자 text 가져와 저장
                revisit = revisit.text if revisit else ''

                # 가져온 리뷰 사진의 url에서 다운받아 저장
                reviewImgSrc = ''
                if reviewImg:
                    reviewImgSrc = reviewImg.get("src")  # 리뷰 이미지의 src 속성 값 가져오기
                    if reviewImgSrc:
                        reviewImgPath = path + r"\reviewImg\\" + nickName + '_review.jpg'
                        urllib.request.urlretrieve(reviewImgSrc, reviewImgPath)
                    else:
                        reviewImgPath = ''

                time.sleep(0.06)

                list_sheet.append([value, nickName, nameImgPath, date, revisit, reviewImgPath]) 
                time.sleep(0.06)
            
        except Exception as e:
            print(e)
            # Save the file(temp)
            file_name = 'naver_review_exception_' + str(value) + '.xlsx'
            xlsx.save(file_name)

finally:
    driver.quit()
    # Save the file
    file_name = './naver_review3.xlsx'
    xlsx.save(file_name)


excel_file = 'naver_review3.xlsx'  #xlsx 파일 불러와서
csv_file = 'naver_review3.csv'     #csv 파일 변환

df = pd.read_excel(excel_file)

# DataFrame을 CSV 파일로 저장하기 (UTF-8 인코딩)
df.to_csv(csv_file, index=False, encoding='utf-8-sig')

print("파일 변환이 끝났습니다.")