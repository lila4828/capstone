from selenium.webdriver.common.by import By
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from openpyxl import Workbook
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import requests
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# BS4 setting for secondary access
session = requests.Session()
headers = {"User-Agent": "user value"}

retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504])

session.mount('http://', HTTPAdapter(max_retries=retries))

def cafe() :
    driver = webdriver.Chrome()
    keyword = '목포시 카페'
    url = f'https://map.naver.com/p/search/{keyword}'
    driver.get(url)
    action = ActionChains(driver)

    naver_res = pd.DataFrame(columns=['카페 번호','카페 이름','도로명 주소', 'URL'])
    last_name = ''

    def search_iframe():
        driver.switch_to.default_content()
        driver.switch_to.frame("searchIframe")

    def entry_iframe():
        driver.switch_to.default_content()
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH, '//*[@id="entryIframe"]')))

        for i in range(5):
            time.sleep(1)
            
            try:
                driver.switch_to.frame(driver.find_element(By.XPATH, '//*[@id="entryIframe"]'))
                break
            except:
                pass

    def chk_names():
        search_iframe()
        elem = driver.find_elements(By.XPATH, f'//*[@id="_pcmap_list_scroll_container"]/ul/li/div[1]/a[1]/div/div/span[1]')
        name_list = [e.text for e in elem]

        return elem, name_list

    def go_to_next_page():
        try:
            next_page_button = driver.find_element(By.XPATH, '//*[@id="app-root"]/div/div[2]/div[2]/a[7]')
            if "disabled" in next_page_button.get_attribute("class"):
                print("No more pages available. Exiting...")
                return False  # 다음 페이지가 없음을 반환하여 루프를 종료
            next_page_button.click()
            return True  # 다음 페이지로 이동
        except Exception as e:
            print(f"An error occurred while clicking the next page: {e}")
            return False  # 오류 발생으로 인해 다음 페이지로 이동 실패
        
    def crawling_main():
        global naver_res
        addr_list = []
        cafeName_list = []
        restaurant_id_list = []
        cafe_url_list = []
        
        for e in elem:
            e.click()
            entry_iframe()
            soup = BeautifulSoup(driver.page_source, 'html.parser')
        
            # append data

            #카페 이름
            try:
                cafeName_list.append(soup.select('span.Fc1rA')[0].text)             
            except:
                cafeName_list.append(float('nan'))

            #도로명 주소
            try:
                addr_list.append(soup.select('span.LDgIH')[0].text)                 
            except:
                addr_list.append(float('nan'))
                
            #카페 번호
            try:
                url_content = soup.find('meta', {'id': 'og:url'}).get('content')    

                restaurant_id = url_content.split("/restaurant/")[1].split("/")[0]

                restaurant_id_list.append(restaurant_id)
            except:
                restaurant_id_list.append(float('nan'))

            # 카페 URL 생성
            try:
                cafe_url = f'https://m.place.naver.com/restaurant/{restaurant_id}/home'
                cafe_url_list.append(cafe_url)
            except:
                cafe_url_list.append(float('nan'))
            
            search_iframe()
        
        naver_temp = pd.DataFrame([restaurant_id_list, cafeName_list, addr_list, cafe_url_list], index=naver_res.columns).T
        naver_res = pd.concat([naver_res, naver_temp])
        naver_res.to_excel('./naver_crawling.xlsx', engine='openpyxl')

    while True:  # 무한 루프
        time.sleep(2)  # 페이지가 완전히 로드될 때까지 대기
        search_iframe()
        elem, name_list = chk_names()
        if last_name == name_list[-1]:
            break  # 마지막 페이지까지 스크롤한 경우 루프 종료
        
        while True:
            # auto scroll
            driver.execute_script("arguments[0].scrollIntoView(true);", elem[-1])
            elem, name_list = chk_names()
            if last_name == name_list[-1]:
                break  # 스크롤이 마지막까지 이루어진 경우 다음 페이지로 이동
            last_name = name_list[-1]

        crawling_main()
        
        if not go_to_next_page():  # 다음 페이지로 이동 실패한 경우 루프 종료
            break

    excel_file = 'naver_crawling.xlsx'  #xlsx 파일 불러와서
    csv_file = 'naver_crawling.csv'     #csv 파일 변환

    df = pd.read_excel(excel_file)

    # DataFrame을 CSV 파일로 저장하기 (UTF-8 인코딩)
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    print("파일 변환이 끝났습니다.")

def img() :
    # New xlsx file
    xlsx = Workbook()
    list_sheet = xlsx.active
    list_sheet.append(['cafeNumber','cafeImg'])

    # 각각의 카페의 고유 번호 csv에서 가져오기
    csv_file = r'C:\capstone\크롤링, API\naver_cafe.csv'
    data = pd.read_csv(csv_file)

    try:
        driver = webdriver.Chrome()
        driver.implicitly_wait(30)

        for index, row in data.iterrows():
            value = row.iloc[0]

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
                            """
                            img_path = path + r"\cafeImg\\" + str(value) + '_' + str(i) +'.jpg'   # 이미지 경로 생성
                            urllib.request.urlretrieve(img_src, img_path)                 # 이미지 다운로드
                            """
                        else:
                            img_src = ''

                        list_sheet.append([value, img_src])
                except Exception as e:
                    print(e)

                time.sleep(0.06)
                
            except Exception as e:
                print(e)
        
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

def review() :
    def remove_special_characters(text):        # 정규 표현식을 사용하여 특수 문자 제거
        return re.sub(r'[\\/:*?"<>|]', '', text)

    # New xlsx file
    xlsx = Workbook()
    list_sheet = xlsx.active
    list_sheet.append(['cafeNumber','nickName', 'nameImg', 'date', 'revisit', 'reviewImg'])

    # 각각의 카페의 고유 번호 csv에서 가져오기
    csv_file = r'C:\capstone\크롤링, API\naver_cafe.csv'
    data = pd.read_csv(csv_file)

    try:
        driver = webdriver.Chrome()
        driver.implicitly_wait(30)

        for index, row in data.iterrows():
            value = row.iloc[0]

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

                # 여기 select 부분이 문제가 생긴듯
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
                            """
                            nameImgPath = path + r"\nameImg\\" + nickName +'.jpg'
                            urllib.request.urlretrieve(nameImgSrc, nameImgPath)
                            """
                        else:
                            nameImgSrc = ''
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
                            """
                            reviewImgPath = path + r"\reviewImg\\" + nickName + '_review.jpg'
                            urllib.request.urlretrieve(reviewImgSrc, reviewImgPath)
                            """
                        else:
                            reviewImgSrc = ''
                    else:
                        reviewImgPath = ''

                    time.sleep(0.06)

                    list_sheet.append([value, nickName, nameImgSrc, date, revisit, reviewImgSrc]) 
                    time.sleep(0.06)
                
            except Exception as e:
                print(e)

    finally:
        driver.quit()
        # Save the file
        file_name = './naver_review.xlsx'
        xlsx.save(file_name)

    excel_file = 'naver_review.xlsx'  #xlsx 파일 불러와서
    csv_file = 'naver_review.csv'     #csv 파일 변환

    df = pd.read_excel(excel_file)

    # DataFrame을 CSV 파일로 저장하기 (UTF-8 인코딩)
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    print("파일 변환이 끝났습니다.")

#cafe()
#img()
review()