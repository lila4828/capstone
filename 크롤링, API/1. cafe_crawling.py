# import package
import pandas as pd
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# run webdriver
driver = webdriver.Chrome()
keyword = '목포시 카페'
url = f'https://map.naver.com/p/search/{keyword}'
driver.get(url)
action = ActionChains(driver)

naver_res = pd.DataFrame(columns=['카페 번호','카페 이름','도로명 주소', 'URL'])
path = r"C:\Users\djdj4\vscode\capstone"
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

def crawling_main():
    global naver_res
    addr_list = []
    cafeName_list = []
    restaurant_id_list = []
    cafe_url_list = []
    img_list = []
    
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
    naver_res.to_excel('./naver_crawling2.xlsx', engine='openpyxl')


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

excel_file = 'naver_crawling2.xlsx'  #xlsx 파일 불러와서
csv_file = 'naver_crawling2.csv'     #csv 파일 변환

df = pd.read_excel(excel_file)

# DataFrame을 CSV 파일로 저장하기 (UTF-8 인코딩)
df.to_csv(csv_file, index=False, encoding='utf-8-sig')

print("파일 변환이 끝났습니다.")