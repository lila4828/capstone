import requests
import csv

def cafe_info_save():
    # FastAPI 서버 URL
    url = 'http://localhost:8000/cafe_save/'

    # CSV 파일 경로
    csv_file_path = r"C:\capstone\크롤링, API\naver_cafe.csv"

    # CSV 파일 읽기
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 각 행에서 cafe_info 추출
            cafe_info = {
                "cafeNumber": int(float(row['카페 번호'])),
                "cafeName": row['카페 이름'],
                "cafeUrl": row['URL'],
                "cafeAddress": row['도로명 주소'],
                "cafeLat": float(row['Lat']),  # 위도
                "cafeLon": float(row['Lon']),  # 경도
            }
            
            # FastAPI 서버에 POST 요청 전송
            response = requests.post(url, json=cafe_info)

    # 응답 확인
    if response.status_code == 200:
        print(f"카페 정보가 성공적으로 서버에 전송되었습니다.")
    else:
        print(f"서버에 카페 정보를 저장하는데 실패했습니다. 응답 코드: {response.status_code}")

def add_cafe_images():
    # CSV 파일 경로
    csv_file_path = r'C:\capstone\크롤링, API\\naver_img.csv'

    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        image_list = []
        image_num = 1
        cafe_number = None

        for row in reader:
            if cafe_number is None:
                cafe_number = row['cafeNumber']
                
            if cafe_number == row['cafeNumber']:
                image_info = {
                    "number": image_num,                     # 이미지 번호
                    "imgAddress": row['cafeImg']        # 이미지 주소
                }
                image_num += 1
                image_list.append(image_info)
            else:
                if len(image_list) > 0:
                    url = f'http://localhost:8000/cafe/images/{cafe_number}'    # FastAPI 서버 URL
                    response = requests.post(url, json=image_list)

                    # 응답 확인
                    if response.status_code == 200:
                        print(f"카페 번호 {cafe_number}의 이미지 정보가 성공적으로 서버에 전송되었습니다.")
                    else:
                        print(f"서버에 카페 번호 {cafe_number}의 이미지 정보를 전송하는데 실패했습니다. 응답 코드: {response.status_code}")

                # 다음 카페 번호로 이동
                cafe_number = row['cafeNumber']
                image_list = []
                image_num = 1

        # 마지막 남은 이미지 정보 전송
        if len(image_list) > 0:
            url = f'http://localhost:8000/cafe/images/{cafe_number}'    # FastAPI 서버 URL
            response = requests.post(url, json=image_list)

            # 응답 확인
            if response.status_code == 200:
                print(f"카페 번호 {cafe_number}의 이미지 정보가 성공적으로 서버에 전송되었습니다.")
            else:
                print(f"서버에 카페 번호 {cafe_number}의 이미지 정보를 전송하는데 실패했습니다. 응답 코드: {response.status_code}")

        return response.text  # 또는 다른 값으로 변경 가능

def cafe_review_save():
    # CSV 파일 경로
    csv_file_path = r'C:\capstone\크롤링, API\naver_review.csv'

    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        review_list = []
        review_number = 1
        cafeNum = None

        for row in reader:
            if cafeNum is None:
                cafeNum = row['cafeNumber']

            if cafeNum == row['cafeNumber']:
                review_info = {
                    "number": review_number,                # 리뷰 번호
                    "user": row['nickName'],                # 리뷰자
                    "userImg": row['nameImg'],              # 리뷰자 이미지 주소
                    "text": row['revisit'],                 # 리뷰 내용
                    "imgAddress": row['reviewImg'],         # 리뷰 이미지 주소
                    "date": row['date']                     # 리뷰 날짜
                }
                review_number += 1
                review_list.append(review_info)
            else:
                if len(review_list) > 0:  # review_list에 데이터가 있는 경우에만 요청을 보냅니다.
                    url = f'http://localhost:8000/cafe/reviews/{cafeNum}'    # FastAPI 서버 URL
                    response = requests.post(url, json=review_list)

                    # 응답 확인
                    if response.status_code == 200:
                        print(f"카페 번호 {cafeNum}의 리뷰 정보가 성공적으로 서버에 전송되었습니다.")
                    else:
                        print(f"서버에 카페 번호 {cafeNum}의 리뷰 정보를 전송하는데 실패했습니다. 응답 코드: {response.status_code}")

                # 마지막 카페 번호로 이동하고, 현재 리뷰 정보를 새로운 리스트에 담습니다.
                cafeNum = row['cafeNumber']
                review_list = [review_info]
                review_number = 1

            # 마지막 남은 리뷰 정보 전송
        if len(review_list) > 0:
            url = f'http://localhost:8000/cafe/reviews/{cafeNum}'    # FastAPI 서버 URL
            response = requests.post(url, json=review_list)

            # 응답 확인
            if response.status_code == 200:
                print(f"카페 번호 {cafeNum}의 리뷰 정보가 성공적으로 서버에 전송되었습니다.")
            else:
                print(f"서버에 카페 번호 {cafeNum}의 리뷰 정보를 전송하는데 실패했습니다. 응답 코드: {response.status_code}")

#--------------------------------------------------------------------------------------------------------------------------

# 저장 기능 실행
cafe_info_save()
add_cafe_images()

#리뷰 저장 해야됨
#cafe_review_save()