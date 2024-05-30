import csv
import requests
import os
import glob
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

def get_accommodation_info(place_name):
    # 카카오맵 API 호출을 위한 기본 URL과 API 키 설정
    base_url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    api_key = 'dd9e94f94c661563a78d155f2aeb7870'  # Rest API 키(by yu)

    # API 호출을 위한 파라미터 설정
    params = {
        'query': place_name,  # 검색할 가게 이름
        'category_group_code': 'AD5',  # 숙박시설 카테고리 그룹 코드
        'size': 1  # 검색 결과의 개수 (최대 1개)
    }

    # API 호출
    headers = {'Authorization': f'KakaoAK {api_key}'}
    response = requests.get(base_url, params=params, headers=headers)

    # 응답 확인
    if response.status_code == 200:
        # JSON 형식으로 응답 데이터 가져오기
        data = response.json()
        # 검색 결과에서 첫 번째 가게 정보 추출
        if 'documents' in data and data['documents']:
            place_info = data['documents'][0]
            return place_info
        else:
            print("가게를 찾을 수 없습니다.")
            return None
    else:
        print(f"Error: {response.status_code}")
        return None

def get_ratings(place_url):
    # Chrome 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 브라우저 창을 띄우지 않음
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # ChromeDriver 설정
    chrome_driver_path = ChromeDriverManager().install()
    driver = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)

    # 장소 URL로 이동
    driver.get(place_url)
    time.sleep(4)  # 페이지 로딩 대기

    # 페이지 소스를 BeautifulSoup으로 파싱
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 평점 추출
    rating = None
    rating_element = soup.select_one('.grade_star > em')  # 평점의 CSS 선택자

    if rating_element:
        rating = rating_element.text.strip()

    driver.quit()
    return rating

# CSV 파일을 읽어 가게 이름을 가져오는 함수
def read_place_names_and_addresses_from_csv(file_path):
    place_data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 헤더 읽기
        for row in reader:
            place_data.append(row)  # 모든 열 데이터 추가
    return header, place_data

# CSV 파일에 가게 이름, 주소, 평점을 추가하는 함수
def write_place_data_with_ratings_to_csv(file_path, header, place_data_with_ratings):
    header.append('평점')  # 헤더에 '평점' 추가
    header.append('카테고리')  # 헤더에 '카테고리' 추가
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # 헤더 작성
        writer.writerows(place_data_with_ratings)  # 데이터 작성

# 폴더 내 모든 CSV 파일에 대해 작업 수행
csv_folder_path = 'data/accommodation'
csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))

for csv_file_path in csv_files:
    print(f"Processing file: {csv_file_path}")
    
    # 가게 데이터(이름과 주소)를 읽어옴
    header, place_data = read_place_names_and_addresses_from_csv(csv_file_path)

    # 각 가게 이름에 대해 평점을 가져옴
    place_data_with_ratings = []
    for row in place_data:
        place_name = row[0]
        print(f"Fetching info for {place_name}...")
        place_info = get_accommodation_info(place_name)
        if place_info:
            place_url = f"http://place.map.kakao.com/{place_info['id']}"
            rating = get_ratings(place_url)
            category = place_info.get('category_name', '카테고리 없음')
            row.append(rating)  # 평점을 기존 행에 추가
            row.append(category)  # 카테고리를 기존 행에 추가
            place_data_with_ratings.append(row)
            
            print(f"Rating for {place_name}: {rating}")
            print(f"Category for {place_name}: {category}")
        else:
            print(f"Info not found for {place_name}")
            row.append('평점 없음')  # 평점을 찾을 수 없는 경우 '평점 없음' 추가
            row.append('카테고리 없음')  # 카테고리를 찾을 수 없는 경우 '카테고리 없음' 추가
            place_data_with_ratings.append(row)

    # CSV 파일에 데이터 저장
    write_place_data_with_ratings_to_csv(csv_file_path, header, place_data_with_ratings)
    print(f"Finished processing file: {csv_file_path}")
