import csv
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

def get_place_info(place_name):
    # 카카오맵 API 호출을 위한 기본 URL과 API 키 설정
    base_url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    api_key = 'dd9e94f94c661563a78d155f2aeb7870'  # Rest API 키(by yu)

    # API 호출을 위한 파라미터 설정
    params = {
        'query': place_name,  # 검색할 가게 이름
        'category_group_code': 'FD6',  # 음식점 카테고리 그룹 코드
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

    # 더 많은 리뷰를 로드하기 위해 페이지 스크롤
    for _ in range(5):  # 더 많은 리뷰를 원한다면 범위를 조정
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # 리뷰 로딩 대기

    # 페이지 소스를 BeautifulSoup으로 파싱
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 평점 추출
    ratings = []

    rating_elements = soup.select('.grade_star > em')  # 평점의 CSS 선택자

    for rating_element in rating_elements[:5]:  # 평점 5개만 추출
        ratings.append(rating_element.text.strip())

    driver.quit()
    return ratings

# CSV 파일을 읽어 가게 이름을 가져오는 함수
def read_place_names_from_csv(file_path):
    place_names = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 헤더 건너뛰기
        for row in reader:
            place_names.append(row[0])  # 첫 번째 열에 가게 이름이 있다고 가정
    return place_names

# 가게 이름을 읽어옴
csv_file_path = 'data/place/place_음식점_공덕역.csv'
place_names = read_place_names_from_csv(csv_file_path)

# 각 가게 이름에 대해 평점을 가져옴
for place_name in place_names:
    print(f"Fetching info for {place_name}...")
    place_info = get_place_info(place_name)
    if place_info:
        place_url = f"http://place.map.kakao.com/{place_info['id']}"
        ratings = get_ratings(place_url)
        
        print(f"Ratings for {place_name}:")
        for idx, rating in enumerate(ratings, start=1):
            print(f"Rating {idx}: {rating}")
    else:
        print(f"Info not found for {place_name}")
