from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

def get_reviews_and_ratings(place_url):
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

    # 리뷰와 평점 추출
    reviews = []
    ratings = []

    review_elements = soup.select('.txt_comment')  # 리뷰 텍스트의 CSS 선택자
    rating_elements = soup.select('.grade_star > em')  # 평점의 CSS 선택자

    for review_element, rating_element in zip(review_elements, rating_elements):
        reviews.append(review_element.text.strip())
        ratings.append(rating_element.text.strip())

    driver.quit()
    return reviews, ratings

# 특정 장소의 URL
place_url = "http://place.map.kakao.com/8374003"

# 리뷰와 평점 가져오기
reviews, ratings = get_reviews_and_ratings(place_url)

# 리뷰와 평점 출력
for idx, (review, rating) in enumerate(zip(reviews, ratings), start=1):
    print(f"Review {idx}: {review}")
    print(f"Rating {idx}: {rating}")
