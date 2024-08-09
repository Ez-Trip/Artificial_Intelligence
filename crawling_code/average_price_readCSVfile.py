import os
import requests
import pandas as pd
import time
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 카카오 REST API 키 설정
KAKAO_API_KEY = 'dd9e94f94c661563a78d155f2aeb7870'

# 데이터 디렉토리 설정
input_dir = 'data/place'
output_dir = 'data/price'

# 카카오 검색 API 호출 함수
def get_place_id(place_name):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

    # 음식점 : FD6 , 문화시설 : CT1 , 관광명소 AT4 , 숙박 : AD5, 카페 : CE7
    params = {"query": place_name, "category_group_code": "FD6"}
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        documents = response.json().get('documents', [])
        if documents:
            place_info = documents[0]
            return place_info['id'], place_info['address_name']
    return None, None

# 평균 가격을 가져오는 함수
def get_average_price(place_id):
    url = f"https://place.map.kakao.com/{place_id}"
    
    # Chrome WebDriver 설정
    options = uc.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    
    driver = uc.Chrome(options=options)
    driver.get(url)
    
    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.list_menu')))
    except Exception as e:
        print(f"Error: {e}")
        try:
            alert = driver.switch_to.alert
            alert.accept()
        except:
            pass
    
    time.sleep(3)  # 페이지 로드 시간을 늘려봅니다
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    menu_items = soup.select('ul.list_menu > li')
    
    prices = []
    for item in menu_items[:5]:  # 상위 5개 메뉴 항목만 사용
        price_elem = item.select_one('em.price_menu')
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            price_value = price_text.replace('가격:', '').strip()  # "가격:" 제거하고 앞뒤 공백 제거
            price_value = price_value.replace(',', '')  # 쉼표 제거
            try:
                prices.append(int(price_value))
            except ValueError:
                continue
    
    driver.quit()
    
    if prices:
        return sum(prices) / len(prices)
    else:
        return None

# 모든 CSV 파일 처리
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        # 파일 경로 설정
        input_csv_path = os.path.join(input_dir, filename)
        
        # 지하철 역 이름 추출
        station_name = filename.split('_')[2].replace('.csv', '')
        output_csv_path = os.path.join(output_dir, f'price_{station_name}.csv')
        
        # CSV 파일 읽기
        data = pd.read_csv(input_csv_path)
        
        # 결과 저장을 위한 리스트 생성
        output_data = []
        
        # 각 음식점에 대해 평균 단가 조회 및 추가
        for index, row in data.iterrows():
            place_name = row['name']
            place_id, place_address = get_place_id(place_name)
            if place_id:
                avg_price = get_average_price(place_id)
                if avg_price:
                    output_data.append({
                        '가게명': place_name,
                        '가게주소': place_address,
                        '평균 단가': int(avg_price)
                    })
                else:
                    output_data.append({
                        '가게명': place_name,
                        '가게주소': place_address,
                        '평균 단가': '정보 없음'
                    })
            else:
                output_data.append({
                    '가게명': place_name,
                    '가게주소': '정보 없음',
                    '평균 단가': '정보 없음'
                })
            
            time.sleep(2)  # API 호출 제한을 피하기 위한 대기 시간 설정
        
        # 리스트를 데이터프레임으로 변환
        output_df = pd.DataFrame(output_data)
        
        # 결과를 새로운 CSV 파일로 저장
        output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"Completed processing file: {filename}. The output file is saved at: {output_csv_path}")
