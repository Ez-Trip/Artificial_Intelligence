import requests
import pandas as pd
import time
import undetected_chromedriver as uc
from bs4 import BeautifulSoup

# 카카오 REST API 키 설정
KAKAO_API_KEY = 'dd9e94f94c661563a78d155f2aeb7870'

# CSV 파일 경로 설정
input_csv_path = 'data/place/place_음식점_홍대입구역.csv'
output_csv_path = 'data/price/price_홍대입구역.csv'

# CSV 파일 읽기
data = pd.read_csv(input_csv_path)

# 카카오 검색 API 호출 함수
def get_place_id(place_name):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": place_name, "category_group_code": "FD6"}
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        documents = response.json().get('documents', [])
        if documents:
            place_info = documents[0]
            return place_info['id']
    return None

# 평균 가격을 가져오는 함수
def get_average_price(place_id):
    url = f"https://place.map.kakao.com/{place_id}"
    
    # Chrome WebDriver 설정
    options = uc.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    driver = uc.Chrome(options=options)
    driver.get(url)
    time.sleep(2)  # 페이지 로딩 대기

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    price_elements = soup.select('.price')
    
    prices = []
    for elem in price_elements[:5]:  # 상위 5개 가격만 사용
        price_text = elem.get_text(strip=True).replace(',', '').replace('원', '')
        try:
            prices.append(int(price_text))
        except ValueError:
            continue
    
    driver.quit()
    
    if prices:
        return sum(prices) / len(prices)
    else:
        return None

# 결과 저장을 위한 데이터프레임 생성
output_data = pd.DataFrame(columns=data.columns.tolist() + ['평균 단가'])

# 각 음식점에 대해 평균 단가 조회 및 추가
for index, row in data.iterrows():
    place_name = row['name']
    place_id = get_place_id(place_name)
    if place_id:
        avg_price = get_average_price(place_id)
        if avg_price:
            row['평균 단가'] = avg_price
        else:
            row['평균 단가'] = '정보 없음'
    else:
        row['평균 단가'] = '정보 없음'
    
    output_data = pd.concat([output_data, pd.DataFrame([row])], ignore_index=True)
    time.sleep(1)  # API 호출 제한을 피하기 위한 대기 시간 설정

# 결과를 새로운 CSV 파일로 저장
output_data.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"Completed. The output file is saved at: {output_csv_path}")
