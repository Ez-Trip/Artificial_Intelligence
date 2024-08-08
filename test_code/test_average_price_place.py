# 특정 가계 메뉴 5개의 평균금액 구하는 코드

import time
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 특정 음식점 ID 설정
place_id = '1217230294'  # 홍대인파스타의 카카오맵 ID

# 음식점 상세 정보 URL 설정
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
    avg_price = sum(prices) / len(prices)
    avg_price = int(avg_price)
    print(f"Average Price: {avg_price} 원")
else:
    print("가격 정보를 추출할 수 없습니다.")
