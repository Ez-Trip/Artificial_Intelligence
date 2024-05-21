import requests

def get_restaurants_in_area(area_name, page):
    # 카카오맵 API 호출을 위한 기본 URL과 API 키 설정
    base_url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    api_key = 'dd9e94f94c661563a78d155f2aeb7870'  # Rest API 키(by yu)

    # API 호출을 위한 파라미터 설정
    params = {
        'query': area_name + ' 음식점',  # 검색할 키워드 (특정 구역명 + 음식점)
        'category_group_code': 'FD6',  # 음식점 카테고리 그룹 코드 (FD6: 음식점)
        'size': 15,  # 검색 결과의 개수 (최대 15개)
        'page': 2  # 페이지 번호
    }

    # API 호출
    headers = {'Authorization': f'KakaoAK {api_key}'}
    response = requests.get(base_url, params=params, headers=headers)

    # 응답 확인
    if response.status_code == 200:
        # JSON 형식으로 응답 데이터 가져오기
        data = response.json()
        # 검색 결과에서 음식점 정보 추출
        restaurants = []
        for place in data['documents']:
            restaurant_name = place['place_name']
            restaurant_address = place['address_name']
            restaurants.append({'name': restaurant_name, 'address': restaurant_address})
        return restaurants
    else:
        # 에러 발생 시
        print(f"Error: {response.status_code}")
        return None
    
# 함수를 호출하여 특정 지역의 음식점 목록을 가져오는 코드
area_name = '공덕역'  # 검색할 지역명
page = 4  # 페이지 번호

# 함수 호출
restaurants = get_restaurants_in_area(area_name, page)

# 검색 결과 출력
if restaurants:
    print(f"{area_name} 지역의 음식점 목록:")
    for idx, restaurant in enumerate(restaurants, start=1):
        print(f"{idx}. 이름: {restaurant['name']}, 주소: {restaurant['address']}")
else:
    print("음식점을 찾을 수 없습니다.")
