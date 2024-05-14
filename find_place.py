import requests
import csv

def get_restaurants_in_area(area_name, page):
    # 카카오맵 API 호출을 위한 기본 URL과 API 키 설정
    base_url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    api_key = 'dd9e94f94c661563a78d155f2aeb7870'  # Rest API 키(by yu)

    # API 호출을 위한 파라미터 설정
    params = {
        'query': area_name + ' 음식점',  # 검색할 키워드 (특정 구역명 + 음식점)
        'category_group_code': 'FD6',  # 음식점 카테고리 그룹 코드 (FD6: 음식점)
        'size' : 15,  # 검색 결과의 개수 (최대 15개)
        'page': page  # 페이지 번호
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

def append_unique_restaurants_to_csv(restaurants, file_path):
    # CSV 파일에서 기존의 음식점 정보 읽어오기
    existing_restaurants = set()
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_restaurants.add((row['name'], row['address']))
    except FileNotFoundError:
        pass

    # 새로운 음식점 정보 중에서 기존에 없는 것만 추가하기
    new_restaurants = [(restaurant['name'], restaurant['address']) for restaurant in restaurants
                       if (restaurant['name'], restaurant['address']) not in existing_restaurants]

    # 새로운 음식점 정보를 CSV 파일에 추가하기
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'address']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not existing_restaurants:
            writer.writeheader()  # 기존 파일에 음식점 정보가 없는 경우에만 헤더를 추가
        for name, address in new_restaurants:
            writer.writerow({'name': name, 'address': address})

# 특정 구역에서의 음식점 정보 가져오기
def get_restaurants_for_area(area_name):
    restaurants = []
    for page in range(1, 46):
        page_restaurants = get_restaurants_in_area(area_name, page)
        if page_restaurants:
            restaurants.extend(page_restaurants)
        else:
            break
    return restaurants

area_name = '합정역'
restaurants = get_restaurants_for_area(area_name)
new_file_path = ''
if restaurants:
    print(f"{area_name}에서의 음식점 정보:")
    for idx, restaurant in enumerate(restaurants, start=1):
        print(f"{idx}. 이름: {restaurant['name']}, 주소: {restaurant['address']}")
    
    # CSV 파일에 새로운 음식점 정보 추가
    csv_file_path = f'data/place_음식점_{area_name}.csv'
    append_unique_restaurants_to_csv(restaurants, csv_file_path)
    print(f"새로운 음식점 정보가 {csv_file_path} 파일에 추가되었습니다.")

    # csv 파일 이름 변수 따로 저장
    new_file_path = csv_file_path

else:
    print("음식점 정보를 가져오는 데 실패했습니다.")


# 중복된 가게명 삭제
def remove_duplicate_entries(file_path):
    # 중복을 제거할 음식점 정보를 담을 집합
    unique_restaurants = set()

    # CSV 파일에서 음식점 정보 읽어오기
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            unique_restaurants.add((row['name'], row['address']))

    # 중복을 제거한 음식점 정보를 다시 CSV 파일에 저장하기
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'address']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for name, address in unique_restaurants:
            writer.writerow({'name': name, 'address': address})

# CSV 파일에서 중복된 내용을 삭제하고 다시 저장
remove_duplicate_entries(new_file_path)
print(f"CSV 파일 '{new_file_path}'에서 중복된 내용을 삭제하고 다시 저장했습니다.")
