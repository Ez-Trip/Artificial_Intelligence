import requests
import csv
import os

def get_cultural_facilities_in_area(area_name, page):
    # 카카오맵 API 호출을 위한 기본 URL과 API 키 설정
    base_url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    api_key = 'dd9e94f94c661563a78d155f2aeb7870'  # Rest API 키(by yu)

    # API 호출을 위한 파라미터 설정
    params = {
        'query': area_name + '카페',  # 검색할 키워드 (특정 구역명 + 카페)
        'category_group_code': 'CE7',  # 카페 카테고리 그룹 코드 (CE7: 카페)
        'size': 15,  # 검색 결과의 개수 (최대 15개)
        'page': page  # 페이지 번호
    }

    # API 호출
    headers = {'Authorization': f'KakaoAK {api_key}'}
    response = requests.get(base_url, params=params, headers=headers)

    # 응답 확인
    if response.status_code == 200:
        # JSON 형식으로 응답 데이터 가져오기
        data = response.json()
        # 검색 결과에서 카페 정보 추출
        cafe_list = []
        for place in data['documents']:
            cafe_name = place['place_name']
            cafe_address = place['address_name']
            cafe_list.append({'name': cafe_name, 'address': cafe_address})
        return cafe_list
    else:
        # 에러 발생 시
        print(f"Error: {response.status_code}")
        return None

def append_unique_cafes_to_csv(cafe_list, file_path):
    # 디렉토리 생성 (필요한 경우)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # CSV 파일에서 기존의 카페 정보 읽어오기
    existing_cafes = set()
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_cafes.add((row['name'], row['address']))
    except FileNotFoundError:
        pass

    # 새로운 카페 정보 중에서 기존에 없는 것만 추가하기
    new_cafes = [(cafe['name'], cafe['address']) for cafe in cafe_list
                      if (cafe['name'], cafe['address']) not in existing_cafes]

    # 새로운 카페 정보를 CSV 파일에 추가하기
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'address']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not existing_cafes:
            writer.writeheader()  # 기존 파일에 카페 정보가 없는 경우에만 헤더를 추가
        for name, address in new_cafes:
            writer.writerow({'name': name, 'address': address})

# 특정 구역에서의 카페 정보 가져오기
def get_cafes_for_area(area_name):
    cafes = []
    for page in range(1, 46):
        page_cafes = get_cultural_facilities_in_area(area_name, page)
        if page_cafes:
            cafes.extend(page_cafes)
        else:
            break
    return cafes

# 중복된 카페명 삭제
def remove_duplicate_entries(file_path):
    # 중복을 제거할 카페 정보를 담을 집합
    unique_cafes = set()

    # CSV 파일에서 카페 정보 읽어오기
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            unique_cafes.add((row['name'], row['address']))

    # 중복을 제거한 카페 정보를 다시 CSV 파일에 저장하기
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'address']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for name, address in unique_cafes:
            writer.writerow({'name': name, 'address': address})

# 역 리스트
station_list = [
    '합정역', '홍대입구역','신촌역', '이대역', '아현역', '마포역', '공덕역', 
    '애오개역','월드컵경기장역', '마포구청역', '망원역', '상수역', '광흥창역', 
    '대흥역', '서강대역', '디지털미디어시티역'
]

# 모든 역에 대해 카페 정보 수집 및 저장
for station in station_list:
    cafes = get_cafes_for_area(station)
    new_file_path = ''
    if cafes:
        print(f"{station}에서의 카페 정보:")
        for idx, cafe in enumerate(cafes, start=1):
            print(f"{idx}. 이름: {cafe['name']}, 주소: {cafe['address']}")
        
        # CSV 파일에 새로운 카페 정보 추가
        csv_file_path = f'data/cafe/place_카페_{station}.csv'
        append_unique_cafes_to_csv(cafes, csv_file_path)
        print(f"새로운 카페 정보가 {csv_file_path} 파일에 추가되었습니다.")

        # CSV 파일에서 중복된 내용을 삭제하고 다시 저장
        remove_duplicate_entries(csv_file_path)
        print(f"CSV 파일 '{csv_file_path}'에서 중복된 내용을 삭제하고 다시 저장했습니다.")
    else:
        print(f"{station}에서 카페 정보를 가져오는 데 실패했습니다.")
