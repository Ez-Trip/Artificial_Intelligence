import requests
import csv
import os

def get_parking_in_area(area_name, page):
    # 카카오맵 API 호출을 위한 기본 URL과 API 키 설정
    base_url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    api_key = 'dd9e94f94c661563a78d155f2aeb7870'  # Rest API 키(by yu)

    # API 호출을 위한 파라미터 설정
    params = {
        'query': area_name + '주차장',  # 검색할 키워드 (특정 구역명 + 주차장)
        'category_group_code': 'PK6',  # 주차장 카테고리 그룹 코드 (PK6: 주차장)
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
        # 검색 결과에서 주차장 정보 추출
        parkings = []
        for place in data['documents']:
            parking_name = place['place_name']
            parking_address = place['address_name']
            parkings.append({'name': parking_name, 'address': parking_address})
        return parkings
    else:
        # 에러 발생 시
        print(f"Error: {response.status_code}")
        return None

def append_unique_parkings_to_csv(parkings, file_path):
    # 디렉토리 생성 (필요한 경우)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # CSV 파일에서 기존의 주차장 정보 읽어오기
    existing_parkings = set()
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_parkings.add((row['name'], row['address']))
    except FileNotFoundError:
        pass

    # 새로운 주차장 정보 중에서 기존에 없는 것만 추가하기
    new_parkings = [(parking['name'], parking['address']) for parking in parkings
                    if (parking['name'], parking['address']) not in existing_parkings]

    # 새로운 주차장 정보를 CSV 파일에 추가하기
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'address']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not existing_parkings:
            writer.writeheader()  # 기존 파일에 주차장 정보가 없는 경우에만 헤더를 추가
        for name, address in new_parkings:
            writer.writerow({'name': name, 'address': address})

# 특정 구역에서의 주차장 정보 가져오기
def get_parkings_for_area(area_name):
    parkings = []
    for page in range(1, 46):
        page_parkings = get_parking_in_area(area_name, page)
        if page_parkings:
            parkings.extend(page_parkings)
        else:
            break
    return parkings

# 중복된 주차장명 삭제
def remove_duplicate_entries(file_path):
    # 중복을 제거할 주차장 정보를 담을 집합
    unique_parkings = set()

    # CSV 파일에서 주차장 정보 읽어오기
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            unique_parkings.add((row['name'], row['address']))

    # 중복을 제거한 주차장 정보를 다시 CSV 파일에 저장하기
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'address']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for name, address in unique_parkings:
            writer.writerow({'name': name, 'address': address})

# 역 리스트
station_list = [
    '합정역', '홍대입구역','신촌역', '이대역', '아현역', '마포역', '공덕역', 
    '애오개역','월드컵경기장역', '마포구청역', '망원역', '상수역', '광흥창역', 
    '대흥역', '서강대역', '디지털미디어시티역'
]

# 모든 역에 대해 주차장 정보 수집 및 저장
for station in station_list:
    parkings = get_parkings_for_area(station)
    new_file_path = ''
    if parkings:
        print(f"{station}에서의 주차장 정보:")
        for idx, parking in enumerate(parkings, start=1):
            print(f"{idx}. 이름: {parking['name']}, 주소: {parking['address']}")
        
        # CSV 파일에 새로운 주차장 정보 추가
        csv_file_path = f'data/parking/place_주차장_{station}.csv'
        append_unique_parkings_to_csv(parkings, csv_file_path)
        print(f"새로운 주차장 정보가 {csv_file_path} 파일에 추가되었습니다.")

        # csv 파일 이름 변수 따로 저장
        new_file_path = csv_file_path

        # CSV 파일에서 중복된 내용을 삭제하고 다시 저장
        remove_duplicate_entries(new_file_path)
        print(f"CSV 파일 '{new_file_path}'에서 중복된 내용을 삭제하고 다시 저장했습니다.")
    else:
        print(f"{station}에서 주차장 정보를 가져오는 데 실패했습니다.")
