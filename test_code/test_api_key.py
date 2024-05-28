import requests

def get_place_info(latitude, longitude):
    # 카카오맵 API 호출을 위한 기본 URL과 API 키 설정
    base_url = 'https://dapi.kakao.com/v2/local/geo/coord2address.json'
    api_key = 'dd9e94f94c661563a78d155f2aeb7870'  # Rest API 키(by yu)

    # API 호출을 위한 파라미터 설정
    params = {
        'x': longitude,  # 경도
        'y': latitude,   # 위도
        'input_coord': 'WGS84',
        'output_coord': 'TM',
    }

    # API 호출
    headers = {'Authorization': f'KakaoAK {api_key}'}
    response = requests.get(base_url, params=params, headers=headers)

    # 응답 확인
    if response.status_code == 200:
        # JSON 형식으로 응답 데이터 가져오기
        data = response.json()
        # 장소 정보 추출
        place_name = data['documents'][0]['place_name']
        return place_name
    else:
        # 에러 발생 시
        print(f"Error: {response.status_code}")
        return None

# 위도와 경도 설정
latitude = 37.5665  # 서울의 위도
longitude = 126.9780  # 서울의 경도

# 장소 정보 가져오기
place_info = get_place_info(latitude, longitude)
if place_info:
    print(f"장소 정보: {place_info}")
else:
    print("장소 정보를 가져오는 데 실패했습니다.")
