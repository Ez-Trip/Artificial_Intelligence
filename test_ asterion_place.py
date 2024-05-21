import requests

def get_place_info(place_name):
    # 카카오맵 API 호출을 위한 기본 URL과 API 키 설정
    base_url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    api_key = 'dd9e94f94c661563a78d155f2aeb7870'  # Rest API 키(by yu)

    # API 호출을 위한 파라미터 설정
    params = {
        'query': place_name,  # 검색할 가게 이름
        'category_group_code': 'FD6',  # 음식점 카테고리 그룹 코드 (FD6: 카페)
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

# 가게 이름을 입력받고 가게 정보를 가져오는 예시
place_name = "부산갈매기"  # 사용자가 입력한 가게 이름
place_info = get_place_info(place_name)
if place_info:
    print("가게 정보:", place_info)
