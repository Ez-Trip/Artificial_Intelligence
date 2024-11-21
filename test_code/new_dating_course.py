import os
import pandas as pd
import itertools

# 경로 설정
input_folder = 'data'
output_folder = 'data/new_dating_course'
os.makedirs(output_folder, exist_ok=True)  # 저장 폴더 생성

# 역 목록
stations = [
    "공덕역", "광흥창역", "대흥역", "디지털미디어시티역", "마포구청역",
    "마포역", "망원역", "상수역", "서강대역", "신촌역", "아현역",
    "애오개역", "월드컵경기장역", "이대역", "합정역", "홍대입구역"
]

# 카테고리 및 코드 매핑
categories = {
    'A': {'name': '음식점', 'folder': 'place'},
    'B': {'name': '숙박', 'folder': 'accommodation'},
    'C': {'name': '카페', 'folder': 'cafe'},
    'D': {'name': '문화시설', 'folder': 'cultural'},
    'E': {'name': '관광명소', 'folder': 'tourist'}
}

# 세부 카테고리 매핑
detail_categories = {
    'A1': ('음식점', '한식'),
    'A2': ('음식점', '일식'),
    'A3': ('음식점', '양식'),
    'A4': ('음식점', '중식'),
    'B1': ('숙박', ['호텔', '특급호텔']),
    'B2': ('숙박', ['여관,모텔', '모텔', '여관']),
    'B3': ('숙박', ['게스트하우스']),
    'C1': ('카페', None),
    'D1': ('문화시설', None),
    'E1': ('관광명소', None)
}

# 세부 카테고리 추출 함수
def extract_detail_category(category_str, main_category):
    try:
        categories_list = category_str.split(' > ')
        if main_category == '음식점':
            if len(categories_list) >= 2:
                return categories_list[1]  # 두 번째 요소
            else:
                return categories_list[-1]
        elif main_category == '숙박':
            if len(categories_list) >= 3:
                return categories_list[2]  # 세 번째 요소
            else:
                return categories_list[-1]
        else:
            # 다른 카테고리는 마지막 요소 사용
            return categories_list[-1]
    except:
        return category_str

# 랜덤 아이템 선택 함수
def get_random_item(df):
    if df.empty:
        return None
    return df.sample(n=1).to_dict(orient='records')[0]

# 데이터 로드 함수
def load_data(station):
    data = {}
    for cat_code, cat_info in categories.items():
        cat_name = cat_info['name']
        folder_name = cat_info['folder']
        file_path = f"{input_folder}/{folder_name}/place_{cat_name}_{station}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # 누락된 값이나 '알 수 없음' 제외
            df = df.dropna()
            df = df[~df.isin(['알수없음']).any(axis=1)]
            # '카테고리' 컬럼에서 세부 카테고리 추출
            if '카테고리' in df.columns:
                df['세부카테고리'] = df['카테고리'].apply(lambda x: extract_detail_category(x, cat_name))
            data[cat_name] = df
        else:
            data[cat_name] = pd.DataFrame()
    return data

# 가능한 모든 선호도 코드의 조합 생성 함수
def generate_preference_combinations():
    preference_codes = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'C1', 'D1', 'E1']
    combinations = []
    for r in range(1, len(preference_codes)+1):
        # 모든 가능한 길이의 조합 생성
        combos = itertools.permutations(preference_codes, r)
        combinations.extend(combos)
    return combinations

# 데이트 코스 생성 함수
def generate_courses(data, station_name):
    course_list = []
    preference_combinations = generate_preference_combinations()

    for pref_codes in preference_combinations:
        course = {}
        course_code = ''.join(pref_codes)
        used_categories = set()

        for code in pref_codes:
            cat_name, detail = detail_categories.get(code, (None, None))
            if not cat_name:
                continue  # 잘못된 코드인 경우 건너뜀

            used_categories.add(cat_name)
            df = data.get(cat_name, pd.DataFrame())
            if df.empty:
                print(f"{station_name}의 {cat_name} 데이터가 없습니다.")
                break  # 해당 카테고리에 데이터가 없으면 코스 생성 불가

            if detail:
                # 세부 카테고리가 리스트인 경우 처리
                if isinstance(detail, list):
                    df_filtered = df[df['세부카테고리'].isin(detail)]
                else:
                    df_filtered = df[df['세부카테고리'] == detail]
                if df_filtered.empty:
                    print(f"{station_name}의 {cat_name}에서 '{detail}'에 해당하는 데이터가 없습니다.")
                    break
            else:
                df_filtered = df.copy()

            item = get_random_item(df_filtered)
            if item is None:
                print(f"{station_name}의 {cat_name}에서 아이템을 선택할 수 없습니다.")
                break

            course[cat_name] = item

        else:
            # 선호도 코드에 포함되지 않은 카테고리 처리
            for cat_info in categories.values():
                cat_name = cat_info['name']
                if cat_name not in course:
                    df = data.get(cat_name, pd.DataFrame())
                    if df.empty:
                        print(f"{station_name}의 {cat_name} 데이터가 없습니다.")
                        break  # 해당 카테고리에 데이터가 없으면 코스 생성 불가

                    item = get_random_item(df)
                    if item is None:
                        print(f"{station_name}의 {cat_name}에서 아이템을 선택할 수 없습니다.")
                        break

                    course[cat_name] = item

            else:
                # 모든 카테고리에서 아이템을 선택한 경우 코스 추가
                course_data = []
                for cat_info in categories.values():
                    cat_name = cat_info['name']
                    item = course.get(cat_name)
                    if not item:
                        continue  # 아이템이 없으면 건너뜀
                    # 데이터 파일에 존재하는 컬럼들만 사용하고, 빈 값은 제외
                    item_fields = [
                        item.get('name', ''),
                        item.get('세부카테고리', ''),
                        item.get('address', ''),
                        item.get('평점', ''),
                        item.get('카테고리', '')
                    ]
                    # 빈 값을 제외하고 필드 추가
                    item_fields = [field for field in item_fields if field]
                    # 필드들을 공백으로 구분하여 하나의 문자열로 만듦
                    item_str = ' '.join(item_fields)
                    # 각 카테고리별로 대괄호로 묶어서 추가
                    course_data.append(f"[{cat_name}: {item_str}]")

                # 코스코드 추가
                course_data.append(course_code)
                # 필드들을 쉼표로 구분하여 하나의 문자열로 만듦
                course_str = ','.join(course_data)
                course_list.append({'데이트코스': course_str})

    # 데이터프레임으로 변환
    course_df = pd.DataFrame(course_list)
    return course_df

# 메인 함수
def create_dating_course():
    for station in stations:
        data = load_data(station)
        course_df = generate_courses(data, station)

        if course_df.empty:
            print(f"{station}에서 데이트 코스를 생성할 수 없습니다.")
            continue

        # 결과 저장
        station_folder = f"{output_folder}/{station}"
        os.makedirs(station_folder, exist_ok=True)
        output_path = f"{station_folder}/dating_course_{station}.csv"
        course_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"{station}의 데이트 코스를 저장했습니다: {output_path}")

# 코드 실행
if __name__ == "__main__":
    create_dating_course()