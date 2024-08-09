import os
import pandas as pd

# 데이터 디렉토리 설정
place_dir = 'data/place'
price_dir = 'data/price'

# 모든 price_파일 처리
for filename in os.listdir(price_dir):
    if filename.startswith('price_') and filename.endswith('.csv'):
        # 지하철 역 이름 추출
        station_name = filename.split('_')[1].replace('.csv', '')
        
        # 관련된 place_ 파일 경로 설정
        place_filename = f'place_음식점_{station_name}.csv'
        place_csv_path = os.path.join(place_dir, place_filename)
        
        # price 파일과 place 파일 읽기
        price_csv_path = os.path.join(price_dir, filename)
        price_data = pd.read_csv(price_csv_path)
        place_data = pd.read_csv(place_csv_path)
        
        # 평균 단가 정보를 name 컬럼을 기준으로 place 파일에 병합
        merged_data = pd.merge(place_data, price_data[['가게명', '평균 단가']], left_on='name', right_on='가게명', how='left')
        
        # 가게명 컬럼 제거 및 컬럼 순서 조정
        merged_data = merged_data.drop(columns=['가게명'])
        merged_data = merged_data[['name', 'address', '평점', '카테고리', '평균 단가']]
        
        # 결과를 원래의 place 파일에 덮어쓰기
        merged_data.to_csv(place_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"Completed updating file: {place_filename} with average prices from {filename}")
