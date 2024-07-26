import os
import glob
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from random import randint

# 데이터 전처리 함수들
def read_and_preprocess_csv_files(folder_path, is_cafe=False, is_tourist=False, is_cultural=False, is_accommodation=False):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    all_data = []

    for file_path in csv_files:
        station_name = os.path.basename(file_path).split('_')[-1].replace('.csv', '')
        df = pd.read_csv(file_path)
        
        if is_cafe:
            df['카테고리'] = '카페'
        elif is_tourist:
            df['카테고리'] = '관광명소'
        elif is_cultural:
            df['카테고리'] = '문화시설'
        elif is_accommodation:
            df['카테고리'] = df['카테고리'].apply(preprocess_accommodation_category)
        else:
            df['카테고리'] = df['카테고리'].apply(preprocess_category)
        
        df['평점'] = df['평점'].apply(preprocess_rating) if '평점' in df.columns else 0.0
        df['역이름'] = station_name
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

def preprocess_category(category):
    if pd.isna(category):
        return '기타'
    category = category.replace('"', '')
    return category.split(' > ')[1] if ' > ' in category else category

def preprocess_accommodation_category(category):
    if pd.isna(category):
        return '기타'
    category = category.replace('"', '')
    if '여관,모텔' in category:
        return '모텔'
    elif '호텔' in category:
        return '호텔'
    elif '게스트하우스' in category:
        return '게스트하우스'
    else:
        return '기타 숙박'

def preprocess_rating(rating):
    if pd.isna(rating):
        return 0.0
    rating = re.findall(r'\d+\.\d+', str(rating))
    return float(rating[0]) if rating else 0.0

# 데이터 로드 및 전처리
csv_folder_path_place = 'data/place'
data_place = read_and_preprocess_csv_files(csv_folder_path_place)

csv_folder_path_cafe = 'data/cafe'
data_cafe = read_and_preprocess_csv_files(csv_folder_path_cafe, is_cafe=True)

csv_folder_path_tourist = 'data/tourist'
data_tourist = read_and_preprocess_csv_files(csv_folder_path_tourist, is_tourist=True)

csv_folder_path_cultural = 'data/cultural'
data_cultural = read_and_preprocess_csv_files(csv_folder_path_cultural, is_cultural=True)

csv_folder_path_accommodation = 'data/accommodation'
data_accommodation = read_and_preprocess_csv_files(csv_folder_path_accommodation, is_accommodation=True)

# 데이터 결합
data = pd.concat([data_place, data_cafe, data_tourist, data_cultural, data_accommodation], ignore_index=True)

# 레이블 인코딩
label_encoders = {}
for column in ['카테고리', '역이름']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 특성 및 레이블 설정
features = data[['카테고리', '역이름']]
labels = data['평점']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

class PlaceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features.iloc[idx].values, dtype=torch.float32),
            'label': torch.tensor(self.labels.iloc[idx], dtype=torch.float32)
        }

train_dataset = PlaceDataset(X_train, y_train)
test_dataset = PlaceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# RNN 모델 정의
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_size = 2
hidden_size = 128
output_size = 1
num_epochs = 50
learning_rate = 0.001

model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        features = batch['features'].unsqueeze(1)
        labels = batch['label']

        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

# 모델 평가
model.eval()
with torch.no_grad():
    total_loss = 0
    for batch in test_loader:
        features = batch['features'].unsqueeze(1)
        labels = batch['label']
        
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        total_loss += loss.item()
    
    print(f'Test Loss: {total_loss / len(test_loader):.4f}')

def transform_label(label_encoder, value):
    try:
        return label_encoder.transform([value])[0]
    except ValueError:
        # 레이블이 존재하지 않는 경우, 기본값을 반환하거나 처리
        return -1

def recommend_top_places(model, data, station_name, category):
    station_id = transform_label(label_encoders['역이름'], station_name)
    category_id = transform_label(label_encoders['카테고리'], category)
    
    if station_id == -1 or category_id == -1:
        print(f"Error: Station name or category not found. Station: {station_name}, Category: {category}")
        return pd.DataFrame()  # 빈 데이터프레임 반환

    filtered_data = data[(data['역이름'] == station_id) & (data['카테고리'] == category_id) & (data['평점'] > 2.5)]
    
    if filtered_data.empty:
        print(f"No data found for station: {station_name}, category: {category}")
        return pd.DataFrame()
    
    features = torch.tensor(filtered_data[['카테고리', '역이름']].values, dtype=torch.float32)
    features = features.unsqueeze(1)
    
    with torch.no_grad():
        predictions = model(features).squeeze()
    
    filtered_data = filtered_data.copy()
    filtered_data['predicted_rating'] = predictions
    top_places = filtered_data.sort_values(by='predicted_rating', ascending=False)
    
    return top_places

def recommend_cafes(model, data, station_name):
    station_id = transform_label(label_encoders['역이름'], station_name)
    cafe_category_id = transform_label(label_encoders['카테고리'], '카페')
    
    if station_id == -1 or cafe_category_id == -1:
        print(f"Error: Station name or category not found. Station: {station_name}, Category: 카페")
        return pd.DataFrame()  # 빈 데이터프레임 반환

    filtered_data = data[(data['역이름'] == station_id) & (data['카테고리'] == cafe_category_id) & (data['평점'] > 2.5)]
    
    if filtered_data.empty:
        print(f"No data found for station: {station_name}")
        return pd.DataFrame()
    
    features = torch.tensor(filtered_data[['카테고리', '역이름']].values, dtype=torch.float32)
    features = features.unsqueeze(1)
    
    with torch.no_grad():
        predictions = model(features).squeeze()
    
    filtered_data = filtered_data.copy()
    filtered_data['predicted_rating'] = predictions
    top_cafes = filtered_data.sort_values(by='predicted_rating', ascending=False)
    
    return top_cafes

def recommend_accommodations(model, data, station_name, accommodation_type):
    station_id = transform_label(label_encoders['역이름'], station_name)
    accommodation_category_id = transform_label(label_encoders['카테고리'], accommodation_type)
    
    if station_id == -1 or accommodation_category_id == -1:
        print(f"Error: Station name or accommodation type not found. Station: {station_name}, Accommodation Type: {accommodation_type}")
        return pd.DataFrame()  # 빈 데이터프레임 반환

    filtered_data = data[(data['역이름'] == station_id) & (data['카테고리'] == accommodation_category_id) & (data['평점'] > 2.5)]
    
    if filtered_data.empty:
        print(f"No data found for station: {station_name}, accommodation type: {accommodation_type}")
        return pd.DataFrame()
    
    features = torch.tensor(filtered_data[['카테고리', '역이름']].values, dtype=torch.float32)
    features = features.unsqueeze(1)
    
    with torch.no_grad():
        predictions = model(features).squeeze()
    
    filtered_data = filtered_data.copy()
    filtered_data['predicted_rating'] = predictions
    top_accommodations = filtered_data.sort_values(by='predicted_rating', ascending=False)
    
    return top_accommodations

def recommend_tourist_and_cultural(data, station_name, category):
    station_id = label_encoders['역이름'].transform([station_name])[0]
    category_id = label_encoders['카테고리'].transform([category])[0]
    
    filtered_data = data[(data['역이름'] == station_id) & (data['카테고리'] == category_id)]
    return filtered_data

station_list = [
    "공덕역", "광흥창역", "대흥역", "디지털미디어시티역", "마포구청역", 
    "마포역", "망원역", "상수역", "서강대역", "신촌역", "아현역", 
    "애오개역", "월드컵경기장역", "이대역", "합정역", "홍대입구역"
]

category_list = ["한식", "일식", "중식", "양식", "기타"]

accommodation_list = ["모텔", "호텔", "게스트하우스"]

# 데이터 코스 생성 및 저장
def create_and_save_random_courses(model, data, station_name, num_courses, csv_path):
    all_courses = []
    for _ in range(num_courses):
        course_data = {
            '음식점 이름': '', '음식점 카테고리': '', '음식점 평점': '', '음식점 주소': '',
            '카페이름': '', '카페주소': '', '카페 평점': '',
            '문화시설 이름': '', '문화시설 주소': '',
            '관광시설 이름': '', '관광시설 주소': '',
            '숙박업소 이름': '', '숙박업소 카테고리': '', '숙박업소 평점': '', '숙박업소 주소': ''
        }
        
        # 음식점 추천
        random_category = category_list[randint(0, len(category_list) - 1)]
        top_places = recommend_top_places(model, data, station_name, random_category)
        if not top_places.empty:
            place = top_places.sample(n=1).iloc[0]
            course_data.update({
                '음식점 이름': place['name'], '음식점 카테고리': random_category, '음식점 평점': place['predicted_rating'], '음식점 주소': place['address']
            })
        
        # 카페 추천
        top_cafes = recommend_cafes(model, data, station_name)
        if not top_cafes.empty:
            cafe = top_cafes.sample(n=1).iloc[0]
            course_data.update({
                '카페이름': cafe['name'], '카페주소': cafe['address'], '카페 평점': cafe['predicted_rating']
            })
        
        # 관광명소 추천
        top_tourist = recommend_tourist_and_cultural(data, station_name, '관광명소')
        if not top_tourist.empty:
            tourist = top_tourist.sample(n=1).iloc[0]
            course_data.update({
                '관광시설 이름': tourist['name'], '관광시설 주소': tourist['address']
            })
        
        # 문화시설 추천
        top_cultural = recommend_tourist_and_cultural(data, station_name, '문화시설')
        if not top_cultural.empty:
            cultural = top_cultural.sample(n=1).iloc[0]
            course_data.update({
                '문화시설 이름': cultural['name'], '문화시설 주소': cultural['address']
            })
        
        # 숙박시설 추천
        random_accommodation = accommodation_list[randint(0, len(accommodation_list) - 1)]
        top_accommodations = recommend_accommodations(model, data, station_name, random_accommodation)
        if not top_accommodations.empty:
            accommodation = top_accommodations.sample(n=1).iloc[0]
            course_data.update({
                '숙박업소 이름': accommodation['name'], '숙박업소 카테고리': random_accommodation, '숙박업소 평점': accommodation['predicted_rating'], '숙박업소 주소': accommodation['address']
            })
        
        # 데이터프레임으로 변환
        all_courses.append(course_data)
    
    # 전체 코스 데이터프레임으로 변환
    courses_df = pd.DataFrame(all_courses)
    
    # CSV 파일로 저장
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))
    courses_df.to_csv(csv_path, index=False)
    print(f"Course data saved to {csv_path}")

# 모든 역에 대해 100개의 랜덤 코스 생성 및 저장
for station_name in station_list:
    csv_path = f"data/dating_course_recommendation/course_{station_name}.csv"
    create_and_save_random_courses(model, data, station_name, 100, csv_path)
