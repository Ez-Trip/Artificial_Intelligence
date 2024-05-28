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

# 데이터 전처리 함수들
def read_and_preprocess_csv_files(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    all_data = []

    for file_path in csv_files:
        # 파일명에서 역 이름 추출
        station_name = os.path.basename(file_path).split('_')[-1].replace('.csv', '')
        
        df = pd.read_csv(file_path)
        df['카테고리'] = df['카테고리'].apply(preprocess_category)
        df['평점'] = df['평점'].apply(preprocess_rating)
        
        # 역 이름을 데이터프레임에 추가
        df['역이름'] = station_name
        
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

def preprocess_category(category):
    if pd.isna(category):
        return '기타'
    category = category.replace('"', '')
    return category.split(' > ')[0]

def preprocess_rating(rating):
    if pd.isna(rating):
        return 0.0
    rating = re.findall(r'\d+\.\d+', str(rating))
    return float(rating[0]) if rating else 0.0

# 데이터 로드 및 전처리
csv_folder_path = 'data/place'
data = read_and_preprocess_csv_files(csv_folder_path)

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

# 모델 저장 경로
model_dir = 'model'
model_path = os.path.join(model_dir, 'test_place_learning.pth')

# 모델 학습 및 저장
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if os.path.exists(model_path):
    # 기존 모델 불러오기
    model.load_state_dict(torch.load(model_path))
    print("Model loaded from", model_path)
else:
    print("Training new model")

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

    # 매 epoch마다 모델 저장
    torch.save(model.state_dict(), model_path)

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

# 특정 역의 음식점 추천 함수
def recommend_top_places_near_station(model, data, station_name, top_n=3):
    station_id = label_encoders['역이름'].transform([station_name])[0]
    filtered_data = data[data['역이름'] == station_id]
    
    features = torch.tensor(filtered_data[['카테고리', '역이름']].values, dtype=torch.float32)
    features = features.unsqueeze(1)
    
    with torch.no_grad():
        predictions = model(features).squeeze()
    
    filtered_data['predicted_rating'] = predictions
    top_places = filtered_data.sort_values(by='predicted_rating', ascending=False).head(top_n)
    
    return top_places

# 사용자로부터 역 이름 입력 받기
station_list = [
    "공덕역", "광흥창역", "대흥역", "디지털미디어시티역", "마포구청역", 
    "마포역", "망원역", "상수역", "서강대역", "신촌역", "아현역", 
    "애오개역", "월드컵경기장역", "이대역", "합정역", "홍대입구역"
]

print("Available stations: ", ", ".join(station_list))
station_name = input("Enter a station name: ")

# 해당 역 주변의 평점이 좋은 음식점 추천
top_places = recommend_top_places_near_station(model, data, station_name)

# 추천 결과 출력
print(f"\nTop {len(top_places)} places near {station_name}:")
for idx, row in top_places.iterrows():
    print(f"{idx+1}. Name: {row['name']}, Rating: {row['predicted_rating']}")
