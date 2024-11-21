import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# 데이터 로드 및 전처리
def load_and_preprocess_data(folder_path):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    all_data = []

    for file in csv_files:
        df = pd.read_csv(file)
        station_name = os.path.basename(file).split('_')[1].replace('.csv', '')
        df['역이름'] = station_name
        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True)

    # 결측값 처리
    data = data.fillna('')  # 결측값을 빈 문자열로 대체

    # 모든 데이터를 문자열로 변환
    data = data.astype(str)

    return data

data_folder_path = 'data/dating_course_recommendation'
data = load_and_preprocess_data(data_folder_path)

# 레이블 인코딩 (모든 문자열 컬럼 인코딩)
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 특성 및 레이블 설정
features = data[['역이름', '음식점 카테고리', '숙박업소 카테고리']].astype(float)
labels = data[['음식점 이름', '카페이름', '문화시설 이름', '관광시설 이름', '숙박업소 이름']].astype(float)

# 데이터셋 및 데이터로더 정의
class CourseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.values
        self.labels = labels.values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

dataset = CourseDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# RNN 모델 정의
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x.unsqueeze(1), h0)
        out = self.fc(out[:, -1, :])
        return out

input_size = 3  # 역이름, 음식점 카테고리, 숙박업소 카테고리
hidden_size = 256  # 은닉층의 노드 수
output_size = 5  # 음식점 이름, 카페이름, 문화시설 이름, 관광시설 이름, 숙박업소 이름
model = RNNModel(input_size, hidden_size, output_size)

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        features = batch['features']
        labels = batch['labels']

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

# 모델을 사용한 데이트 코스 추천 함수
def recommend_course(model, station_name, food_category, accommodation_category):
    model.eval()
    with torch.no_grad():
        # 입력값을 인코딩
        station_id = label_encoders['역이름'].transform([station_name])[0]
        food_category_id = label_encoders['음식점 카테고리'].transform([food_category])[0]
        accommodation_category_id = label_encoders['숙박업소 카테고리'].transform([accommodation_category])[0]

        # 입력값을 텐서로 변환
        input_features = torch.tensor([station_id, food_category_id, accommodation_category_id], dtype=torch.float32)
        input_features = input_features.unsqueeze(0)  # 배치 차원 추가
        predicted_labels = model(input_features).squeeze().numpy()

        # 예측된 결과를 디코딩
        predicted_course = {
            '음식점 이름': label_encoders['음식점 이름'].inverse_transform([int(predicted_labels[0])])[0],
            '카페이름': label_encoders['카페이름'].inverse_transform([int(predicted_labels[1])])[0],
            '문화시설 이름': label_encoders['문화시설 이름'].inverse_transform([int(predicted_labels[2])])[0],
            '관광시설 이름': label_encoders['관광시설 이름'].inverse_transform([int(predicted_labels[3])])[0],
            '숙박업소 이름': label_encoders['숙박업소 이름'].inverse_transform([int(predicted_labels[4])])[0]
        }

        return predicted_course

# 예시 사용
station_name = '공덕역'
food_category = '한식'
accommodation_category = '호텔'

recommended_course = recommend_course(model, station_name, food_category, accommodation_category)
print(recommended_course)
