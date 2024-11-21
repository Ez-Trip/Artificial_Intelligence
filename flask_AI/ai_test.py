from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from datetime import datetime

app = Flask(__name__)

# server 용
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://app_user:q1w2e3r4@db:3306/ez_trip'

# local 용
#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:q1w2e3r4@localhost:3306/Ez-Trip'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Schedule(db.Model):
    __tablename__ = 'Schedule'
    id = db.Column(db.BigInteger, primary_key=True)  # BIGINT로 설정
    title = db.Column(db.String(50))
    path = db.Column(db.String(100))
    date = db.Column(db.Date, default=datetime.utcnow)
    image = db.Column(db.String(255))
    price = db.Column(db.Integer)
    member_id = db.Column(db.Integer)
    path_details = db.relationship('PathDetail', back_populates='schedule', lazy=True, cascade="all, delete-orphan")

class PathDetail(db.Model):
    __tablename__ = 'PathDetail'
    id = db.Column(db.BigInteger, primary_key=True)  # BIGINT로 설정
    segment_code = db.Column(db.String(10))
    place_name = db.Column(db.String(100))
    address = db.Column(db.String(255))
    price = db.Column(db.Integer)
    segment_type = db.Column(db.String(50))
    schedule_id = db.Column(db.BigInteger, db.ForeignKey('Schedule.id'))  # BIGINT로 설정

    schedule = db.relationship("Schedule", back_populates='path_details')
    
    def __init__(self, segment_code, place_name, address, price, segment_type, schedule):
        self.segment_code = segment_code
        self.place_name = place_name
        self.address = address
        self.price = price
        self.segment_type = segment_type
        self.schedule = schedule  # schedule 연결

    # 필요한 setter 메소드들
    def set_schedule(self, schedule):
        self.schedule = schedule

# SnsPost 모델 (예제)
class SnsPost(db.Model):
    __tablename__ = 'SnsPost'
    id = db.Column(db.BigInteger, primary_key=True)
    schedule_id = db.Column(db.BigInteger, db.ForeignKey('Schedule.id'))

def reset_database():
    with app.app_context():
        # 외래 키 제약 조건 무시
        db.session.execute(text('SET FOREIGN_KEY_CHECKS = 0;'))

        # 테이블 데이터 삭제
        db.session.execute(text('DELETE FROM SnsPost'))
        db.session.execute(text('DELETE FROM PathDetail'))
        db.session.execute(text('DELETE FROM Schedule'))

        # 외래 키 제약 조건 다시 활성화
        db.session.execute(text('SET FOREIGN_KEY_CHECKS = 1;'))
        db.session.commit()

# # 앱 시작 시 테이블 생성 및 데이터베이스 초기화
# with app.app_context():
#     db.create_all()  # 테이블이 없는 경우 생성
#     reset_database()  # 데이터 초기화

# 추천 코스 생성 엔드포인트
@app.route('/api/schedules/recommend', methods=['POST'])
def recommend_course():
    try:
        data = request.get_json()
        station_name = data.get('stationName', '마포역')
        total_budget = data.get('totalBudget', 300000)
        preference = data.get('preference', 'A3B1C1D1')
        member_id = data.get('memberId')

        # 더미 AI 추천 결과
        course_detail = [
            {"segment_code": "A3", "place_name": "소문난쭈꾸미 마포본점", "address": "서울 마포구 도화동 179-15", "price": 78000, "segment_type": "한식"},
            {"segment_code": "B1", "place_name": "에버8", "address": "서울 서대문구 대현동 104-5", "price": 150000, "segment_type": "호텔"},
            {"segment_code": "C1", "place_name": "뜨랑블랑", "address": "서울 마포구 마포동 236-5", "price": 10000, "segment_type": "카페"},
            {"segment_code": "D1", "place_name": "갤러리일상", "address": "서울 마포구 도화동 251-6", "price": 3000, "segment_type": "문화시설"}
            #{"segment_code": "E1", "place_name": "소금길", "address": "서울 마포구 염리동 9-245", "price": 5000, "segment_type": "관광시설"}
        ]
        total_price = sum(item["price"] for item in course_detail)

        # Schedule 생성
        new_schedule = Schedule(
            title="추천 코스",
            path=station_name,
            price=total_price,
            member_id=member_id
        )
        db.session.add(new_schedule)
        db.session.flush()  # Schedule ID가 PathDetail에 할당되도록 플러시

        # PathDetail 추가
        for detail in course_detail:
            pathdetail = PathDetail(
                segment_code=detail["segment_code"],
                place_name=detail["place_name"],
                address=detail["address"],
                price=detail["price"],
                segment_type=detail["segment_type"],
                schedule=new_schedule  # schedule을 여기서 제대로 전달
            )
            db.session.add(pathdetail)

        db.session.commit()

        # 클라이언트에 반환할 데이터 구조
        response = {
            'title': new_schedule.title,
            'path': new_schedule.path,
            'totalPrice': total_price,
            'pathDetails': [
                {
                    'segmentCode': detail.segment_code,
                    'placeName': detail.place_name,
                    'address': detail.address,
                    'price': detail.price,
                    'segmentType': detail.segment_type
                }
                for detail in new_schedule.path_details
            ]
        }
        return jsonify(response)

    except Exception as e:
        db.session.rollback()  # 롤백하여 데이터베이스 상태를 유지
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3309)