# File test.py

# Import Flask app và các model từ file flask_rec_app.py
from face_rec_flask_app import app, db, User

def print_all_users():
    with app.app_context():
        # Thực hiện truy vấn lấy tất cả người dùng
        users = User.query.all()
        for user in users:
            print(f"Họ tên: {user.name}, Chức vụ: {user.position}, Số điện thoại: {user.phone}, Email: {user.email}, Ngày sinh: {user.birthdate}, Địa chỉ: {user.address}")

if __name__ == '__main__':
    print_all_users()
