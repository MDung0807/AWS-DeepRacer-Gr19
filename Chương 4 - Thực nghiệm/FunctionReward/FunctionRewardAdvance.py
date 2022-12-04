def reward_function(params):
    #Hàm dùng để khuyến khích model chạy thẳng và nhanh 
    # Đọc các tham số đầu vào
    distance_from_center = params['distance_from_center'] # khoảng cách đến đường trung tâm
    track_width = params['track_width'] # độ rộng của đường đua
    abs_steering = abs(params['steering_angle'])  # góc lái của model
    speed = params['speed'] # tốc độ của model
    
    # Đặt 3 điểm để so sánh với khoảng cách đến đường trung tâm
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track

   # Nếu thay đổi góc lái nhỏ và tốc độ cao sẽ nhận thêm thưởng.
    if abs(abs_steering) < 0.1 and speed > 3.5:
        reward *= 1.8
    elif abs(abs_steering) < 0.2 and speed > 2:
        reward *= 1.1
    return float(reward)