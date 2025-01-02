import cv2
import numpy as np

def process_image(image):
    """改進的文件邊緣檢測流程"""
    # 獲取圖像尺寸
    height, width = image.shape[:2]
    print(f"圖像尺寸: {width}x{height}")
    
    # 1. 改進的預處理
    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用自適應直方圖均衡化增強對比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 使用適當大小的高斯模糊
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 2. 改進的邊緣檢測
    # 使用Canny邊緣檢測，自適應閾值
    median = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))
    edges = cv2.Canny(blurred, lower, upper)
    
    # 3. 改進的形態學處理
    # 使用較小的kernel進行膨脹操作
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # 4. 改進的輪廓檢測
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 繪製所有輪廓用於調試
    debug_image = image.copy()
    cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
    
    print(f"找到 {len(contours)} 個初始輪廓")
    
    if not contours:
        return False, None
    
    # 5. 改進的輪廓篩選
    # 按面積排序輪廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    max_contour = contours[0]
    max_area = cv2.contourArea(max_contour)
    print(f"最大輪廓面積: {max_area}")
    
    # 面積閾值設為圖像面積的1%（降低閾值使檢測更寬鬆）
    min_area = (width * height) * 0.01
    print(f"最小輪廓面積閾值: {min_area}")
    
    if max_area < min_area:
        print("最大輪廓面積太小")
        return False, None
    
    # 6. 改進的輪廓近似
    # 使用多邊形近似來平滑輪廓
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    # 確保找到四個角點
    if len(approx) == 4:
        # 重要：將輪廓轉換為正確的格式
        approx = approx.reshape(4, 1, 2)
        return True, approx
    else:
        # 如果不是四邊形，使用最小面積矩形
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 重要：將box轉換為正確的格式
        box = box.reshape(4, 1, 2)
        return True, box

def draw_contours(image, contour):
    """改進的輪廓繪製函數"""
    result = image.copy()
    
    # 確保輪廓格式正確
    if contour.shape[1:] != (1, 2):
        contour = contour.reshape(-1, 1, 2)
    
    # 繪製輪廓
    cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)
    
    # 標示角點
    for i in range(len(contour)):
        point = tuple(contour[i][0])
        cv2.circle(result, point, 10, (0, 0, 255), -1)
        cv2.putText(result, str(i), 
                   (point[0]-20, point[1]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return result