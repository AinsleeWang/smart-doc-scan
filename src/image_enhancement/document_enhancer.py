import cv2
import numpy as np
from scipy.spatial import distance as dist

class DocumentEnhancer:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

    def order_points(self, pts):
        """
        將四個點按照左上、右上、右下、左下的順序排列
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # 計算左上和右下
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # 計算右上和左下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    def four_point_transform(self, image, pts):
        """
        四點透視變換
        """
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # 計算新影像的寬度
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # 計算新影像的高度
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # 建立目標點
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # 計算透視變換矩陣並應用
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def enhance_image_quality(self, image):
        # 轉換到灰度圖進行初步處理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用較大的核心進行自適應直方圖均衡化
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16,16))
        enhanced_gray = clahe.apply(gray)
        
        # 使用高斯模糊減少噪點
        blurred = cv2.GaussianBlur(enhanced_gray, (3,3), 0)
        
        # 使用自適應二值化處理陰影
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,  # 塊大小
            10   # 常數
        )
        
        # 轉回BGR以便後續處理
        enhanced_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # 輕微銳化以提高文字清晰度
        kernel = np.array([[-0.5,-0.5,-0.5],
                        [-0.5, 5,-0.5],
                        [-0.5,-0.5,-0.5]])
        sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)
        
        # 使用較輕微的降噪參數
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, 
                                                None, 
                                                10,  # 降低濾波強度
                                                10,
                                                7,
                                                21)

        # 移除中間步驟的顯示，直接返回最終結果

        return denoised

    def process_document(self, image, corners):
        """
        執行透視變換和影像增強
        """
        # 透視變換
        warped = self.four_point_transform(image, corners.reshape(4, 2))
        
        if self.debug_mode:
            cv2.imshow('Warped', warped)
            cv2.waitKey(0)

        # 影像品質優化
        enhanced = self.enhance_image_quality(warped)
        
        return enhanced

def main(image_path, corners):
    """
    主函數：處理文件影像
    """
    # 讀取影像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("無法讀取影像")

    # 建立增強器實例
    enhancer = DocumentEnhancer(debug_mode=True)
    
    # 處理文件
    result = enhancer.process_document(image, corners)
    
    # 儲存結果
    output_path = 'enhanced_document.jpg'
    cv2.imwrite(output_path, result)
    print(f"處理後的影像已儲存至: {output_path}")

    return result