import cv2
import os
import argparse
from datetime import datetime
from src.edge_detection.detector import process_image
from src.image_enhancement.document_enhancer import DocumentEnhancer
from src.utils.config_handler import Conf

def test_document_scanner(conf_path, image_path):
    """
    改進的文件掃描測試功能
    
    Args:
        conf_path: 配置文件路徑
        image_path: 圖片文件路徑
    """
    # 檢查文件路徑
    if not os.path.exists(image_path):
        print(f"錯誤：找不到圖片文件 {image_path}")
        return
    
    if not os.path.exists(conf_path):
        print(f"錯誤：找不到配置文件 {conf_path}")
        return

    try:
        # 加載配置文件
        conf = Conf(conf_path)
        
        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print("錯誤：無法讀取圖片")
            return
            
        # 步驟1：顯示原圖
        cv2.imshow('1. Original Image', image)
        
        # 步驟2：進行邊緣檢測
        success, corners = process_image(image)
        
        if success:
            # 顯示邊緣檢測結果
            scan_result = image.copy()
            cv2.drawContours(scan_result, [corners], -1, (0, 255, 0), 2)
            cv2.imshow('2. Document Detection', scan_result)
            
            # 步驟3：進行透視變換和影像增強
            enhancer = DocumentEnhancer(debug_mode=False)  # 關閉debug模式
            enhanced_result = enhancer.process_document(image, corners)
            cv2.imshow('3. Final Result', enhanced_result)
            
            # 創建輸出目錄
            output_dir = os.path.join('data', 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成輸出檔名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"enhanced_{base_name}_{timestamp}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存處理後的成品
            cv2.imwrite(output_path, enhanced_result)
            print(f"處理後的成品已保存至: {output_path}")
            
            # 等待按鍵並關閉所有視窗
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("未能檢測到文件邊緣")
            
    except Exception as e:
        import traceback
        print(f"處理過程中發生錯誤：{str(e)}")
        print(traceback.format_exc())

def main():
    """主程式"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True,
                    help="配置文件的路徑")
    ap.add_argument("-i", "--image", required=True,
                    help="要處理的圖片路徑")
    args = vars(ap.parse_args())

    test_document_scanner(args["conf"], args["image"])

if __name__ == "__main__":
    main()