import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout,QInputDialog, QSlider, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect
import cv2
import numpy as np
from UI import Ui_MainWindow  # 你的 UI 文件


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 当前图像和处理后的图像
        self.current_image = None
        self.original_image = None
        self.processed_image = None
        self.rotation_angle = 0  # 用于追踪当前的旋转角度

        # 绑定图片选择按钮
        self.pushButton_9.clicked.connect(self.select_image)
        self.pushButton_10.clicked.connect(self.save_image)

        # 创建 10 个滑动条，并绑定到功能
        self.sliders = []
        self.create_sliders()

        # 绑定按钮功能
        self.pushButton_11.clicked.connect(self.histogram_equalization)
        self.pushButton_4.clicked.connect(self.crop_image)
        self.pushButton_6.clicked.connect(self.rotate_image)
        self.pushButton_18.clicked.connect(self.apply_beautification)
        self.pushButton_17.clicked.connect(self.dehaze_image)
        self.pushButton_19.clicked.connect(self.add_watermark)
        self.pushButton_15.clicked.connect(self.apply_hsl)

        # 绑定文字按钮
        self.pushButton_8.clicked.connect(self.add_text)
        # 绑定暂存按钮
        self.store_button.clicked.connect(self.store_image)
        # 绑定重置按钮
        self.reset_button.clicked.connect(self.reset_image)

    def create_sliders(self):
        """创建 10 个 QSlider 控件并绑定功能"""
        slider_functions = [
            self.adjust_brightness,    # 亮度
            self.adjust_contrast,      # 对比度
            self.adjust_exposure,      # 曝光
            self.adjust_saturation,    # 饱和度
            self.apply_smoothing,      # 平滑
            self.apply_sharpening,     # 锐化
            self.adjust_hue,           # 色调
            self.adjust_temperature,   # 色温
            self.apply_curve,          # 曲线调整
            self.adjust_vividness      # 光感
        ]
    #设置滑动调在UI上的显示位置
        for i, func in enumerate(slider_functions):
            slider = QSlider(Qt.Horizontal, self)
            slider.setRange(-50, 50)  # 设置滑动条范围
            slider.setValue(0)          # 初始值为 0
            slider.valueChanged.connect(lambda value, f=func: f(value))
            slider.setGeometry(950, 105 + i * 66, 200, 24)  # 动态设置滑动条位置
            slider.show()
            self.sliders.append(slider)

    #选择图片
    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.original_image = self.current_image.copy()  # 保存原始图像的副本
            self.processed_image = self.current_image.copy()
            self.display_image(self.current_image)

    #保存图片
    def save_image(self):

        if self.processed_image is None:
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if file_name:
            cv2.imwrite(file_name, self.processed_image)

    #在界面上展示图片
    def display_image(self, image):
        if image is None:
            return
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # 设置适应 QLabel 大小显示图像
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(False)  # 图像正常显示，而不是最大化

    #重置图像
    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()  # 恢复到原始图像
            self.display_image(self.original_image)
            self.reset_sliders()
    #重置滑动条
    def reset_sliders(self):
        for slider in self.sliders:
            slider.setValue(0)  # 重置滑动条值为 0

    # 旋转功能（每次点击旋转 90°）
    def rotate_image(self):
        if self.current_image is None:
            return

        # 更新旋转角度，每次点击旋转 90°
        self.rotation_angle += 90
        if self.rotation_angle == 360:
            self.rotation_angle = 0  # 重置为 0，完成一圈旋转

        # 计算旋转矩阵
        height, width = self.current_image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), self.rotation_angle, 1)

        # 旋转图像
        self.processed_image = cv2.warpAffine(self.current_image, rotation_matrix, (width, height))
        self.display_image(self.processed_image)

    # 保存修改
    def store_image(self):
        if self.processed_image is not None:
            self.current_image = self.processed_image
            self.display_image(self.current_image)

    # 亮度调整
    def adjust_brightness(self, value):
        if self.current_image is None :
            return
        self.processed_image = cv2.convertScaleAbs(self.current_image, alpha=1, beta=value)
        self.display_image(self.processed_image)

    # 对比度调整
    def adjust_contrast(self, value):
        if self.current_image is None:
            return
        alpha = 1 + value / 100.0
        self.processed_image = cv2.convertScaleAbs(self.current_image, alpha=alpha, beta=0)
        self.display_image(self.processed_image)

    # 曝光调整
    def adjust_exposure(self, value):
        if self.current_image is None:
            return
        self.processed_image = cv2.convertScaleAbs(self.current_image, alpha=1 + value / 100.0, beta=value)
        self.display_image(self.processed_image)

    # 饱和度调整
    def adjust_saturation(self, value):
        if self.current_image is None:
            return
        hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_image[..., 1] *= (1 + value / 100.0)
        hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)
        self.processed_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
        self.display_image(self.processed_image)

    # 平滑处理
    def apply_smoothing(self, value):
        if self.current_image is None:
            return
        kernel_size = max(1, 5 + value // 30 * 2)  # 动态调整卷积核大小
        self.processed_image = cv2.GaussianBlur(self.current_image, (kernel_size, kernel_size), 0)
        self.display_image(self.processed_image)

    # 锐化处理：使用拉普拉斯滤波器
    def apply_sharpening(self, value):
        if self.current_image is None:
            return

        # 拉普拉斯滤波器
        laplacian = cv2.Laplacian(self.current_image, cv2.CV_64F)

        # 转换为绝对值，避免负数，并将其转换回 uint8
        laplacian = cv2.convertScaleAbs(laplacian)

        # 将拉普拉斯结果的绝对值加回原图，形成锐化效果
        sharpened = cv2.addWeighted(self.current_image, 1.0, laplacian, value / 80.0, 0)

        # 更新处理后的图像
        self.processed_image = sharpened
        self.display_image(self.processed_image)

    # 色调调整
    def adjust_hue(self, value):
        if self.current_image is None:
            return
        hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_image[..., 0] += value
        hsv_image[..., 0] = np.mod(hsv_image[..., 0], 180)
        self.processed_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
        self.display_image(self.processed_image)

    # 色温调整
    def adjust_temperature(self, value):
        if self.current_image is None:
            return
        b, g, r = cv2.split(self.current_image)
        r = cv2.add(r, value)
        b = cv2.add(b, -value)
        self.processed_image = cv2.merge((b, g, r))
        self.display_image(self.processed_image)

    # 曲线调整
    def apply_curve(self, value):
        if self.current_image is None:
            return
        look_up_table = np.zeros((1, 256), dtype=np.uint8)
        for i in range(256):
            look_up_table[0, i] = np.clip(i + value, 0, 255)
        self.processed_image = cv2.LUT(self.current_image, look_up_table)
        self.display_image(self.processed_image)

    # 光感调整
    def adjust_vividness(self, value):
        if self.current_image is None:
            return
        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1 + value / 100.0), 0, 255)
        self.processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.display_image(self.processed_image)

    # 直方图均衡化
    def histogram_equalization(self):
        if self.current_image is None:
            return
        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)
        self.processed_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image)

    #自定义裁剪图片
    def crop_image(self):
        if self.current_image is None:
            return

        roi = cv2.selectROI(windowName="roi", img=self.current_image, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi
        if roi != (0, 0, 0, 0):
            cut = self.current_image[y:y + h, x:x + w]
        cv2.imshow("roi", self.current_image)
        cv2.waitKey(20)
        cv2.destroyAllWindows()
        self.processed_image = cut
        self.display_image(self.processed_image)

    #添加文字

    def add_text(self):
        """添加文字到选定区域"""
        if self.current_image is None:
            return

        # 用户框选区域
        roi = cv2.selectROI(windowName="Select Text Area", img=self.current_image, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi

        # 检查是否选择了有效区域
        if roi == (0, 0, 0, 0):
            cv2.destroyAllWindows()
            return

        # 弹出文本输入框
        text, ok = QInputDialog.getText(self, "输入文字", "请输入要添加的文字：")
        if not ok or not text:
            cv2.waitKey(10)
            cv2.destroyAllWindows()
            return

        # 在图像上绘制文字
        self.processed_image = self.current_image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w / 200, h / 50)  # 动态调整字体大小
        font_thickness = max(1, int(font_scale * 2))  # 动态调整字体粗细
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # 计算文字的左上角坐标，确保文字居中显示在选区
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2

        # 在选定区域绘制文字
        cv2.putText(self.processed_image, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness,
                    cv2.LINE_AA)

        # 显示结果图像
        cv2.imshow("Text Area", self.processed_image)
        cv2.waitKey(10)
        cv2.destroyAllWindows()

        # 更新显示
        self.display_image(self.processed_image)

    # 美颜滤镜：为人脸图片添加美颜功能
    def apply_beautification(self):

        if self.current_image is None:
            return

        # 检查是否已经应用过美颜处理，防止重复处理
        if hasattr(self, 'beautified') and self.beautified:
            return  # 如果已经处理过美颜，则不再处理

        # 使用灰度图进行人脸检测
        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 对每一张检测到的人脸进行美颜处理
        for (x, y, w, h) in faces:
            face = self.current_image[y:y + h, x:x + w]

            # 1. 磨皮：对皮肤区域使用双边滤波
            smoothed_face = cv2.bilateralFilter(face, d=9, sigmaColor=75, sigmaSpace=75)

            # 将处理过的人脸区域更新到原图
            self.current_image[y:y + h, x:x + w] = smoothed_face

        # 标记美颜已经应用
        self.processed_image = self.current_image
        self.display_image(self.processed_image)

    # 去雾功能
    def dehaze_image(self):
        if self.current_image is None:
            return
        self.processed_image = cv2.detailEnhance(self.current_image, sigma_s=10, sigma_r=0.15)
        self.display_image(self.processed_image)

    # 水印功能
    def add_watermark(self):
        if self.current_image is None:
            return
        watermark = "JMU"
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (440, 470)
        color = (0, 0, 255)
        self.processed_image = self.current_image.copy()
        cv2.putText(self.processed_image, watermark, position, font, 1, color, 2, cv2.LINE_AA)
        self.display_image(self.processed_image)

    # HSL功能
    def apply_hsl(self):
        if self.current_image is None:
            return
        hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        h, s, l = cv2.split(hsv_image)
        h[:] = np.clip(h + 10, 0, 179)
        hsv_image = cv2.merge([h, s, l])
        self.processed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        self.display_image(self.processed_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
