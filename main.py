import sys
import platform
import numpy as np
import mvsdk

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QRadioButton, QCheckBox, QSlider,
    QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox,
    QGridLayout, QButtonGroup
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

#####################################
# Алгоритмы из предыдущих наработок
#####################################

def process_mode_30(pair, k):
    """
    Режим 30 Гц: берём два кадра A, B и считаем B - k*A
    """
    if len(pair) < 2:
        return None
    A = pair[-2].astype(np.float32)
    B = pair[-1].astype(np.float32)
    diff = B - k * A
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return diff

def process_mode_15(group, k):
    """
    Режим 15 Гц: 4 кадра (A, A, B, B).
    Усредняем первые два (A_avg) и последние два (B_avg), вычитаем B_avg - k*A_avg.
    """
    if len(group) < 4:
        return None
    A_avg = (group[-4].astype(np.float32) + group[-3].astype(np.float32)) / 2
    B_avg = (group[-2].astype(np.float32) + group[-1].astype(np.float32)) / 2
    diff = B_avg - k * A_avg
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return diff

def process_mode_20(group, k):
    """
    Режим 20 Гц: 3 кадра (A, B, C).
    Усредняем A и C -> A_prime, вычитаем B - k*A_prime.
    """
    if len(group) < 3:
        return None
    A_prime = (group[-3].astype(np.float32) + group[-1].astype(np.float32)) / 2
    B = group[-2].astype(np.float32)
    diff = B - k * A_prime
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return diff

def temporal_median_filter(diff_buffer):
    """
    Временная медианная фильтрация по N последним кадрам разницы.
    """
    stacked = np.stack(diff_buffer, axis=0)  # shape (N, H, W) или (N, H, W, C)
    median_filtered = np.median(stacked, axis=0).astype(np.uint8)
    return median_filtered

def apply_threshold(image, threshold_value):
    """
    Простой порог: все пиксели < threshold_value => 0
    """
    _, thresh_img = \
        mvsdk.Camera.cv2.threshold(image, threshold_value, 255, mvsdk.Camera.cv2.THRESH_TOZERO)
    # Если не хотим зависеть от cv2, можно сделать NumPy-вариант:
    # thresh_img = image.copy()
    # thresh_img[thresh_img < threshold_value] = 0
    return thresh_img

class AdaptiveK:
    """
    Адаптивный коэффициент k, подстраивает k, чтобы «тёмные» кадры были почти чёрными.
    """
    def __init__(self, initial_k=0.95, learning_rate=0.01, target_dark_level=5.0):
        self.k = initial_k
        self.learning_rate = learning_rate
        self.target_dark_level = target_dark_level

    def update(self, dark_frame):
        """
        dark_frame – кадр без подсветки (или усреднённый "тёмный"),
        чем он ярче, тем сильнее нужно уменьшать k.
        """
        mean_dark = np.mean(dark_frame)
        error = mean_dark - self.target_dark_level
        self.k -= self.learning_rate * error
        self.k = max(0.5, min(1.5, self.k))
        return self.k


############################################
# Функция-помощник: numpy → QPixmap
############################################
def convert_frame_to_qpixmap(frame):
    """
    Если frame - GRAY (H, W) или (H, W, 1) → QImage.Format_Grayscale8
    Если frame - BGR (H, W, 3) → конвертируем BGR->RGB → QImage.Format_RGB888
    """
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        # GRAY
        h, w = frame.shape[:2]
        bytes_per_line = w
        qimg = QImage(
            frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8
        )
    else:
        # BGR -> RGB
        h, w, ch = frame.shape
        rgb = frame[..., ::-1]  # BGR->RGB
        bytes_per_line = ch * w
        qimg = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
    return QPixmap.fromImage(qimg)


############################################
# Основной класс PyQt5
############################################
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demo IR Camera Processing (PyQt + MvSDK + Все режимы)")

        # Параметры и буферы
        self.hCamera = None
        self.monoCamera = False
        self.pFrameBuffer = None
        self.FrameBufferSize = 0

        self.frame_buffer = []  # Для хранения исходных кадров
        self.diff_buffer = []   # Для временной медианной фильтрации

        # Текущий режим (30 Гц, 15 Гц, 20 Гц)
        self.mode = 1  # 1=30, 2=15, 3=20

        # Настройки фильтров
        self.use_temporal_filter = False
        self.temporal_window = 3

        self.use_threshold = False
        self.threshold_value = 20

        self.use_adaptive_k = False
        self.base_k = 0.95
        self.adaptive_k = AdaptiveK(initial_k=self.base_k, learning_rate=0.01)

        self.init_ui()

        # Инициируем камеру
        try:
            self.init_camera()
        except mvsdk.CameraException as e:
            print(f"Camera Init error: {e}")
            sys.exit(1)

        # Таймер для покадрового считывания
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        """
        Создание элементов GUI: радиокнопки, чекбоксы, спинбоксы, лейблы под картинку и т.д.
        """
        # --- Режимы (30 Гц, 15 Гц, 20 Гц) ---
        self.radio_30 = QRadioButton("30 Гц (2 кадра)")
        self.radio_15 = QRadioButton("15 Гц (4 кадра)")
        self.radio_20 = QRadioButton("20 Гц (3 кадра)")

        self.radio_30.setChecked(True)  # по умолчанию
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_30, 1)
        self.mode_group.addButton(self.radio_15, 2)
        self.mode_group.addButton(self.radio_20, 3)
        self.mode_group.buttonClicked[int].connect(self.on_mode_changed)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.radio_30)
        mode_layout.addWidget(self.radio_15)
        mode_layout.addWidget(self.radio_20)

        mode_group_box = QGroupBox("Режим (частота фонарика)")
        mode_group_box.setLayout(mode_layout)

        # --- Дополнительные опции ---
        self.check_temp_filter = QCheckBox("Временная медианная фильтрация")
        self.check_temp_filter.stateChanged.connect(self.on_temp_filter_changed)

        self.spin_temp_window = QSpinBox()
        self.spin_temp_window.setRange(2, 10)
        self.spin_temp_window.setValue(3)
        self.spin_temp_window.valueChanged.connect(self.on_temp_window_changed)

        self.check_threshold = QCheckBox("Адаптивный порог")
        self.check_threshold.stateChanged.connect(self.on_threshold_changed)

        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(1, 255)
        self.spin_threshold.setValue(20)
        self.spin_threshold.valueChanged.connect(self.on_threshold_value_changed)

        self.check_adaptive_k = QCheckBox("Адаптивное k")
        self.check_adaptive_k.stateChanged.connect(self.on_adaptive_k_changed)

        self.spin_init_k = QDoubleSpinBox()
        self.spin_init_k.setRange(0.0, 2.0)
        self.spin_init_k.setSingleStep(0.01)
        self.spin_init_k.setValue(0.95)
        self.spin_init_k.valueChanged.connect(self.on_init_k_changed)

        self.spin_learn_rate = QDoubleSpinBox()
        self.spin_learn_rate.setRange(0.001, 0.1)
        self.spin_learn_rate.setSingleStep(0.001)
        self.spin_learn_rate.setValue(0.01)
        self.spin_learn_rate.valueChanged.connect(self.on_learn_rate_changed)

        opts_layout = QGridLayout()
        # 1-я строка: медианная фильтрация
        opts_layout.addWidget(self.check_temp_filter, 0, 0)
        opts_layout.addWidget(self.spin_temp_window, 0, 1)
        # 2-я строка: порог
        opts_layout.addWidget(self.check_threshold, 1, 0)
        opts_layout.addWidget(self.spin_threshold, 1, 1)
        # 3-я строка: адаптивное k
        opts_layout.addWidget(self.check_adaptive_k, 2, 0)
        opts_layout.addWidget(self.spin_init_k, 2, 1)
        opts_layout.addWidget(self.spin_learn_rate, 2, 2)

        opts_group_box = QGroupBox("Доп. опции")
        opts_group_box.setLayout(opts_layout)

        # --- Кнопка выхода ---
        self.btn_exit = QPushButton("Выход")
        self.btn_exit.clicked.connect(self.close)

        # --- Метки для отображения изображений ---
        self.label_orig = QLabel("Original")
        self.label_proc = QLabel("Processed")
        self.label_orig.setScaledContents(True)
        self.label_proc.setScaledContents(True)

        images_layout = QHBoxLayout()
        images_layout.addWidget(self.label_orig)
        images_layout.addWidget(self.label_proc)

        # --- Общая компоновка ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(mode_group_box)
        main_layout.addWidget(opts_group_box)
        main_layout.addLayout(images_layout)
        main_layout.addWidget(self.btn_exit)

        self.setLayout(main_layout)
        self.setMinimumSize(900, 500)

    def init_camera(self):
        """
        Логика инициализации камеры, как в cv_grab.py (но без cv2).
        """
        # Поиск устройств
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            raise mvsdk.CameraException(mvsdk.CAMERA_STATUS_NO_DEVICE_FOUND)

        i = 0  # берем первую камеру, при необходимости спросить пользователя
        DevInfo = DevList[i]
        print("Use camera:", DevInfo.GetFriendlyName())

        # Открытие
        self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        cap = mvsdk.CameraGetCapability(self.hCamera)

        # Цвет/моно
        self.monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
        if self.monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # Непрерывный захват
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        # Пример: выключим автоэкспозицию, поставим 30ms
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, 30_000)

        # Запуск
        mvsdk.CameraPlay(self.hCamera)

        # Подготовка буфера
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax
        if not self.monoCamera:
            FrameBufferSize *= 3
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        self.FrameBufferSize = FrameBufferSize

        print("Camera init complete.")

    #############################################
    # Обработчики сигналов UI
    #############################################
    def on_mode_changed(self, mode_id):
        self.mode = mode_id
        # Сброс буферов
        self.frame_buffer.clear()
        self.diff_buffer.clear()

    def on_temp_filter_changed(self, state):
        self.use_temporal_filter = (state == Qt.Checked)
        if not self.use_temporal_filter:
            self.diff_buffer.clear()

    def on_temp_window_changed(self, value):
        self.temporal_window = value
        self.diff_buffer.clear()

    def on_threshold_changed(self, state):
        self.use_threshold = (state == Qt.Checked)

    def on_threshold_value_changed(self, value):
        self.threshold_value = value

    def on_adaptive_k_changed(self, state):
        self.use_adaptive_k = (state == Qt.Checked)
        # Обнулим в AdaptiveK текущее значение, чтобы начать с init_k
        self.adaptive_k.k = self.spin_init_k.value()

    def on_init_k_changed(self, value):
        self.base_k = value
        # Если адаптивное k выключено, то base_k используется напрямую.
        # Если включено, можно обновить текущий k:
        if not self.use_adaptive_k:
            self.adaptive_k.k = value

    def on_learn_rate_changed(self, value):
        self.adaptive_k.learning_rate = value

    #############################################
    # Чтение кадров и обработка (таймер ~30мс)
    #############################################
    def update_frame(self):
        if self.hCamera is None:
            return
        try:
            # Получить RAW-кадр
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            # Прогнать через ISP (в pFrameBuffer -> BGR8 / MONO8)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

            # Под Windows надо перевернуть
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, FrameHead, 1)

            # Сформировать numpy
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)

            ch = 1 if (FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8) else 3
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, ch))

            # Вывод "Original"
            pix_orig = convert_frame_to_qpixmap(frame)
            self.label_orig.setPixmap(pix_orig)

            # Копим кадры для нужного режима
            self.frame_buffer.append(frame)

            # Вычислим diff в зависимости от режима
            current_k = self.adaptive_k.k if self.use_adaptive_k else self.base_k

            diff = None
            if self.mode == 1:
                # 30 Гц → нужно 2 кадра
                diff = process_mode_30(self.frame_buffer, current_k)
                if diff is not None:
                    # Урезаем буфер, чтобы в frame_buffer[-2:] были последние два кадра
                    self.frame_buffer = self.frame_buffer[-1:]
            elif self.mode == 2:
                # 15 Гц → нужно 4 кадра
                diff = process_mode_15(self.frame_buffer, current_k)
                if diff is not None:
                    # Оставим последние 2 кадра, чтобы они стали "A" для следующего цикла
                    self.frame_buffer = self.frame_buffer[-2:]
            elif self.mode == 3:
                # 20 Гц → нужно 3 кадра
                diff = process_mode_20(self.frame_buffer, current_k)
                if diff is not None:
                    # Сбрасываем полностью
                    self.frame_buffer.clear()

            # Если diff сформировался
            if diff is not None:
                # Временная медианная фильтрация
                if self.use_temporal_filter:
                    self.diff_buffer.append(diff)
                    if len(self.diff_buffer) > self.temporal_window:
                        self.diff_buffer.pop(0)
                    if len(self.diff_buffer) == self.temporal_window:
                        diff = temporal_median_filter(self.diff_buffer)
                else:
                    self.diff_buffer.clear()

                # Адаптивное k: обновим коэффициент на основе "темного" кадра (например, A)
                # Но для 30 Гц "темный" – frame_buffer[-1] (или diff??).
                # Логичнее брать кадр, который предположительно "без подсветки".
                if self.use_adaptive_k:
                    if self.mode == 1 and len(self.frame_buffer) >= 1:
                        dark_frame = self.frame_buffer[-1]
                        self.adaptive_k.update(dark_frame)
                    elif self.mode == 2 and len(self.frame_buffer) >= 2:
                        # "темные" A,A → усредним
                        A_avg = (self.frame_buffer[-2].astype(np.float32) +
                                 self.frame_buffer[-1].astype(np.float32)) / 2
                        self.adaptive_k.update(A_avg)
                    elif self.mode == 3:
                        # Только что "почистили" frame_buffer,
                        # поэтому придется помнить, какой кадр был "A" (сложнее).
                        # Можно до clear() взять group[-3] и group[-1], но это уже сделано в process_mode_20.
                        pass

                # Адаптивный порог
                if self.use_threshold:
                    diff = apply_threshold(diff, self.threshold_value)

                pix_proc = convert_frame_to_qpixmap(diff)
                self.label_proc.setPixmap(pix_proc)

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed:", e.message)

    #############################################
    # Завершение
    #############################################
    def closeEvent(self, event):
        """
        При закрытии окна освобождаем камеру и буфер.
        """
        if self.hCamera is not None:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = None
        if self.pFrameBuffer is not None:
            mvsdk.CameraAlignFree(self.pFrameBuffer)
            self.pFrameBuffer = None
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
