import cv2
import numpy as np

# Инициализировать камеру
cap = cv2.VideoCapture(0)

# Создать объект BackgroundSubtractorMOG2
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Захватить кадр из камеры
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Вычитание фона
    fgmask = fgbg.apply(gray)

    # Применение операций морфологического преобразования
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Нахождение контуров руки
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = None
    max_area = 0

    for contour in contours:
        # Игнорировать слишком маленькие контуры
        area = cv2.contourArea(contour)
        if area < 1000  :
            continue

        # Найти контур с максимальной площадью
        if area > max_area:
            max_area = area
            cnt = contour

    # Отрисовка контура руки на кадре
    if cnt is not None:
        cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 2)

    # Показать результат
    cv2.imshow("Frame", frame)

    # Прервать цикл по нажатию клавиши "q"
    if cv2.waitKey(1) == ord('q'):
        break

# Остановить захват кадров и освободить ресурсы
cap.release()
cv2.destroyAllWindows()
