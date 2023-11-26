import cv2

# создаем объект для захвата видеопотока
cap = cv2.VideoCapture(0)

# создаем классификатор Хаара для обнаружения жестов рук
hand_cascade = cv2.CascadeClassifier('haarcascade_hand.xml')

# создаем словарь с соответствиями жестов и соответствующих текстов
gesture_names = {
    'rock': 'Камень',
    'paper': 'Бумага',
    'scissors': 'Ножницы'
}

# цикл для обработки каждого кадра видеопотока
while True:
    # захватываем кадр из видеопотока
    ret, frame = cap.read()

    # преобразуем кадр в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # обнаруживаем жесты рук на кадре
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

    # проходимся по всем обнаруженным жестам рук
    for (x,y,w,h) in hands:
        # вырезаем область с жестом руки из кадра
        hand_roi = gray[y:y+h, x:x+w]

        # проводим классификацию жеста
        # здесь должен быть ваш код для классификации жеста
        gesture = 'paper' # здесь для примера мы всегда используем жест "бумага"

        # получаем соответствующий текст для жеста
        gesture_text = gesture_names.get(gesture, 'gg')

        # выводим текст на экран
        cv2.putText(frame, gesture_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # рисуем прямоугольник вокруг жеста
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    # выводим кадр на экран
    cv2.imshow('Hand Gestures', frame)

    # проверяем нажатие клавиши "q" для выхода из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
