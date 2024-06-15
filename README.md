# AtomicHack2.0

## Краткое описание идеи

Проект разрабатывается с целью автоматизации процесса поиска дефектов в области сварных швов.

Исходными данными являются фотографии с изображениями сварных швов.

На сайте проекта необходимо загрузить фотографии путем перетаскивания их в область загрузки, либо путем выбора папки с фотографиями.

Загруженные фотографии обрабатываются с помощью нейронной сети (нейронных сетей) на предмет наличия дефектов, которая будет производить классификацию изображений.

В реализации проекта будут использованы 2 модели. 

Первая находит дефект на изображении, вторая классифицирует найденный дефект.

В результате будет изображение сварного шва с отмеченными дефектами и указанием типов дефектов.

## Установка и запуск проекта (Python 3.10). TODO добавить расширенные конфигурации системы
1. Установить необходимые библиотеки
```bash
pip install -r requirements.txt
```
2. Запустить локально сервер:
```bash
python -m flask --app app run
```

## Инструкция по использованию
В `detection_detector` лежит demo.png сейчас можете его загружать, нажимаете отправить и получаете размеченный результат как картинку.


## Проект

### Проблема

В нашем понимании проблема состоит в следующем. Проверка состояния сварных швов, контроль качества сварки в настоящее время является чрезвычайно трудоемким процессом. Особенно, если швы располагаются в труднодоступных или опасных местах. Проверка занимает значительное количество трудовых ресурсов. 

Помочь исправить сложившуюся ситуацию может использование нейронных сетей, а именно сетей, связанных с детектированием объектов. В нашем случае стоят две задачи: детектирование - определение наличия дефектов сварного шва и классификация - распознавание вида дефекта.

### Исходые данные

Для решения задач детектирования и классификации использовался датасет, полученный от организаторов хакатона АО «Атомэнергомаш». Датасет содержит 1162 цветные фотографии различных деталей со сварными швами с дефектами и без дефектов. Сфотографированы 12 деталей, длинные сварные швы сфотографированы вдоль по частям с интервалом, так, что нерересекающиеся изображения отстоят друг от друга примерно на 14-17 фотографий. Изображения в основном качественные, но попадаются и размытые, как образец №12. Все дефекты на изображениях отмечены прямоугольниками, координаты дефектов заданы.

### Архитектура решения

Для реализации проекта и обучения модели была выбрана архитектура YOLO и программные модули, реализованные на этой архитектуре, которые является свободным программным обеспечением. Для обучения моделей были использованы две версии YOLO - v9 и v10. То-есть проведены два обучения с разными датасетами. 

Основное обучение проводилось на версии v9, поскольку она входит в стандартный установочный пакет YOLO (ultralitics), а v10 - нет и это осложняет реализацию проекта.

# Артем, напиши про свой проект, датасет, как тренировался, результаты.

Датасет для обучения на версии 10 формировался в таком порядке: для валидационного датасета выбиралось каждое десятое изображение. Остальные изображения включались в тренировочный датасет. При таком разбиении каждая часть шва  представлена и в тренировочном и в валидационном датасете. Все фотографии были приведены к размеру 640х640 пикселей.
Изначально дефекты были поделены на 5 классов. Но, поскольку были фотографии и без дефектов, решено было добавить шестой класс - без дефектов. Метка на этих изображених - квадрат, центр которого совпадает с центром изображения, а ширина и высота равны одной десятой ширины и высоты изображения соответственно. Обучение проводилось на платформе Yandex DataSphere.
Размер батча данных подбирался для полной загрузки памяти GPU. Все параметры обучения модели были установлены по умолчанию.
Лучший результат обучения был зафиксирован на 152-й эпохе.
