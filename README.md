# AtomicHack2.0

## Краткое описание идеи

Проект разрабатывается с целью автоматизации процесса поиска дефектов в области сварных швов.

Исходными данными являются фотографии с изображениями сварных швов.

На сайте проекта необходимо загрузить фотографии путем перетаскивания их в область загрузки, либо путем выбора папки с фотографиями.

Загруженные фотографии обрабатываются с помощью нейронной сети (нейронных сетей) на предмет наличия дефектов, которая будет производить классификацию изображений.

В реализации проекта будут использованы 2 модели. 

Первая находит дефект на изображении, вторая классифицирует найденный дефект.

В результате будет изображение сварного шва с отмеченными дефектами и указанием типов дефектов.