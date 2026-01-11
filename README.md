# Детекция логотипа Apple — сдача (ml_ops)

## Кратко

Я собрал решение под ТЗ: REST API, Docker, генерация синтетического датасета, обучение и валидация. Модель уже обучена, но результаты на синтетике не считаю "идеальными" для реального мира.

## Требования из ТЗ

- REST API на порту 8000
- Детекция логотипа Apple (надкусанное яблоко, любой цвет)
- Форматы: JPEG, PNG, BMP, WEBP
- Время инференса: <= 10 секунд
- Скрипт валидации с F1-score при IoU=0.5
- Публичные ссылки на веса и валидационные данные

## Структура

```
ml_ops/
  apple/
    app/                # REST API (FastAPI)
    scripts/            # генерация/обучение/валидация/экспорт примеров
    data_and_mpdel.ipynb# основной ноутбук (генерация + обучение)
    data/               # локальные данные (не коммитятся)
    runs/               # артефакты обучения (не коммитятся)
    models/             # веса (локально)
    Dockerfile
    requirements.txt
  README.md
```

## Генерация датасета (синтетика)

Я генерирую синтетические изображения из реальных логотипов с прозрачным фоном (папка `true_apple/`). На выходе получается YOLO-разметка и сплит train/val.
Основной пайплайн также оформлен в ноутбуке `apple/data_and_mpdel.ipynb`.

```bash
python apple/scripts/generate_dataset.py \
  --src apple/true_apple \
  --out apple/data \
  --num 1200 \
  --split 0.8 \
  --seed 42
```

Скрипт создаст:
- `apple/data/labeled/images`, `apple/data/labeled/labels`
- `apple/data/splits/train|val/...`
- `apple/data/apple.yaml`

## Обучение

```bash
python apple/scripts/train.py \
  --data apple/data/apple.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --project apple/runs \
  --name apple_logo
```

## Метрики и честность

Результаты на синтетике могут быть очень высокими. Это не означает идеальную модель на реальных фото. Для честной оценки нужен отдельный реальный validation набор с >=100 положительных примеров.

Финальные метрики на синтетической валидации (epoch 50, `runs/apple_logo2/results.csv`):
- Precision: 0.9997
- Recall: 1.0000
- mAP@0.5: 0.9950
- mAP@0.5:0.95: 0.9797

## Валидация (F1-score @ IoU=0.5)

```bash
python apple/scripts/validate.py \
  --images apple/data/splits/val/images \
  --labels apple/data/splits/val/labels \
  --weights apple/runs/apple_logo2/weights/best.pt \
  --conf 0.25 \
  --iou 0.5
```

## Примеры успешной/неуспешной работы

Скрипт экспортирует изображения с предсказаниями и делит их на `success` и `fail`:

```bash
python apple/scripts/export_examples.py \
  --images apple/data/splits/val/images \
  --labels apple/data/splits/val/labels \
  --weights apple/runs/apple_logo2/weights/best.pt \
  --out apple/examples \
  --conf 0.25 \
  --iou 0.5
```

Примерный сэмпл (10 success / 10 fail) лежит в `apple/examples_sample`.

## API

Запуск локально:

```bash
pip install -r apple/requirements.txt
uvicorn apple.app.main:app --host 0.0.0.0 --port 8000
```

Пример запроса:

```bash
curl -X POST "http://localhost:8000/detect" -F "file=@path/to/image.jpg"
```

## Docker

```bash
docker build -t apple-logo-detector -f apple/Dockerfile apple
docker run -p 8000:8000 -v %cd%/apple/models:/app/models apple-logo-detector
```

## Публичные ссылки (добавить)

- Веса модели: TODO
- Валидационный набор: TODO
- Обучающий набор: TODO
- Кривые обучения (results.png, BoxPR_curve.png и др.): TODO
- Примеры работы модели (success/fail): TODO

## Комментарий по требованиям

- В репозитории есть скрипты генерации синтетики и обучения.
- Данные, веса и артефакты обучения не коммитятся (слишком большие). Их нужно выложить на файлообменник и вставить ссылки сюда.
