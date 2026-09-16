# Аналіз і прогнозування ефективності спортивних тренувань

## Структура проєкту

```
fitness_project/
├── app.py                          # Streamlit-застосунок (точка входу)
├── pipeline.py                     # Головний оркестратор 
├── requirements.txt
├── trainings_dataset.csv           # Тренувальний датасет
├── data/                           
├── models/                         # Збережені ML-моделі (.pkl)
├── reports/                        # Згенеровані PDF-звіти
└── src/
    ├── database/                   
    │   └── mongo_db.py             
    ├── a1_data_intake/             # Блок A1 IDEF0
    │   ├── validator.py            # A1.1 — Валідація вхідних даних
    │   ├── cleaner.py              # A1.2 — Очищення та нормалізація
    │   └── loader.py               # A1.3 — Завантаження та збереження
    ├── a2_metrics/                 # Блок A2 IDEF0
    │   └── metrics.py              # A2.1–A2.4 — CTL, ATL, TSB, Readiness
    ├── a3_analysis/                # Блок A3 IDEF0
    │   ├── features.py             # Конструювання ознак та часових рядів
    │   ├── regression.py           # A3.1 — Прогноз активності
    │   ├── classification.py       # A3.2 — Класифікація готовності
    │   └── clustering.py           # A3.3 — Кластеризація поведінки
    └── a4_recommendations/         # Блок A4 IDEF0
        ├── recommender.py          # A4.1 — Генерація рекомендацій
        ├── visualizer.py           # A4.2 — Аналітичні графіки (Plotly)
        └── report_generator.py     # A4.3 — PDF-звіт
```

## Встановлення

```bash
pip install -r requirements.txt
```

## Перший запуск

```bash
docker run -d --name mongo-db -p 27017:27017 mongo:7
$env:MONGO_URI="mongodb://localhost:27017/"
$env:MONGO_DB_NAME="fitness_training_db"
streamlit run app.py
```
## Наступні запуски

```bash
docker start mongo-db
$env:MONGO_URI="mongodb://localhost:27017/"
$env:MONGO_DB_NAME="fitness_training_db"
streamlit run app.py
```

## Вхідний датасет

CSV-файл із колонками:

| Колонка | Тип | Опис |
|---------|-----|------|
| user_id | int | Ідентифікатор користувача |
| date | date | Дата запису |
| age | int | Вік |
| gender | str | Стать (Male/Female) |
| height_cm | float | Зріст (см) |
| weight_kg | float | Вага (кг) |
| steps | int | Кроки за день |
| calories_burned | float | Спалені калорії |
| sleep_hours | float | Тривалість сну (год) |
| heart_rate_avg | int | Середній пульс |
| workout_type | str | Тип тренування |
| workout_duration_minutes | float | Тривалість тренування (хв) |
| water_intake_liters | float | Споживання води (л) |
| stress_level | int | Рівень стресу (1–10) |
| mood | str | Настрій |
| user_type | str | Тип користувача |


## Задачі машинного навчання

### Регресія (прогноз кроків)
- **Цільова змінна:** `steps_next = Steps(t+1)` — кроки наступного дня
- **Ознаки:** лагові (lag1, lag3, lag7, lag14), ковзні середні MA7/MA14, CTL/ATL/TSB
- **Розбиття:** TimeSeriesSplit (80/20, часовий порядок збережено)
- **Моделі:** LinearRegression, Ridge, RandomForest, GradientBoosting, XGBoost

### Класифікація рівня готовності
- **Цільова змінна:** `Readiness_Level` (Низька/Середня/Висока) — rule-based мітки
- **Ознаки:** TSB, sleep_hours, heart_rate_avg, ATL, лагові ознаки
- **Моделі:** LogisticRegression, RandomForest, GradientBoosting

### Кластеризація
- **Вхід:** агрегований профіль користувача (середні за весь період)
- **Алгоритм:** KMeans з визначенням k через силуетний коефіцієнт
- **Виходи:** типи поведінки (Малоактивний / Помірно активний / Активний)
