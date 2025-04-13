import os
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
import random
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Константы и настройки
NUM_TRANSFORMERS = 100
BASE_YEAR = 2010
CRITICAL_VALUES = {
    "electric_strength": 38,
    "moisture_content": 15,
    "dielectric_loss_tangent": 0.03,
}
WARNING_RATIO = 0.9
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
base_folder = os.path.join(desktop, "Трансформаторы_Анализ")

# 1. Функции инициализации
def setup_folders():
    """Создает базовую структуру папок"""
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        print(f"Создана базовая папка: {base_folder}")
    else:
        print(f"Используется существующая папка: {base_folder}")

def create_database():
    """Создает тестовую базу данных"""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Создание таблиц
    cursor.execute("""
    CREATE TABLE transformers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        factory_number TEXT UNIQUE NOT NULL,
        manufacture_date TEXT NOT NULL,
        power_type TEXT NOT NULL,
        voltage_class TEXT NOT NULL,
        location TEXT NOT NULL,
        quality_group INTEGER NOT NULL
    )""")

    cursor.execute("""
    CREATE TABLE measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transformer_id INTEGER NOT NULL,
        inspection_date TEXT NOT NULL,
        operation_years INTEGER NOT NULL,
        electric_strength REAL NOT NULL,
        moisture_content REAL NOT NULL,
        dielectric_loss_tangent REAL NOT NULL,
        FOREIGN KEY (transformer_id) REFERENCES transformers (id)
    )""")

    # Заполнение тестовыми данными
    QUALITY_GROUPS = {
        1: {"es_decay": 0.005, "mc_growth": 0.01, "tgd_growth": 0.002},
        2: {"es_decay": 0.015, "mc_growth": 0.03, "tgd_growth": 0.004},
        3: {"es_decay": 0.025, "mc_growth": 0.05, "tgd_growth": 0.008},
    }

    power_types = ["100MVA", "150MVA", "200MVA"]
    voltage_classes = ["110kV", "220kV", "330kV"]
    locations = ["Подстанция №1", "Подстанция №2", "ГПП", "ТЭЦ-3", "ПС Северная"]

    for i in range(1, NUM_TRANSFORMERS + 1):
        quality_group = random.choices([1, 2, 3], weights=[7, 2, 1])[0]
        params = QUALITY_GROUPS[quality_group]

        manufacture_year = BASE_YEAR + random.randint(0, 5)
        manufacture_date = f"{manufacture_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

        cursor.execute(
            """INSERT INTO transformers 
            (factory_number, manufacture_date, power_type, voltage_class, location, quality_group)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                f"TN-{manufacture_year}-{i:03d}",
                manufacture_date,
                random.choice(power_types),
                random.choice(voltage_classes),
                random.choice(locations),
                quality_group,
            ),
        )

        tf_id = cursor.lastrowid

        base_es = 45 + random.uniform(-2, 2)
        base_mc = 8 + random.uniform(-1, 1)
        base_tgd = 0.015 + random.uniform(-0.003, 0.003)

        for years in [0, 3, 6, 9, 12, 15]:
            es = base_es * (1 - years * params["es_decay"] * random.uniform(0.8, 1.2))
            mc = base_mc * (1 + years * params["mc_growth"] * random.uniform(0.8, 1.2))
            tgd = base_tgd * (1 + years * params["tgd_growth"] * random.uniform(0.8, 1.2))

            if quality_group == 3 and random.random() < 0.3:
                if random.random() < 0.5:
                    es *= random.uniform(0.7, 0.9)
                else:
                    mc *= random.uniform(1.2, 1.5)

            inspection_date = (
                datetime.strptime(manufacture_date, "%Y-%m-%d")
                + timedelta(days=years * 365)
            ).strftime("%Y-%m-%d")

            cursor.execute(
                """INSERT INTO measurements 
                (transformer_id, inspection_date, operation_years,
                 electric_strength, moisture_content, dielectric_loss_tangent)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    tf_id,
                    inspection_date,
                    years,
                    max(20, round(es, 2)),
                    min(25, round(mc, 2)),
                    min(0.05, round(tgd, 5)),
                ),
            )

    conn.commit()
    return conn

# 2. Функции анализа данных
def analyze_parameter(values, param_name, years):
    """Анализирует параметры трансформатора"""
    current_value = values[-1]
    critical = CRITICAL_VALUES[param_name]
    warning = critical * WARNING_RATIO

    if param_name == "electric_strength":
        is_critical = current_value < critical
        is_warning = current_value < warning
    else:
        is_critical = current_value > critical
        is_warning = current_value > warning

    status = "CRITICAL" if is_critical else "WARNING" if is_warning else "NORMAL"
    
    if len(years) > 1:
        slope, _, _, _, _ = linregress(years, values)
    else:
        slope = 0

    return {
        "current_value": current_value,
        "status": status,
        "trend": slope,
        "unit": "кВ" if param_name == "electric_strength" else "ppm" if param_name == "moisture_content" else "",
    }

def predict_time_to_failure(values, critical_value, param_name):
    """Прогнозирует время до отказа"""
    years = list(range(len(values)))
    try:
        slope, intercept, _, _, _ = linregress(years, values)
        if slope == 0:
            return None

        if param_name == "electric_strength":
            time_to_failure = (critical_value - intercept) / slope - years[-1]
        else:
            time_to_failure = (critical_value - intercept) / slope - years[-1]

        return max(0, time_to_failure)
    except:
        return None

# 3. ML-функции
def predict_with_ml(measurements):
    """Прогнозирование с использованием ML"""
    if len(measurements) < 3:
        return None
    
    years = np.array([m[0] for m in measurements]).reshape(-1, 1)
    es = np.array([m[1] for m in measurements])
    mc = np.array([m[2] for m in measurements])
    tgd = np.array([m[3] for m in measurements])
    
    time_horizons = [1, 3, 5]
    predictions = {}
    
    for param, values in zip(['electric_strength', 'moisture_content', 'dielectric_loss_tangent'], [es, mc, tgd]):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(years, values)
        
        param_predictions = {}
        for horizon in time_horizons:
            rf_pred = rf.predict([[years[-1][0] + horizon]])[0]
            param_predictions[horizon] = rf_pred
        
        predictions[param] = param_predictions
    
    return predictions

def cluster_transformers(conn):
    """Улучшенная функция кластеризации с отображением конкретных трансформаторов"""
    cursor = conn.cursor()
    
    # Получаем данные с именами трансформаторов
    cursor.execute("""
    SELECT t.id, t.factory_number, t.quality_group, 
           AVG(m.electric_strength), AVG(m.moisture_content), 
           AVG(m.dielectric_loss_tangent), MAX(m.operation_years)
    FROM transformers t
    JOIN measurements m ON t.id = m.transformer_id
    GROUP BY t.id
    """)
    data = cursor.fetchall()
    
    if len(data) < 3:
        return None
    
    # Отделяем идентификаторы и названия от параметров
    tf_ids = [d[0] for d in data]
    tf_names = [d[1] for d in data]
    X = np.array([d[3:] for d in data])  # Параметры для кластеризации
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    cluster_analysis = []
    for cluster_id in range(3):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
            
        # Получаем данные для кластера
        cluster_data = X[cluster_indices]
        cluster_tf_names = [tf_names[i] for i in cluster_indices]
        
        analysis = {
            'cluster': cluster_id,
            'count': len(cluster_indices),
            'transformers': cluster_tf_names,  # Список имен трансформаторов
            'avg_es': np.mean(cluster_data[:, 0]),
            'avg_mc': np.mean(cluster_data[:, 1]),
            'avg_tgd': np.mean(cluster_data[:, 2]),
            'avg_age': np.mean(cluster_data[:, 3]),
        }
        cluster_analysis.append(analysis)
    
    return cluster_analysis
# 4. Функции отчетов
def create_transformer_folder(tf_name):
    """Создает папку для трансформатора"""
    tf_folder = os.path.join(base_folder, tf_name)
    os.makedirs(tf_folder, exist_ok=True)
    return tf_folder

def predict_failure(measurements):
    """Анализирует состояние трансформатора"""
    if not measurements:
        return {
            "overall_status": "⚠️ НЕТ ДАННЫХ",
            "next_check": "Требуется проверка",
            "parameters": {},
            "predictions": {},
        }

    years = [m[0] for m in measurements]
    es = [m[1] for m in measurements]
    mc = [m[2] for m in measurements]
    tgd = [m[3] for m in measurements]

    current_status = {
        "electric_strength": analyze_parameter(es, "electric_strength", years),
        "moisture_content": analyze_parameter(mc, "moisture_content", years),
        "dielectric_loss_tangent": analyze_parameter(tgd, "dielectric_loss_tangent", years),
    }

    if any(s["status"] == "CRITICAL" for s in current_status.values()):
        overall_status = "❌ КРИТИЧЕСКОЕ СОСТОЯНИЕ"
        next_check = "Немедленно"
    elif any(s["status"] == "WARNING" for s in current_status.values()):
        overall_status = "⚠️ СОСТОЯНИЕ ПОВЫШЕННОГО ВНИМАНИЯ"
        next_check = "Через 1 год"
    else:
        overall_status = "✅ НОРМАЛЬНОЕ СОСТОЯНИЕ"
        next_check = "Через 3 года"

    predictions = {}
    for param in current_status:
        if current_status[param]["status"] not in ["CRITICAL", "WARNING"]:
            param_values = [m[1] if param == "electric_strength" else 
                          m[2] if param == "moisture_content" else 
                          m[3] for m in measurements]
            pred = predict_time_to_failure(param_values, CRITICAL_VALUES[param], param)
            predictions[param] = pred

    return {
        "overall_status": overall_status,
        "next_check": next_check,
        "parameters": current_status,
        "predictions": predictions,
    }

def generate_report(tf_data, measurements, tf_folder):
    """Генерирует текстовый отчет"""
    report_path = os.path.join(tf_folder, "1_Заключение.txt")
    analysis = predict_failure(measurements)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Аналитический отчет по трансформатору {tf_data['factory_number']}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Место установки: {tf_data['location']}\n")
        f.write(f"Тип: {tf_data['power_type']} {tf_data['voltage_class']}\n")
        f.write(f"Год выпуска: {tf_data['manufacture_date'][:4]}\n")
        f.write(f"Лет эксплуатации: {measurements[-1][0]}\n\n")
        
        f.write(f"ОБЩИЙ СТАТУС: {analysis['overall_status']}\n")
        f.write(f"Рекомендуемый срок следующей проверки: {analysis['next_check']}\n\n")
        
        f.write("ДЕТАЛЬНЫЙ АНАЛИЗ ПАРАМЕТРОВ:\n")
        f.write("-"*50 + "\n")
        for param, data in analysis["parameters"].items():
            param_name = param.replace("_", " ").upper()
            f.write(f"{param_name}:\n")
            f.write(f"  Текущее значение: {data['current_value']} {data['unit']}\n")
            if data["status"] == "CRITICAL":
                f.write("  ⚠️ КРИТИЧЕСКОЕ ЗНАЧЕНИЕ! Превышены допустимые нормы\n")
            elif data["status"] == "WARNING":
                f.write("  ⚠ ВНИМАНИЕ: Приближение к критическим значениям\n")
            else:
                f.write("  ✅ В пределах нормы\n")
            f.write(f"  Тренд: {data['trend']:.3f} {data['unit']}/год\n\n")

        if analysis["predictions"]:
            f.write("\nПРОГНОЗ ВРЕМЕНИ ДО КРИТИЧЕСКИХ ЗНАЧЕНИЙ:\n")
            f.write("-"*50 + "\n")
            for param, time in analysis["predictions"].items():
                if time is not None:
                    param_name = param.replace("_", " ").upper()
                    f.write(f"{param_name}: ~{time:.1f} лет до критического значения\n")

        f.write("\nРЕКОМЕНДАЦИИ:\n")
        f.write("-"*50 + "\n")
        if analysis["overall_status"].startswith("❌"):
            f.write("1. НЕМЕДЛЕННО ВЫВЕСТИ ИЗ ЭКСПЛУАТАЦИИ\n")
            f.write("2. Провести полную диагностику\n")
            f.write("3. Запланировать капитальный ремонт или замену\n")
        elif analysis["overall_status"].startswith("⚠️"):
            f.write("1. Увеличить частоту проверок до 1 раза в год\n")
            f.write("2. Провести внеплановую диагностику\n")
            f.write("3. Рассмотреть возможность планового ремонта\n")
        else:
            f.write("1. Продолжить плановые проверки каждые 3 года\n")
            f.write("2. Мониторить ключевые параметры\n")

def create_colored_table(measurements, tf_folder):
    """Создает цветную таблицу измерений"""
    df = pd.DataFrame(
        measurements,
        columns=["Лет эксплуатации", "Эл.прочность (кВ)", "Влагосод. (ppm)", "Tan δ"],
    )

    cell_width = 150
    cell_height = 40
    cols = len(df.columns)
    rows = len(df) + 1

    img_width = cell_width * cols
    img_height = cell_height * rows

    img = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    colors = {
        "header": (200, 200, 200),
        "normal": (255, 255, 255),
        "warning": (255, 255, 150),
        "critical": (255, 150, 150),
        "border": (100, 100, 100),
        "text": (0, 0, 0),
        "text_warning": (150, 100, 0),
        "text_critical": (150, 0, 0),
    }

    for col_idx, col_name in enumerate(df.columns):
        x0 = col_idx * cell_width
        y0 = 0
        x1 = x0 + cell_width
        y1 = y0 + cell_height

        draw.rectangle([x0, y0, x1, y1], fill=colors["header"])
        draw.rectangle([x0, y0, x1, y1], outline=colors["border"])

        bbox = draw.textbbox((0, 0), col_name, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.text(
            (x0 + (cell_width - text_width) / 2, y0 + (cell_height - text_height) / 2),
            col_name,
            fill=colors["text"],
            font=font,
        )

    for row_idx, row in df.iterrows():
        for col_idx, col_name in enumerate(df.columns):
            value = row[col_name]
            x0 = col_idx * cell_width
            y0 = (row_idx + 1) * cell_height
            x1 = x0 + cell_width
            y1 = y0 + cell_height

            if col_name == "Эл.прочность (кВ)":
                if value < CRITICAL_VALUES["electric_strength"] * WARNING_RATIO:
                    cell_color = colors["critical"]
                    text_color = colors["text_critical"]
                elif value < CRITICAL_VALUES["electric_strength"]:
                    cell_color = colors["warning"]
                    text_color = colors["text_warning"]
                else:
                    cell_color = colors["normal"]
                    text_color = colors["text"]
            elif col_name == "Влагосод. (ppm)":
                if value > CRITICAL_VALUES["moisture_content"] / WARNING_RATIO:
                    cell_color = colors["critical"]
                    text_color = colors["text_critical"]
                elif value > CRITICAL_VALUES["moisture_content"]:
                    cell_color = colors["warning"]
                    text_color = colors["text_warning"]
                else:
                    cell_color = colors["normal"]
                    text_color = colors["text"]
            elif col_name == "Tan δ":
                if value > CRITICAL_VALUES["dielectric_loss_tangent"] / WARNING_RATIO:
                    cell_color = colors["critical"]
                    text_color = colors["text_critical"]
                elif value > CRITICAL_VALUES["dielectric_loss_tangent"]:
                    cell_color = colors["warning"]
                    text_color = colors["text_warning"]
                else:
                    cell_color = colors["normal"]
                    text_color = colors["text"]
            else:
                cell_color = colors["normal"]
                text_color = colors["text"]

            draw.rectangle([x0, y0, x1, y1], fill=cell_color)
            draw.rectangle([x0, y0, x1, y1], outline=colors["border"])

            if isinstance(value, float):
                text = f"{value:.2f}" if value < 1 else f"{value:.1f}"
            else:
                text = str(value)

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            draw.text(
                (x0 + (cell_width - text_width) / 2, y0 + (cell_height - text_height) / 2),
                text,
                fill=text_color,
                font=font,
            )

    table_path = os.path.join(tf_folder, "2_Измерения.png")
    img.save(table_path, quality=95)
    print(f"Цветная таблица сохранена: {table_path}")

def plot_transformer_data(measurements, tf_data, tf_folder):
    """Создает графики параметров"""
    graph_path = os.path.join(tf_folder, "3_Графики.png")

    years = [m[0] for m in measurements]
    es = [m[1] for m in measurements]
    mc = [m[2] for m in measurements]
    tgd = [m[3] for m in measurements]

    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Трансформатор {tf_data['factory_number']}", y=1.02)

    plt.subplot(3, 1, 1)
    plt.plot(years, es, marker="o", color="blue")
    plt.axhline(y=CRITICAL_VALUES["electric_strength"], color="r", linestyle="--", label="Критический уровень")
    plt.title("Электрическая прочность")
    plt.ylabel("кВ")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(years, mc, marker="s", color="green")
    plt.axhline(y=CRITICAL_VALUES["moisture_content"], color="r", linestyle="--", label="Критический уровень")
    plt.title("Влагосодержание")
    plt.ylabel("ppm")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(years, tgd, marker="^", color="orange")
    plt.axhline(y=CRITICAL_VALUES["dielectric_loss_tangent"], color="r", linestyle="--", label="Критический уровень")
    plt.title("Тангенс диэлектрических потерь")
    plt.xlabel("Лет эксплуатации")
    plt.ylabel("tan δ")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close()

def generate_ml_report(cluster_analysis, tf_folder):
    """Улучшенный отчет по кластеризации с выводом списка трансформаторов"""
    if not cluster_analysis:
        return
    
    report_path = os.path.join(tf_folder, "4_Кластерный_анализ.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Кластерный анализ трансформаторов\n")
        f.write("="*50 + "\n\n")
        f.write(f"Всего проанализировано трансформаторов: {sum(c['count'] for c in cluster_analysis)}\n\n")
        
        for cluster in cluster_analysis:
            f.write(f"\nКластер #{cluster['cluster'] + 1} ({cluster['count']} трансформаторов):\n")
            f.write("-"*50 + "\n")
            
            # Вывод списка трансформаторов в кластере
            f.write("Трансформаторы в кластере:\n")
            for i, tf_name in enumerate(cluster['transformers']):
                if i % 5 == 0 and i != 0:  # 5 трансформаторов в строке
                    f.write("\n")
                f.write(f"{tf_name}, ")
            f.write("\n\n")
            
            f.write("Средние параметры:\n")
            f.write(f"- Электрическая прочность: {cluster['avg_es']:.1f} кВ\n")
            f.write(f"- Влагосодержание: {cluster['avg_mc']:.1f} ppm\n")
            f.write(f"- Тангенс диэлектрических потерь: {cluster['avg_tgd']:.5f}\n")
            f.write(f"- Средний возраст: {cluster['avg_age']:.1f} лет\n")
            
            # Рекомендации для кластера
            f.write("\nРекомендации:\n")
            if cluster['avg_es'] < CRITICAL_VALUES['electric_strength'] * 1.1:
                f.write("- Критическая электрическая прочность! Требуется проверка\n")
            if cluster['avg_mc'] > CRITICAL_VALUES['moisture_content'] * 0.9:
                f.write("- Высокое влагосодержание! Необходима сушка\n")
            if cluster['avg_tgd'] > CRITICAL_VALUES['dielectric_loss_tangent'] * 0.9:
                f.write("- Высокий tan δ! Требуется диагностика изоляции\n")
            
            f.write("\n")
# 5. Главная функция анализа
def analyze_all_transformers(conn):
    """Анализирует все трансформаторы"""
    cursor = conn.cursor()
    cluster_analysis = cluster_transformers(conn)
    
    cursor.execute("SELECT id, factory_number FROM transformers")
    transformers = cursor.fetchall()

    for tf_id, tf_name in transformers:
        try:
            tf_folder = create_transformer_folder(tf_name)
            print(f"Обработка трансформатора: {tf_name}")

            cursor.execute(
                """SELECT factory_number, manufacture_date, power_type, voltage_class, location
                FROM transformers WHERE id = ?""",
                (tf_id,),
            )
            row = cursor.fetchone()
            if not row:
                continue

            tf_data = dict(zip(
                ["factory_number", "manufacture_date", "power_type", "voltage_class", "location"],
                row
            ))

            cursor.execute(
                """SELECT operation_years, electric_strength, moisture_content, dielectric_loss_tangent
                FROM measurements WHERE transformer_id = ? ORDER BY operation_years""",
                (tf_id,),
            )
            measurements = cursor.fetchall()

            if not measurements:
                print(f"Нет данных измерений для трансформатора {tf_name}")
                continue

            generate_report(tf_data, measurements, tf_folder)
            create_colored_table(measurements, tf_folder)
            plot_transformer_data(measurements, tf_data, tf_folder)
            
            if cluster_analysis:
                generate_ml_report(cluster_analysis, tf_folder)

        except Exception as e:
            print(f"Ошибка при обработке трансформатора {tf_name}: {str(e)}")
            continue

    print(f"\nОтчеты успешно сгенерированы в папке: {base_folder}")

# 6. Запуск программы
if __name__ == "__main__":
    setup_folders()
    conn = create_database()
    analyze_all_transformers(conn)
    conn.close()