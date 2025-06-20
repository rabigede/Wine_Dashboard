first_page = {
    'FIO': 'Макеев Всеволод Юрьевич',
    'GROUP': 'ФИТ-232',
    'TOPIC': 'Модель для классификации вина по цветам',
    'title': 'Информация о разработчике 👨‍💻',
    'page_title': "О разработчике",
    'page_icon': "👨‍💻",
    'markdown': """
Это веб-приложение предназначено для демонстрации моделей машинного обучения, построенных на данных о вине.  
На следующих страницах вы сможете:
- ознакомиться с данными и их анализом;
- увидеть визуализации;
- получить предсказание качества вина на основе характеристик.
""",
}


second_page = {
    'markdown_1': """
- Признак `color` добавлен вручную при объединении красного и белого вина.
- Пропущенные значения отсутствуют.
- Все признаки, кроме `color`, являются числовыми.
""",
    "domain_description": """
Набор данных содержит информацию о химическом составе и качестве красного и белого вина. 
Он используется для задач регрессии и классификации в машинном обучении, где цель — предсказать 
оценку качества вина на основе его химических свойств.
""",
    "features": {
        "fixed acidity": "Постоянная кислотность (г/дм³)",
        "volatile acidity": "Летучая кислотность (г/дм³)",
        "citric acid": "Лимонная кислота (г/дм³)",
        "residual sugar": "Остаточный сахар (г/дм³)",
        "chlorides": "Хлориды (г/дм³)",
        "free sulfur dioxide": "Свободный диоксид серы (мг/дм³)",
        "total sulfur dioxide": "Общий диоксид серы (мг/дм³)",
        "density": "Плотность (г/см³)",
        "pH": "Уровень pH",
        "sulphates": "Сульфаты (г/дм³)",
        "alcohol": "Содержание алкоголя (% об.)",
        "quality": "Оценка качества (от 0 до 10)",
        "color": "Цвет вина (white/red)"
    },
    "preprocessing_notes": """
Цвет вина был добавлен вручную на основе объединения двух отдельных CSV-файлов (белого и красного вина).
Других серьёзных изменений не проводилось. Пропущенные значения отсутствуют.
Все признаки числовые, кроме признака цвета.
"""
}


third_page = {
    "page_title": "Визуализации зависимостей",
    "description": """
    Ниже представленные графики, показывающие влияние различных факторов на целевую переменную
    """
}

fourth_page = {
    "title": "Предсказание цвета вина",
    "target": "Цвет вина"
}
