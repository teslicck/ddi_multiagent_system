"""Adult teaching calculators. Each tool schema also describes its input units."""
from math import sqrt
from typing import Annotated, Literal

from pydantic import Field


Weight = Annotated[float, Field(strict=True, ge=20, le=450, title="Масса тела, кг")]
Height = Annotated[float, Field(strict=True, ge=100, le=250, title="Рост, см")]
Age = Annotated[int, Field(strict=True, ge=18, le=120, title="Возраст, лет")]
Creatinine = Annotated[float, Field(strict=True, ge=0.1, le=25, title="Креатинин сыворотки, мг/дл")]
Eyes = Annotated[int, Field(strict=True, ge=1, le=4, title="Открывание глаз (E), баллы")]
Verbal = Annotated[int, Field(strict=True, ge=1, le=5, title="Речевая реакция (V), баллы")]
Motor = Annotated[int, Field(strict=True, ge=1, le=6, title="Двигательная реакция (M), баллы")]
Resp = Annotated[int, Field(strict=True, ge=1, le=100, title="Частота дыхания, в минуту")]
Pressure = Annotated[int, Field(strict=True, ge=20, le=300, title="Систолическое АД, мм рт. ст.")]


def result(value, unit, components, interpretation, advice, reference, color="info"):
    return dict(value=value, unit=unit, components=components, interpretation=interpretation,
                advice=advice, reference=reference, color=color)


def bmi(weight: Weight, height: Height) -> dict:
    """Антропометрия · Индекс массы тела (BMI), для взрослых от 20 лет."""
    value = weight / (height / 100) ** 2
    if value < 18.5:
        label, color = "Недостаточная масса тела", "warning"
    elif value < 25:
        label, color = "Масса тела в референсном диапазоне", "success"
    elif value < 30:
        label, color = "Избыточная масса тела", "warning"
    else:
        label, color = "Ожирение по критериям ИМТ", "warning"
    return result(round(value, 2), "кг/м²", {"Масса, кг": weight, "Рост, см": height},
                  label, "Учитывайте телосложение и другие показатели здоровья.",
                  "https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html", color)


def cockcroft_gault(age: Age, weight: Weight, creatinine: Creatinine,
                    sex: Literal["мужской", "женский"]) -> dict:
    """Нефрология · Клиренс креатинина по Кокрофту–Голту (Cockcroft–Gault)."""
    factor = 0.85 if sex == "женский" else 1.0
    clearance = (140 - age) * weight * factor / (72 * creatinine)
    return result(round(clearance, 1), "мл/мин",
                  {"Возраст, лет": age, "Масса, кг": weight, "Креатинин, мг/дл": creatinine,
                   "Коэффициент пола": factor}, "Оценка клиренса при стабильном креатинине",
                  "Проверьте выбор расчётной массы тела и единицы измерения перед интерпретацией.",
                  "https://pubmed.ncbi.nlm.nih.gov/1244564/")


def glasgow(eyes: Eyes, verbal: Verbal, motor: Motor) -> dict:
    """Неврология · Шкала комы Глазго (Glasgow Coma Scale, GCS). Введите три оценённых компонента."""
    score = eyes + verbal + motor
    color = "error" if score <= 8 else "warning" if score < 15 else "success"
    label = "Максимальный балл" if score == 15 else "Снижение уровня сознания"
    return result(score, "баллов из 15", {"E": eyes, "V": verbal, "M": motor}, label,
                  "Оценивайте динамику E, V и M отдельно. При невозможности оценить компонент сумму не используют.",
                  "https://doi.org/10.1016/S0140-6736(14)60120-5", color)


def mosteller(weight: Weight, height: Height) -> dict:
    """Антропометрия · Площадь поверхности тела по Мостеллеру (Mosteller BSA)."""
    return result(round(sqrt(weight * height / 3600), 3), "м²",
                  {"Рост, см": height, "Масса, кг": weight}, "Расчётная площадь поверхности тела",
                  "Сверьте рост и массу с актуальными измерениями.",
                  "https://pubmed.ncbi.nlm.nih.gov/3657876/")


def qsofa(respirations: Resp, systolic: Pressure,
          altered_mentation: Annotated[bool, Field(strict=True, title="Изменённое сознание")]) -> dict:
    """Реанимация · Быстрая оценка риска при подозрении на инфекцию (qSOFA), вне ОРИТ."""
    components = {"ЧДД ≥ 22/мин": int(respirations >= 22), "САД ≤ 100 мм рт. ст.": int(systolic <= 100),
                  "Изменённое сознание": int(altered_mentation)}
    score = sum(components.values())
    label = "Повышенный риск неблагоприятного исхода" if score >= 2 else "Менее двух критериев"
    return result(score, "баллов из 3", components, label,
                  "Оцените состояние в клиническом контексте. Низкий балл не исключает сепсис.",
                  "https://doi.org/10.1001/jama.2016.0287", "error" if score >= 2 else "info")


STDIO_TOOLS = (bmi, cockcroft_gault, glasgow)
NETWORK_TOOLS = (*STDIO_TOOLS, mosteller, qsofa)
