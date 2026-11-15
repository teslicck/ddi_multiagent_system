# Оценка поиска (без LLM)

Пар: 40 (с взаимодействием: 30, без: 10).

| Метрика | Значение |
|---|---|
| Чувствительность, DailyMed | 29/30 (97%) |
| Чувствительность, PubMed | 29/30 (97%) |
| **Чувствительность, хотя бы один источник** | **30/30 (100%)** |
| DailyMed: найдено в обеих инструкциях | 23/30 (77%) |
| DailyMed: есть фрагмент из Boxed Warning / Contraindications | 10/30 (33%) |
| DailyMed: найдено только по классу (RxClass / синонимы) | 3/30 (10%) |
| Пары без взаимодействия: источники ничего не вернули | 5/10 (50%) |
| Пары с сетевыми ошибками | 3 |
| Время на пару, медиана | 18 с |

**Пары без взаимодействия, где источники что-то вернули** (финальный ответ решает LLM-агент, см. режим full): metformin+amoxicillin (DailyMed 0, PubMed 1), acetaminophen+amoxicillin (DailyMed 0, PubMed 2), lisinopril+acetaminophen (DailyMed 0, PubMed 4), atorvastatin+acetaminophen (DailyMed 0, PubMed 2), famotidine+acetaminophen (DailyMed 0, PubMed 1)

**Найдено только по классу:** sertraline+tramadol (opioid, ssri), oxycodone+diazepam (benzodiazepine, opioid), meloxicam+lisinopril (ace inhibitor, nsaid)
