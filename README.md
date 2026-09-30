# MachineLearning — задание 1

Решение выполнено на [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/). В исходной таблице 344 наблюдения трёх видов пингвинов. Для анализа выбраны четыре количественных измерения: длина и глубина клюва, длина ласта и масса тела. Исходный CSV сохранён в [`data/penguins.csv`](data/penguins.csv); источник файла — [официальный репозиторий palmerpenguins](https://github.com/allisonhorst/palmerpenguins/blob/main/inst/extdata/penguins.csv).

**[Открыть отчёт с таблицами, графиками и пояснениями по пунктам [2]–[8]](REPORT.md).**

## Как устроено решение

1. Пункт [1]: выбор Palmer Penguins и четырёх измерений.
2. Пункт [2]: размерность, классы, пропуски, описательная статистика, [отфильтрованный CSV](data/penguins_filtered.csv).
3. Пункт [3]: матрица диаграмм рассеяния и гистограмм с 25 интервалами.
4. Пункт [4]: коэффициенты корреляции Пирсона по всей выборке и отдельно по видам.
5. Пункт [5]: LDA для Adélie и Chinstrap с четырьмя признаками и отдельные решающие области на каждой паре признаков.
6. Пункт [6]: граница LDA и прямая линейной регрессии на осях двух измерений клюва.
7. Пункт [7]: области решений LDA, SVM, логистической регрессии и наивного Байеса по глубине клюва и длине ласта.
8. Пункт [8]: матрицы ошибок, sensitivity, specificity, precision, recall, ROC и AUC на контрольной выборке.

Все числа и иллюстрации создаёт [`src/run_analysis.py`](src/run_analysis.py). Таблицы сохраняются в `tables/`, рисунки — в `figures/`. Методика, формулы и ограничения оценки описаны рядом с результатами в отчёте.

## Воспроизведение

Нужен Python 3.10 или новее. Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/run_analysis.py
```

Повторный запуск пересоздаст отфильтрованный CSV, таблицы, рисунки и `REPORT.md`.

Данные Palmer Penguins собраны Kristen Gorman и Palmer Station LTER; [условия использования — CC0](https://github.com/allisonhorst/palmerpenguins#license).
