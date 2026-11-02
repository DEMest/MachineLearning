# MachineLearning

## Задание 2. LDA

[![Открыть в Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DEMest/MachineLearning/blob/main/Task2_LDA_Colab.ipynb)

**[Ноутбук с кодом, таблицами, рисунками и выводами по пунктам [1]–[9]](Task2_LDA_Colab.ipynb)** открывается в Google Colab прямо из GitHub и выполняется целиком через «Среда выполнения → Выполнить все» (около 5–10 минут).

Датасет df1 генерируется в ноутбуке через `make_blobs`: три класса по 1000 объектов и четыре непрерывных признака. Из него строятся df2 … df10k, в которых объекты класса 0 повторены 2 … 10 000 раз. Дальше на паре классов 0–1 и паре признаков x1–x4:

1. LDA и линейный SVM для каждого датасета: решающая функция, зоны классов и объекты.
2. Центр отрезка между центрами масс, общий центр масс и точка пересечения границы LDA с отрезком в зависимости от числа повторов.
3. ROC, PR и PR-gain с AUROC, AUPRC и AUPRGC для двух целевых классов.
4. Кросс-валидация на df10k (3–100 фолдов) с CI95 для AUROC и AUPRC.

## Задание 1. Palmer Penguins

[![Открыть в Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DEMest/MachineLearning/blob/main/MachineLearning_Colab.ipynb)

**[Ноутбук с кодом, таблицами и рисунками по пунктам [1]–[8]](MachineLearning_Colab.ipynb)** можно открыть в Google Colab прямо из GitHub и выполнить целиком через «Среда выполнения → Выполнить все».

Решение выполнено на [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/). В исходной таблице 344 наблюдения трёх видов пингвинов. Для анализа выбраны четыре количественных измерения: длина и глубина клюва, длина ласта и масса тела. Исходный CSV сохранён в [`data/penguins.csv`](data/penguins.csv); источник файла — [официальный репозиторий palmerpenguins](https://github.com/allisonhorst/palmerpenguins/blob/main/inst/extdata/penguins.csv).

**[Открыть отчёт с таблицами, графиками и пояснениями по пунктам [2]–[8]](REPORT.md).**

### Как устроено решение

1. Пункт [1]: выбор Palmer Penguins и четырёх измерений.
2. Пункт [2]: размерность, классы, пропуски, описательная статистика, [отфильтрованный CSV](data/penguins_filtered.csv).
3. Пункт [3]: матрица диаграмм рассеяния и гистограмм с 25 интервалами.
4. Пункт [4]: коэффициенты корреляции Пирсона по всей выборке и отдельно по видам.
5. Пункт [5]: LDA для Adélie и Chinstrap с четырьмя признаками и отдельные решающие области на каждой паре признаков.
6. Пункт [6]: граница LDA и прямая линейной регрессии на осях двух измерений клюва.
7. Пункт [7]: области решений LDA, линейного SVM, логистической регрессии и наивного Байеса по глубине клюва и длине ласта.
8. Пункт [8]: матрицы ошибок, sensitivity, specificity, precision, recall, ROC и AUC на контрольной выборке.

Все числа и иллюстрации создаёт [`src/run_analysis.py`](src/run_analysis.py). Таблицы сохраняются в `tables/`, рисунки — в `figures/`. Методика, формулы и ограничения оценки описаны рядом с результатами в отчёте.

### Воспроизведение

Нужен Python 3.10 или новее. Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/run_analysis.py
```

Повторный запуск пересоздаст отфильтрованный CSV, таблицы, рисунки и `REPORT.md`.

Данные Palmer Penguins собраны Kristen Gorman и Palmer Station LTER; [условия использования — CC0](https://github.com/allisonhorst/palmerpenguins#license).
