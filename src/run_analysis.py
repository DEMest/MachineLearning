"""Reproduce assignment 1, items 2–8, using Palmer Penguins.

Run from the repository root with: python src/run_analysis.py
All output paths are resolved relative to this file, so any working directory works.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CLASS_ORDER = ["Adelie", "Chinstrap", "Gentoo"]
BINARY_CLASSES = ["Adelie", "Chinstrap"]
POSITIVE_CLASS = "Chinstrap"
FEATURE_LABELS = {
    "bill_length_mm": "Длина клюва, мм",
    "bill_depth_mm": "Глубина клюва, мм",
    "flipper_length_mm": "Длина ласта, мм",
    "body_mass_g": "Масса тела, г",
}
COLORS = {"Adelie": "#155e75", "Chinstrap": "#d97706", "Gentoo": "#7c3aed"}
MARKERS = {"Adelie": "o", "Chinstrap": "^", "Gentoo": "s"}
RANDOM_STATE = 42


def save_table(frame: pd.DataFrame, name: str, *, index: bool = False) -> None:
    frame.to_csv(TABLES / name, index=index, float_format="%.6f")


def md(frame: pd.DataFrame, digits: int = 3) -> str:
    return frame.to_markdown(index=False, floatfmt=f".{digits}f")


def save_plot(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=190, bbox_inches="tight")
    plt.close(fig)


def model_score(model, points: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Continuous score; zero (or 0.5 for Naive Bayes) is the class boundary."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(points)).ravel()
    return np.asarray(model.predict_proba(points))[:, 1] - 0.5


def draw_regions(ax, model, data: pd.DataFrame, x: str, y: str, title: str) -> None:
    xlim = data[x].min(), data[x].max()
    ylim = data[y].min(), data[y].max()
    xpad = (xlim[1] - xlim[0]) * 0.07
    ypad = (ylim[1] - ylim[0]) * 0.07
    grid_x, grid_y = np.meshgrid(
        np.linspace(xlim[0] - xpad, xlim[1] + xpad, 250),
        np.linspace(ylim[0] - ypad, ylim[1] + ypad, 250),
    )
    grid = pd.DataFrame({x: grid_x.ravel(), y: grid_y.ravel()})
    score = model_score(model, grid).reshape(grid_x.shape)
    ax.contourf(
        grid_x,
        grid_y,
        score > 0,
        levels=[-0.5, 0.5, 1.5],
        colors=["#dceef3", "#fcebd7"],
        alpha=0.8,
    )
    ax.contour(grid_x, grid_y, score, levels=[0], colors="#1f2937", linewidths=1.7)
    for species in BINARY_CLASSES:
        part = data.loc[data["species"] == species]
        ax.scatter(
            part[x],
            part[y],
            s=38,
            c=COLORS[species],
            marker=MARKERS[species],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.9,
            label=species,
        )
    ax.set_xlabel(FEATURE_LABELS[x])
    ax.set_ylabel(FEATURE_LABELS[y])
    ax.set_title(title)


def raw_lda_coefficients(model, features: list[str]) -> dict[str, float]:
    """Convert LDA coefficients from standardized features to original units."""
    scaler = model.named_steps["standardscaler"]
    lda = model.named_steps["lineardiscriminantanalysis"]
    weights = lda.coef_[0] / scaler.scale_
    intercept = lda.intercept_[0] - np.dot(weights, scaler.mean_)
    return {"intercept": float(intercept), **dict(zip(features, weights))}


def main() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams["axes.unicode_minus"] = False
    DATA.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    raw = pd.read_csv(DATA / "penguins.csv")
    assert len(raw) == 344 and set(raw["species"]) == set(CLASS_ORDER)
    assert len(raw.columns) == 8 and all(c in raw for c in FEATURES)
    filtered = raw.dropna(subset=FEATURES).loc[:, ["species", *FEATURES]].copy()
    filtered.to_csv(DATA / "penguins_filtered.csv", index=False)

    # [2] Original dataset, missing values, and four selected measurements.
    predictors = [column for column in raw if column != "species"]
    any_missing = int(raw[predictors].isna().any(axis=1).sum())
    numeric_missing = int(raw[FEATURES].isna().any(axis=1).sum())
    overview = pd.DataFrame(
        [
            ("Объектов до фильтрации", len(raw)),
            ("Столбцов исходной таблицы", len(raw.columns)),
            ("Признаков без целевой переменной", len(predictors)),
            ("Выбранных количественных признаков", len(FEATURES)),
            ("Целевых классов", raw["species"].nunique()),
            ("Объектов с пропуском в любом признаке", any_missing),
            ("Доля объектов с пропуском в любом признаке, %", 100 * any_missing / len(raw)),
            ("Объектов с пропуском в выбранных 4 признаках", numeric_missing),
            ("Доля объектов с пропуском в выбранных 4 признаках, %", 100 * numeric_missing / len(raw)),
            ("Объектов после фильтрации", len(filtered)),
            ("Островов", raw["island"].nunique()),
            ("Начальный год", int(raw["year"].min())),
            ("Конечный год", int(raw["year"].max())),
        ],
        columns=["Показатель", "Значение"],
    )
    save_table(overview, "02_overview.csv")
    class_counts = pd.DataFrame(
        {
            "Класс": CLASS_ORDER,
            "До фильтрации": [int((raw.species == c).sum()) for c in CLASS_ORDER],
            "После фильтрации": [int((filtered.species == c).sum()) for c in CLASS_ORDER],
        }
    )
    save_table(class_counts, "02_class_counts.csv")
    missingness = pd.DataFrame(
        {
            "Столбец": raw.columns,
            "Пропусков": raw.isna().sum().values,
            "Доля, %": (raw.isna().mean() * 100).values,
        }
    )
    save_table(missingness, "02_missingness.csv")
    numeric_summary = filtered[FEATURES].agg(["count", "mean", "std", "median", "min", "max"]).T
    numeric_summary.index.name = "Признак"
    save_table(numeric_summary.reset_index(), "02_numeric_summary.csv")

    # [3] Every ordered feature pair and a histogram on each diagonal.
    grid = sns.pairplot(
        filtered,
        vars=FEATURES,
        hue="species",
        hue_order=CLASS_ORDER,
        palette=COLORS,
        # pairplot associates markers with first appearance in the source table.
        markers=[MARKERS[c] for c in filtered["species"].drop_duplicates()],
        diag_kind="hist",
        diag_kws={"bins": 25, "alpha": 0.45},
        plot_kws={"alpha": 0.8, "s": 38, "edgecolor": "white", "linewidth": 0.25},
        height=2.55,
    )
    for row in grid.axes:
        for ax in row:
            if ax is not None:
                ax.set_xlabel(FEATURE_LABELS.get(ax.get_xlabel(), ax.get_xlabel()))
                ax.set_ylabel(FEATURE_LABELS.get(ax.get_ylabel(), ax.get_ylabel()))
    grid.fig.suptitle("Пункт 3. Все пары признаков и гистограммы (25 интервалов)", y=1.025)
    save_plot(grid.fig, "03_pairplot.png")

    # [4] Pearson correlation on the same complete observations overall and by class.
    correlations: dict[str, pd.DataFrame] = {"all": filtered[FEATURES].corr(method="pearson")}
    for species in CLASS_ORDER:
        correlations[species] = filtered.loc[filtered.species == species, FEATURES].corr(method="pearson")
    for name, corr in correlations.items():
        save_table(corr, f"04_pearson_{name.lower()}.csv", index=True)

    # One fixed stratified split for the binary work in items [5]–[8].
    binary = filtered.loc[filtered.species.isin(BINARY_CLASSES)].reset_index(drop=True)
    labels = (binary.species == POSITIVE_CLASS).astype(int)
    train_ids, test_ids = train_test_split(
        np.arange(len(binary)), test_size=0.25, stratify=labels, random_state=RANDOM_STATE
    )
    binary_train = binary.iloc[train_ids]
    binary_test = binary.iloc[test_ids]
    y_train = labels.iloc[train_ids]
    y_test = labels.iloc[test_ids]
    split_table = pd.DataFrame(
        {
            "Класс": BINARY_CLASSES,
            "Обучение": [int((binary_train.species == c).sum()) for c in BINARY_CLASSES],
            "Контроль": [int((binary_test.species == c).sum()) for c in BINARY_CLASSES],
        }
    )
    save_table(split_table, "05_08_split.csv")

    # [5] Full four-feature LDA plus six separate two-feature LDA maps.
    full_lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    full_lda.fit(binary_train[FEATURES], y_train)
    full_accuracy = accuracy_score(y_test, full_lda.predict(binary_test[FEATURES]))
    coefficient_rows = [
        {
            "Модель": "Все 4 признака",
            **raw_lda_coefficients(full_lda, FEATURES),
            "Accuracy на контроле": full_accuracy,
        }
    ]
    pair_models = {}
    pairs = list(combinations(FEATURES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (x, y) in zip(axes.flat, pairs):
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        model.fit(binary_train[[x, y]], y_train)
        pair_models[(x, y)] = model
        coefficient_rows.append(
            {
                "Модель": f"{x} + {y}",
                **raw_lda_coefficients(model, [x, y]),
                "Accuracy на контроле": accuracy_score(y_test, model.predict(binary_test[[x, y]])),
            }
        )
        draw_regions(ax, model, binary, x, y, f"{FEATURE_LABELS[x]} × {FEATURE_LABELS[y]}")
    axes[0, 0].legend(title="Класс", loc="best")
    fig.suptitle("Пункт 5. LDA по каждой паре количественных признаков", fontsize=16)
    fig.tight_layout()
    save_plot(fig, "05_lda_all_pairs.png")
    coefficient_table = pd.DataFrame(coefficient_rows)
    save_table(coefficient_table, "05_lda_coefficients.csv")

    # [6] The same two-feature LDA boundary and an ordinary least-squares line.
    regression_pair = ("bill_length_mm", "bill_depth_mm")
    x, y = regression_pair
    regression = LinearRegression().fit(binary_train[[x]], binary_train[y])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    draw_regions(ax, pair_models[regression_pair], binary, x, y, "Пункт 6. Граница LDA и линейная регрессия")
    x_line = np.linspace(binary[x].min(), binary[x].max(), 200)
    y_line = regression.predict(pd.DataFrame({x: x_line}))
    ax.plot(x_line, y_line, color="#9b1c31", linestyle="--", linewidth=2.4, label="Линейная регрессия")
    ax.legend(loc="best")
    save_plot(fig, "06_lda_and_regression.png")
    regression_info = pd.DataFrame(
        {
            "Зависимая переменная": [y],
            "Независимая переменная": [x],
            "Свободный член": [regression.intercept_],
            "Наклон": [regression.coef_[0]],
            "R² на обучении": [regression.score(binary_train[[x]], binary_train[y])],
        }
    )
    save_table(regression_info, "06_regression.csv")

    # [7] A more overlapping pair makes the differences between methods visible.
    comparison_pair = ("bill_depth_mm", "flipper_length_mm")
    x_comparison, y_comparison = comparison_pair
    models = {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
        "Логистическая регрессия": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Наивный Байес": make_pipeline(StandardScaler(), GaussianNB()),
    }
    file_stems = {
        "LDA": "lda",
        "SVM (RBF)": "svm_rbf",
        "Логистическая регрессия": "logistic_regression",
        "Наивный Байес": "gaussian_naive_bayes",
    }
    for name, model in models.items():
        model.fit(binary_train[list(comparison_pair)], y_train)
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_regions(ax, model, binary, x_comparison, y_comparison, f"Пункт 7. {name}")
        ax.legend(loc="best")
        save_plot(fig, f"07_{file_stems[name]}.png")

    # [8] Classification metrics, confusion matrices, and ROC on held-out data.
    metric_rows = []
    confusion_rows = []
    roc_rows = []
    fig_cm, axes_cm = plt.subplots(2, 2, figsize=(10, 9))
    fig_roc, ax_roc = plt.subplots(figsize=(8, 7))
    for ax, (name, model) in zip(axes_cm.flat, models.items()):
        y_pred = model.predict(binary_test[list(comparison_pair)])
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        score = model_score(model, binary_test[list(comparison_pair)])
        fpr, tpr, thresholds = roc_curve(y_test, score)
        area = auc(fpr, tpr)
        metric_rows.append(
            {
                "Метод": name,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Precision": precision,
                "Recall": sensitivity,
                "AUC": area,
                "Accuracy": accuracy_score(y_test, y_pred),
            }
        )
        for true_label, predicted_label, count in [
            ("Adelie", "Adelie", tn),
            ("Adelie", "Chinstrap", fp),
            ("Chinstrap", "Adelie", fn),
            ("Chinstrap", "Chinstrap", tp),
        ]:
            confusion_rows.append(
                {"Метод": name, "Истинный класс": true_label, "Прогноз": predicted_label, "Количество": count}
            )
        for a, b, threshold in zip(fpr, tpr, thresholds):
            roc_rows.append({"Метод": name, "FPR": a, "TPR": b, "Порог": threshold})
        ConfusionMatrixDisplay(
            confusion_matrix=np.array([[tn, fp], [fn, tp]]),
            display_labels=BINARY_CLASSES,
        ).plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(name)
        ax.set_xlabel("Прогноз")
        ax.set_ylabel("Истинный класс")
        ax.grid(False)
        ax_roc.plot(fpr, tpr, linewidth=2.2, label=f"{name}: AUC = {area:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, label="Случайный классификатор")
    ax_roc.set_xlabel("FPR = 1 − specificity")
    ax_roc.set_ylabel("TPR = sensitivity")
    ax_roc.set_title("Пункт 8. ROC на контрольной выборке")
    ax_roc.legend(loc="lower right")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    fig_cm.suptitle("Пункт 8. Матрицы ошибок на контрольной выборке", fontsize=16)
    fig_cm.tight_layout()
    save_plot(fig_cm, "08_confusion_matrices.png")
    save_plot(fig_roc, "08_roc.png")
    metrics = pd.DataFrame(metric_rows)
    save_table(metrics, "08_metrics.csv")
    save_table(pd.DataFrame(confusion_rows), "08_confusion_matrices.csv")
    save_table(pd.DataFrame(roc_rows), "08_roc_points.csv")

    # Human-readable results. The code above remains the source of all numbers.
    report = [
        "# Результаты задания 1",
        "",
        "Источник: [palmerpenguins::penguins](https://github.com/allisonhorst/palmerpenguins/blob/main/inst/extdata/penguins.csv).",
        "Цель — `species`; четыре признака указаны в именах столбцов и измеряются в мм или г.",
        "",
        "## [2] Статистика и фильтрация",
        "",
        overview.assign(
            Значение=[
                f"{value:.2f}" if "%" in label else str(int(value))
                for label, value in overview.itertuples(index=False, name=None)
            ]
        ).to_markdown(index=False, disable_numparse=True),
        "",
        "### Число объектов по классам",
        "",
        md(class_counts, 0),
        "",
        "### Пропуски в исходных столбцах",
        "",
        md(missingness, 2),
        "",
        "Все пропуски четырёх измерений приходятся на два объекта. Они удалены, "
        "но объекты с неизвестным полом и заполненными измерениями сохранены: пол не участвует в анализе.",
        "",
        "### Описательная статистика отфильтрованных измерений",
        "",
        md(numeric_summary.reset_index(), 2),
        "",
        "Отфильтрованные данные: [data/penguins_filtered.csv](data/penguins_filtered.csv).",
        "",
        "## [3] Все пары признаков",
        "",
        "На внедиагональных графиках классы различаются цветом и формой. "
        "На диагонали показаны гистограммы с 25 интервалами.",
        "",
        "![Матрица пар признаков](figures/03_pairplot.png)",
        "",
        "## [4] Корреляции Пирсона",
        "",
        "Корреляции вычислены для всех 342 объектов с заполненными четырьмя измерениями, "
        "а затем отдельно внутри каждого вида. Коэффициент описывает линейную связь, "
        "а не причинную зависимость.",
    ]
    for name, label in [
        ("all", "Вся выборка"),
        ("Adelie", "Adélie"),
        ("Chinstrap", "Chinstrap"),
        ("Gentoo", "Gentoo"),
    ]:
        corr = correlations[name].round(3).reset_index().rename(columns={"index": "Признак"})
        report += ["", f"### {label}", "", md(corr, 3)]
    report += [
        "",
        "**Как читать корреляции.** В общей выборке связь глубины клюва и массы тела "
        "отрицательна (r = -0.472), но внутри каждого вида положительна "
        "(Adélie 0.576; Chinstrap 0.604; Gentoo 0.719). Смешение видов меняет "
        "наблюдаемую связь, поэтому одной общей корреляции здесь недостаточно.",
    ]
    report += [
        "",
        "## [5] LDA на всех четырёх признаках и на каждой паре",
        "",
        f"Классы: Adélie (0) и Chinstrap (1). Положительный класс: {POSITIVE_CLASS}. "
        "Один и тот же стратифицированный разрез используется дальше во всех моделях:",
        "",
        md(split_table, 0),
        "",
        "Признаки стандартизированы по обучающей выборке. Сначала обучена LDA со всеми "
        "четырьмя измерениями; затем для понятных двухмерных зон — отдельная LDA на каждой "
        "из шести пар. Коэффициенты ниже пересчитаны в исходные единицы. "
        "Решающая функция: f(x) = свободный член + сумма(коэффициент × признак); "
        "при f(x) > 0 выбирается Chinstrap.",
        "",
        md(
            coefficient_table.assign(
                **{
                    column: coefficient_table[column].map(
                        lambda value: "" if pd.isna(value) else f"{value:.6g}"
                    )
                    for column in coefficient_table.columns
                    if column != "Модель"
                }
            ),
            3,
        ),
        "",
        "![Шесть решающих областей LDA](figures/05_lda_all_pairs.png)",
        "",
        "## [6] LDA и линейная регрессия",
        "",
        f"Для длины клюва x и глубины клюва y регрессия на обучающей части: "
        f"y = {regression.intercept_:.3f} + ({regression.coef_[0]:.3f}) × x. "
        "Регрессия прогнозирует измерение, а граница LDA разделяет виды.",
        "",
        "![LDA и регрессия](figures/06_lda_and_regression.png)",
        "",
        "## [7] Четыре метода на двух признаках",
        "",
        "Для всех методов используются глубина клюва и длина ласта, одинаковые обучение и контроль. "
        "Эта пара сильнее перекрывает классы, поэтому сравнение методов на ней информативнее. "
        "SVM использует радиальное ядро (RBF), C=1; наивный Байес — гауссовскую модель. "
        "Масштабирование обучено только на обучающей части.",
    ]
    for name in models:
        report += ["", f"### {name}", "", f"![Граница {name}](figures/07_{file_stems[name]}.png)"]
    report += [
        "",
        "## [8] Контрольная оценка",
        "",
        "Матрица ошибок: строки — истинный вид, столбцы — прогноз. "
        "TP и FN относятся к Chinstrap, TN и FP — к Adélie. "
        "Sensitivity = Recall = TP/(TP+FN); Specificity = TN/(TN+FP); "
        "Precision = TP/(TP+FP). AUC — площадь под ROC-кривой.",
        "",
        md(metrics, 3),
        "",
        "На этой паре признаков классы заметно перекрываются. У логистической регрессии "
        "наибольшая accuracy (0.745), но обнаружено только 5 из 17 Chinstrap: "
        "sensitivity/recall = 0.294. Высокая specificity здесь не означает, "
        "что метод хорошо находит положительный класс. Эти значения относятся "
        "только к выбранному разбиению и двум признакам.",
        "",
        "![Матрицы ошибок](figures/08_confusion_matrices.png)",
        "",
        "![ROC-кривые](figures/08_roc.png)",
        "",
        "Контрольная выборка использована один раз для оценки. Это учебный holdout, "
        "поэтому метрики зависят от разбиения и не являются гарантией качества на новых данных.",
        "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Ready: {len(raw)} raw rows, {len(filtered)} filtered rows, {len(binary_test)} test rows")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
