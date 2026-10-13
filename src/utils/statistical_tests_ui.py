"""Shared Streamlit explanations for statistical-test dashboards."""
import numpy as np
import pandas as pd
import streamlit as st

def render_normality_and_test_selection(comparisons, alpha=0.05):
    st.subheader("Test normalności Shapiro–Wilka i wybór testu parami")
    st.markdown(f"""
    Przeprowadziliśmy test Shapiro–Wilka dla **różnic sparowanych wyników każdej
    pary modeli**. Oto wyniki tego testu (α = {alpha:.2f}). Wartość p ≥ α
    oznacza, że nie ma podstaw do odrzucenia założenia o normalności różnic
    (nie jest to dowód idealnej normalności). Dla takich par stosujemy
    parametryczny test t dla prób zależnych. Gdy p < α albo normalności nie
    można wiarygodnie ocenić, stosujemy nieparametryczny test Wilcoxona.
    """)
    if comparisons.empty:
        st.info("Brak porównań parami, dla których można podsumować normalność.")
        return
    p = pd.to_numeric(comparisons["shapiro_p"], errors="coerce")
    cols = st.columns(3)
    cols[0].metric("Normalności nie odrzucono", int(p.ge(alpha).sum()))
    cols[1].metric("Normalność odrzucono", int(p.lt(alpha).sum()))
    cols[2].metric("Nie można ocenić", int(p.isna().sum()))
    sampled_count = (
        int(comparisons["shapiro_sampled"].fillna(False).astype(bool).sum())
        if "shapiro_sampled" in comparisons.columns else 0
    )
    if sampled_count:
        st.info(
            f"Dla {sampled_count} porównań z n > 5000 test Shapiro–Wilka "
            "policzono na deterministycznej losowej próbce 5000 różnic. "
            "Pełne dane nadal są używane w następującym teście t lub Wilcoxona."
        )
    labels = {
        "paired_t": "Test t dla prób zależnych (parametryczny)",
        "wilcoxon": "Test Wilcoxona (nieparametryczny)",
        "insufficient_pairs": "Za mało sparowanych jednostek",
    }
    rows = []
    for name, label in labels.items():
        selected = comparisons.loc[comparisons["test"] == name]
        units = pd.to_numeric(selected["n_units"], errors="coerce").dropna()
        rows.append({
            "Wybrany test": label,
            "Liczba porównań par modeli": len(selected),
            "Łączna liczba jednostek w porównaniach": int(units.sum()),
            "Zakres n na porównanie":
                f"{int(units.min())}–{int(units.max())}" if len(units) else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Łączna liczba jednostek jest sumą n ze wszystkich porównań; ta sama "
        "jednostka może należeć do kilku par. `shapiro_n` podaje liczebność użytą "
        "w teście normalności. Brak wyniku występuje dla mniej niż 3 jednostek "
        "albo stałych różnic."
    )

def render_friedman_explanation(omnibus):
    st.markdown("""
    **Po co test Friedmana?** Jest to nieparametryczny test zbiorczy (omnibus),
    który jednocześnie porównuje co najmniej trzy modele na tych samych,
    kompletnych blokach danych. Hipoteza zerowa mówi, że modele mają takie same
    pozycje/rangi. Test odpowiada na pytanie „czy gdziekolwiek między modelami
    istnieje różnica?”. Nie wskazuje jednak, **które** modele się różnią — do
    tego służą testy parami poniżej wraz z korektą Holma.
    """)
    p = omnibus.get("p_value", np.nan)
    if np.isfinite(p):
        conclusion = (
            "odrzucamy hipotezę o jednakowych rangach modeli"
            if p < 0.05 else
            "nie mamy podstaw do odrzucenia hipotezy o jednakowych rangach modeli"
        )
        st.caption(f"Dla α = 0,05: {conclusion}.")

def render_holm_explanation(n_comparisons):
    st.markdown(f"""
    **Korekta Holma.** W tym widoku wykonujemy {n_comparisons} porównań parami.
    Użycie progu 0,05 osobno dla każdego zwiększałoby ryzyko co najmniej jednego
    fałszywie dodatniego wyniku. Holm porządkuje surowe wartości p od najmniejszej
    i sprawdza je kolejno względem progów α/m, α/(m−1), … . Ta procedura
    step-down kontroluje rodzinny poziom błędu (FWER) i jest mniej konserwatywna
    niż zwykła korekta Bonferroniego.

    Korektę liczymy **wspólnie dla wszystkich {n_comparisons} porównań par
    modeli** dla aktualnych filtrów, metryki i jednostki wnioskowania. Macierz
    istotności korzysta z wartości `p_holm`, nie z surowych wartości p.
    """)
