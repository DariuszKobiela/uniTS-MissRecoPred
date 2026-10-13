"""Explanation of SD2 image-representation invertibility for Streamlit apps."""

import pandas as pd
import streamlit as st


def render_inpainting_transform_invertibility() -> None:
    """Show theoretical and practical reversibility of inpainting encodings."""
    st.warning(
        "Odwracalność reprezentacji ma bezpośredni wpływ na błąd rekonstrukcji: "
        "wynik obejmuje zarówno jakość inpaintingu, jak i stratę encode → decode."
    )
    with st.expander(
        "↩️ Odwracalność transformacji obrazowych GAF / MTF / RP / SPEC",
        expanded=False,
    ):
        st.markdown(
            """
            Poniższe rozróżnienie dotyczy zarówno bazowych, jak i dostrojonych
            wariantów Stable Diffusion 2 — zmiana wag modelu nie zmienia własności
            samej reprezentacji. **Odwracalność matematyczna** zakłada niezmieniony
            obraz i dostęp do wymaganych metadanych. **Odwrót w pipeline** uwzględnia
            resampling, obraz 8-bitowy oraz piksele zmienione przez inpainting.
            """
        )
        rows = [
            {
                "Transformacja": "GAF (Gramian Angular Field)",
                "Odwracalność matematyczna": "Tak, warunkowo",
                "Odwrót w pipeline": "Przybliżony",
                "Dlaczego": (
                    "Przekątną GASF można analitycznie odwrócić dla wartości "
                    "skalowanych do [0,1]. Resampling i kwantyzacja 8-bitowa "
                    "wprowadzają jednak stratę."
                ),
            },
            {
                "Transformacja": "SPEC (spektrogram STFT)",
                "Odwracalność matematyczna": "Tak, gdy zachowamy fazę",
                "Odwrót w pipeline": "Przybliżony",
                "Dlaczego": (
                    "iSTFT jest poprawną odwrotnością przy zachowanej fazie. "
                    "Faza pochodzi jednak z liniowo uzupełnionego sygnału, a "
                    "magnitude jest skalowane, kwantyzowane i modyfikowane."
                ),
            },
            {
                "Transformacja": "RP (ciągły Recurrence Plot)",
                "Odwracalność matematyczna": "Nie w pełni jednoznaczna",
                "Odwrót w pipeline": "Przybliżony",
                "Dlaczego": (
                    "Macierz odległości zachowuje geometrię 1D, ale tylko z "
                    "dokładnością do przesunięcia i odbicia. Orientację i skalę "
                    "dobieramy z obserwowanego/liniowo uzupełnionego sygnału."
                ),
            },
            {
                "Transformacja": "MTF (Markov Transition Field)",
                "Odwracalność matematyczna": "Nie — transformacja stratna",
                "Odwrót w pipeline": "Przybliżony",
                "Dlaczego": (
                    "Kwantylowa dyskretyzacja jest wiele-do-jednego. Dekoder "
                    "wybiera najbardziej pasujący stan i zwraca środek koszyka, "
                    "więc oryginalnych wartości nie da się odzyskać dokładnie."
                ),
            },
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.info(
            "W praktycznym pipeline żadna z czterech reprezentacji nie gwarantuje "
            "bitowo dokładnego powrotu. GAF i SPEC mają warunkowo poprawną "
            "odwrotność matematyczną; RP zachowuje sygnał z niejednoznacznością, "
            "a MTF jest z definicji stratny."
        )
