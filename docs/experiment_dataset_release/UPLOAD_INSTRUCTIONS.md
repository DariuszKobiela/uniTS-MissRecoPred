# Publikacja korekty licencji

Ta paczka zawiera wyłącznie Boiler (CC BY 4.0) i Vibration (CC0 1.0).
Pump oraz jego rekordy metryk są wykluczone; README zawiera link do źródła.

Najpierw wykonaj pełną kontrolę:

```bash
python validate_package.py --deep
```

Następnie opublikuj:

```bash
bash upload.sh
```

Wymagane jest wcześniejsze logowanie `hf auth login` i środowisko projektu.
Skrypt sprawdza sumy kontrolne, zakres zbiorów i metadane, a potem jednym commitem
zastępuje wydanie oraz usuwa mieszane archiwa historyczne i wycofane pliki wydania.
Nie używaj zwykłego uploadu dodającego pliki: pozostawiłby publiczne dane Pump.
Publikator przerywa pracę, jeśli wykryje nieoczekiwane pliki poza zakresem wydania
lub zmianę repozytorium w trakcie publikacji.

Raporty walidacji pozostają poza wysyłanym katalogiem `repo/`.
Wcześniejsze rewizje Git mogą nadal zawierać wycofane pliki. Ten skrypt usuwa je
z bieżącej rewizji; nie przepisuje historii repozytorium.
