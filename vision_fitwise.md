# Projektvision: `fitwise`

*Name auf PyPI frei (geprüft 2026-10-22) · Lizenz: BSD-3-Clause · Stand: 2026-10-22*

> **Ein Aufruf. Ein Fit. Ein Report, dem man trauen kann.**

---

## 1. Problem

Kurvenfits sind die häufigste quantitative Auswertung in Labor und Entwicklung. In Python sind sie trotzdem noch umständlich:

- `scipy.optimize.curve_fit` liefert nur Parameter und Kovarianzmatrix. Startwerte, Gütemaße, Konfidenzbänder und Residuenanalyse sind Handarbeit.
- `lmfit` ist mächtig, aber verbos. Automatische Startwerte (`guess()`) gibt es nur für die eingebauten Modelle, nicht für eigene Funktionen. Für ein eigenes Modell muss der Nutzer die Skala jedes Parameters selbst kennen.
- Unsicherheiten in x **und** y erfordern eine völlig andere API (`scipy.odr`, seit SciPy 1.17 deprecated, Nachfolger `odrpack`). In lmfit gibt es dafür nur ein Beispiel-Notebook.
- Viele Nutzer fitten Modelle, die linear in den Parametern sind, mit einem nichtlinearen Solver und scheitern an schlechten Startwerten, obwohl das Problem direkt und eindeutig lösbar wäre.
- Die Semantik der Unsicherheiten ist eine stille Falle: Ob die Kovarianz mit χ²/ndf skaliert wird (`absolute_sigma`), entscheidet über die angegebenen Fehler und ist für die meisten Nutzer unsichtbar.
- Den Report, den Ingenieure aus Origin, Excel oder MATLAB kennen (Parameter ± σ, χ²/ndf, R², Residuenplot, Konfidenzband), muss man jedes Mal neu zusammenbauen.

Das Ergebnis: Dutzende Zeilen Boilerplate pro Fit, eine inkonsistente Fehlerbehandlung und die stille Rückkehr zu Origin, um „mal eben einen Gauß durchzulegen“.

## 2. Vision

`fitwise` ist die **Ergonomie-Schicht** über `lmfit` und `scipy`. Es erfindet keinen Solver. Es macht die vorhandenen so zugänglich, dass ein Fit in Python schneller und reproduzierbarer ist als in jedem GUI-Tool.

```python
import fitwise as fw

result = fw.fit(fw.models.gaussian, x, y, yerr=dy)
print(result)              # Report wie in Origin
result.plot()              # Daten, Fit, Konfidenzband, Residuen
result.params["sigma"]     # Parameter mit .value, .stderr, .unit
```

Der Nutzer denkt in „Modell + Daten“, nicht in „Solver + Startwerte + Kovarianz“.

**Positionierung in einem Satz:** *Was `requests` für `urllib` ist, ist `fitwise` für `curve_fit`/`lmfit`.*

## 3. Zielgruppe

- Ingenieur:innen und Naturwissenschaftler:innen, die Messdaten auswerten und dafür bisher Origin, Excel oder MATLAB nutzen
- Studierende in Praktika (Physik, Chemie, E-Technik, Maschinenbau)
- Entwickler:innen von Messsoftware, die Fits in Pipelines einbauen und einen stabilen, maschinenlesbaren Report brauchen

**Nicht** Zielgruppe: Statistiker:innen, die Bayes-Inferenz oder MCMC brauchen, sowie ML-Praktiker:innen. Für sie gibt es `scikit-learn`, `emcee` und `PyMC`.

## 4. Kernfunktionen

### 4.1 Ein Einstiegspunkt

```python
fw.fit(model, x, y, *,
       yerr=None, xerr=None,
       p0=None, bounds=None, fixed=None,
       loss="linear", method="auto")
```

| Argument | Typ | Bedeutung |
|---|---|---|
| `model` | Funktion, `fw.Model`, String | `def f(x, a, b)`, eingebautes/komponiertes Modell oder Ausdruck `"a*exp(-x/tau) + c"` |
| `yerr`, `xerr` | Skalar oder Array | **Absolute** 1σ-Unsicherheiten (siehe 4.5) |
| `p0` | `dict` | Startwerte; überschreibt Heuristiken parameterweise |
| `bounds` | `dict[str, (lo, hi)]` | `None` für eine offene Seite |
| `fixed` | `dict[str, float]` | Parameter mit festem Wert |
| `loss` | `"linear"`, `"soft_l1"`, `"huber"`, `"cauchy"` | Robuste Verlustfunktion |
| `method` | `"auto"`, `"linear"`, `"lm"`, `"trf"`, `"odr"`, `"multistart"` | Solver-Wahl; `auto` dokumentiert die Entscheidung im Result |

- Rückgabe ist immer ein `FitResult` mit identischer Struktur, unabhängig vom Solver.
- Startwerte aus der Funktionssignatur werden übernommen: `def f(x, a=1.0, tau=1e-3)` ist eine gültige Art, `p0` anzugeben.
- `x` darf mehrdimensional sein (`shape (n, k)`, z. B. für Flächenfits). Das Modell bekommt dann `x` unverändert übergeben.
- NaN/Inf in den Daten führen zu einem Fehler, nicht zu stillem Wegfiltern. `fw.fit(..., nan_policy="omit")` ist der explizite Ausweg.

Gewichte gibt es bewusst nur über `yerr`, nicht über ein zusätzliches `weights`-Argument. Zwei Wege für dasselbe wären eine Fehlerquelle.

### 4.2 Automatische Solver-Wahl

| Situation | Solver |
|---|---|
| Modell linear in den Parametern, kein `xerr`, `loss="linear"` | Direkte Lösung: `numpy.linalg.lstsq`, mit Bounds `scipy.optimize.lsq_linear`. Keine Startwerte nötig. |
| Nichtlinear, mit oder ohne Bounds | Trust-Region-Reflective (`scipy.optimize.least_squares`) mit normierten Parametern und Neustarts (ADR 5); Levenberg-Marquardt wählbar mit `method="lm"` |
| Nicht konvergiert oder Heuristik unsicher | Multi-Start (Latin-Hypercube innerhalb plausibler Skalen), bester Lauf gewinnt |
| `xerr` angegeben | Orthogonal Distance Regression via `odrpack` (ODRPACK95, unterstützt Bounds) |
| `loss != "linear"` | `scipy.optimize.least_squares` mit der gewählten Loss-Funktion |

**Linearitätserkennung (numerisch, ohne sympy):** Die Jacobi-Matrix bzgl. der Parameter wird an zwei bis drei zufälligen Parameterpunkten ausgewertet. Ist sie (bis auf Toleranz) identisch, ist das Modell linear in den Parametern. Für eingebaute Modelle ist die Linearität deklariert und muss nicht erkannt werden. `sympy` wird nur für String-Modelle optional genutzt, um analytische Ableitungen zu liefern.

### 4.3 Startwerte für eigene Funktionen

Das ist das Alleinstellungsmerkmal und der schwierigste Teil. Die Strategie ist eine Kaskade; die erste Stufe, die gültige Werte liefert, gewinnt:

1. **Explizit:** `p0` oder Default-Werte aus der Funktionssignatur.
2. **Eingebautes Modell:** modellspezifische `guess()`-Heuristik (Peak-Position, FWHM, Abfallzeit …).
3. **Strukturerkennung bei freien Funktionen:** Zuerst wird eine *Rollenanalyse* durchgeführt. An einem Referenzpunkt wird jeder Parameter gestört, und es wird klassifiziert, wie er wirkt:
   - additiver Offset (Ableitung ≈ konstant in x) → aus Median/Rand der Daten
   - Amplitude (Modell skaliert ≈ proportional) → aus Spannweite der Daten
   - x-Verschiebung (Ableitung ∝ dy/dx) → aus Lage des Extremums/Wendepunkts
   - x-Skala (Ableitung ∝ x·dy/dx) → aus FWHM, Abfallzeit, Datenbereich
   - sonstige → Multi-Start
4. **Linearisierung**, wo möglich: Log-Transformation für Exponential- und Potenzgesetz-Strukturen, lineare Teilprobleme direkt lösen (Variable Projection für Parameter, die linear eingehen).
5. **Multi-Start** mit Latin-Hypercube-Sampling als Fallback, begrenzt durch `max_starts` (Standard 50).

Jede Entscheidung ist transparent: `result.initial_guess`, `result.guess_strategy` (pro Parameter) und `result.solver_log` sind einsehbar und erscheinen im Report.

### 4.4 Modellbibliothek

Eingebaute Modelle haben eine physikalische Bedeutung, dokumentierte Parameter (inkl. Rolle und typischer Einheit), eine Startwert-Heuristik, eine deklarierte Linearität und, wo sinnvoll, analytische Ableitungen.

| Gruppe | Modelle |
|---|---|
| Basis | `constant`, `linear`, `polynomial(n)` |
| Peaks | `gaussian`, `lorentzian`, `voigt`, `pseudo_voigt` |
| Zeitverläufe | `exponential`, `exponential(n)` (mehrfach), `damped_sine` |
| Übergänge | `sigmoid`, `boltzmann`, `hill`, `michaelis_menten` |
| Skalen | `powerlaw`, `arrhenius` |
| Stückweise | `linear_breakpoint` |

**Komposition:** `fw.models.gaussian + fw.models.linear`. Bei Namenskollisionen werden Präfixe automatisch vergeben (`g1_amp`, `g2_amp`), alternativ explizit: `fw.models.gaussian.with_prefix("peak_")`. Die Startwerte komponierter Modelle werden sequenziell geschätzt (Untergrund zuerst, dann Peaks auf den Residuen).

### 4.5 Unsicherheiten als erstklassige Bürger

**Festgelegte Semantik (keine stille Entscheidung):**

- Mit `yerr`: `yerr` gilt als absolute 1σ-Unsicherheit. Die Kovarianz wird **nicht** mit χ²/ndf skaliert. Der Report zeigt χ²/ndf als Konsistenzprüfung und warnt bei einer signifikanten Abweichung von 1.
- Ohne `yerr`: Die Unsicherheiten werden aus der Streuung der Residuen geschätzt (Kovarianz skaliert). Der Report sagt das ausdrücklich, und χ²/ndf wird durch den RMSE der Residuen ersetzt.
- Überschreibbar mit `fw.fit(..., scale_covar=True/False)`.

**Funktionen:**

- Parameterunsicherheiten aus der Kovarianz, optional aus Profile-Likelihood (`lmfit.conf_interval`) oder Bootstrap (`uncertainty="bootstrap"`)
- Korrelationsmatrix im Report, starke Korrelationen hervorgehoben
- Konfidenz- **und** Prädiktionsband als Funktionen: `result.confidence_band(x_new, level=0.95)` und `result.prediction_band(...)`
- Optionale Integration: `pint` (Einheiten durchreichen und in den Report übernehmen), `uncertainties` (`result.params.as_ufloat()` für die Fehlerfortpflanzung in Folgerechnungen)

### 4.6 Der Report

`print(result)` liefert:

```
Model:      gaussian(x; amp, cen, sigma) + linear(x; slope, intercept)
Method:     Trust Region Reflective (scipy least_squares) (auto)
Converged:  yes (14 evals)
Data:       128 points, yerr given (absolute)
Start:      amp, cen, sigma: peak heuristic · slope, intercept: edge baseline

Parameter     Value        ± Std.Err     Rel.      Bounds
amp           2.331        0.021        0.9 %     (0, ∞)
cen           0.2104       0.0018       0.9 %     –
sigma         1.512        0.019        1.3 %     (0, ∞)
slope        -0.0031       0.0009      29.0 %     –
intercept     0.104        0.011       10.6 %     –

χ²/ndf = 1.08 (123 dof, p = 0.26)    R² = 0.9987    AIC = -412.3
Correlations > 0.5:  amp–sigma (-0.63)
Warnings:   none
```

- Werte und Unsicherheiten werden nach DIN 1333 / PDG-Konvention gerundet (Unsicherheit auf 2 signifikante Stellen, Wert auf dieselbe Dezimalstelle).
- R² wird angezeigt, weil Nutzer es erwarten. Die Dokumentation erklärt seine Grenzen bei nichtlinearen Modellen.
- Export: `result.to_markdown()`, `result.to_latex()`, `result.to_dict()` (JSON-serialisierbar, mit versioniertem Schema, für Reproduzierbarkeit) und `_repr_html_()` für Jupyter.

### 4.7 Plotting

`result.plot(residuals=True, band="confidence", ax=None)`

Erzeugt Daten mit Fehlerbalken, Fitkurve (fein abgetastet, nicht nur an den Datenpunkten), Band und Residuenpanel (normiert auf `yerr`, wenn gegeben). Die Achsenbeschriftung kommt aus den Einheiten, die Legende enthält optional die Parameter. Gibt die `Figure`/`Axes` zurück, damit Nutzer weiter anpassen können. Ein optionales Stil-Backend (z. B. `plotstyle`) ist über `style=` einhängbar, aber keine Abhängigkeit.

### 4.8 Batch-Fits

`fw.fit_many(model, datasets, shared=None)` für Serien (z. B. 200 Spektren):

- getrennte Fits (Standard), optional parallelisiert
- globale Fits mit gemeinsamen Parametern (`shared=["sigma"]`)
- Startwerte des vorherigen Fits als Startpunkt für den nächsten (`warm_start=True`) für Serien mit langsam variierenden Parametern
- Ergebnis als `FitCollection` mit `.to_dataframe()` (alle Parameter, Unsicherheiten, Gütemaße, Warnungen)

## 5. Nicht-Ziele

- Kein eigener Optimierer. Alle numerischen Kerne kommen aus `scipy`, `lmfit` und `odrpack`.
- Keine Bayes-Inferenz und kein MCMC (Verweis auf `emcee`, `PyMC`). Ein Export zu `emcee` über `result.raw` bzw. lmfit bleibt möglich.
- Keine GUI. Ein Jupyter-Widget ist ein denkbares Add-on, nicht Teil des Kerns.
- Keine Datenimport-Funktionen (das ist ein anderes Projekt).
- Keine impliziten Datenvorverarbeitungen (Glätten, Untergrundabzug, Ausreißerentfernung). Robuste Fits über `loss=` sind der vorgesehene Weg.

## 6. Architektur

```
src/fitwise/
  __init__.py      fit, fit_many, Model, models
  api.py           fit(), fit_many()                 – öffentliche Oberfläche
  _validate.py     Eingabeprüfung, NaN-Policy, Broadcasting von yerr/xerr
  model.py         Model-Klasse, Signatur-Introspektion, Komposition
  expr.py          String-Modelle (asteval aus lmfit; sympy optional)
  models/          eingebaute Modelle, je Datei: Funktion + Metadaten + guess()
  guess/           Startwert-Kaskade: roles.py, linearize.py, multistart.py
  linear.py        Linearitätserkennung + direkte Lösung
  backends/        base.py (Protokoll), least_squares.py, später linear.py, odr.py
  result.py        FitResult, Parameter, Gütemaße, Bänder
  report.py        Text, Markdown, LaTeX, HTML; Rundung
  warnings.py      FitWarning-Hierarchie
  plotting.py      matplotlib-Ausgabe (lazy import)
  units.py         pint/uncertainties-Brücke (optional)
  serialize.py     to_dict()/from_dict(), Schema-Version
```

**Backend-Protokoll:** Jedes Backend nimmt ein normalisiertes `FitProblem` (Modell, Daten, Startwerte, Bounds, Fixierungen) entgegen und liefert ein normalisiertes `BackendResult` (Werte, Kovarianz, Residuen, Konvergenzinfo, Log). `FitResult` wird ausschließlich daraus gebaut. So ist die Gleichheit der Rückgabetypen strukturell garantiert und nicht nur Konvention.

**Abhängigkeiten:**

- Pflicht: `numpy>=1.26`, `scipy>=1.12`, `lmfit>=1.3`
- Extras: `fitwise[plot]` (matplotlib), `fitwise[odr]` (odrpack), `fitwise[units]` (pint, uncertainties), `fitwise[symbolic]` (sympy), `fitwise[batch]` (pandas), `fitwise[all]`
- Python ≥ 3.11

## 7. Designprinzipien

1. **Ein Rückgabetyp.** `FitResult` sieht immer gleich aus, egal welcher Solver lief.
2. **Keine stillen Entscheidungen.** Jede Automatik (Solverwahl, Startwerte, Kovarianzskalierung) ist im Result dokumentiert und überschreibbar.
3. **Warnungen statt falscher Zahlen.** Nicht konvergiert, Parameter an der Grenze, χ²/ndf außerhalb des 99-%-Intervalls, Korrelation > 0.95, singuläre Kovarianz, Unsicherheit > 100 %: Alles erscheint in `result.warnings` und im Report und wird als `FitWarning` über das `warnings`-Modul emittiert. Ein nicht konvergierter Fit hat `result.ok == False`.
4. **Reproduzierbar.** `result.to_dict()` enthält alles, um den Fit exakt zu wiederholen: Modell-Identität, Daten-Hash, Startwerte inkl. Strategie, Solver und Optionen, Seed des Multi-Starts, Versionen.
5. **Dünn bleiben.** Wer mehr braucht, bekommt mit `result.raw` das Objekt des darunterliegenden Backends (z. B. `scipy.optimize.OptimizeResult`).
6. **Schnell genug für Pipelines.** Der Overhead gegenüber direktem `curve_fit` liegt bei einem Gaußfit mit 1000 Punkten unter 5 ms. Lazy Imports für alle optionalen Pakete.

## 8. Roadmap

Details, Arbeitspakete und Aufwände stehen in [`PLAN.md`](PLAN.md).

| Version | Kern | Ziel |
|---|---|---|
| **v0.1 – MVP** | `fit()` mit `yerr`/`p0`/`bounds`/`fixed`, `least_squares`-Backend, 10 eingebaute Modelle mit Heuristiken, Komposition `+`, `FitResult`, Text-Report, Warnungen, `plot()`, `to_dict()` | Gaußfit in 3 Zeilen; erste Testnutzer:innen |
| **v0.2 – Automatik** | Linearitätserkennung + direkte Lösung, Startwert-Kaskade für freie Funktionen (Rollenanalyse, Linearisierung, Multi-Start), String-Modelle, robuste Losses | „Eigene Funktion ohne `p0`“ funktioniert |
| **v0.3 – Unsicherheiten** | ODR-Backend (`odrpack`), Konfidenz-/Prädiktionsbänder, Bootstrap, Profile-Likelihood, `pint`/`uncertainties` | x- und y-Fehler; vollständige Fehlerrechnung |
| **v0.4 – Skalierung & Export** | `fit_many()` inkl. globaler Fits, `FitCollection`/DataFrame, Markdown/LaTeX/HTML-Export, Stil-Backend | Pipeline- und Publikationsreife |
| **v1.0 – Stabil** | API-Freeze, NIST-StRD-Suite in CI, vollständige Doku, Brücke zu `curvelib` | Produktiver Einsatz |

## 9. Erfolgskriterien

**Funktional (in CI messbar):**

- Ein Gaußfit mit Fehlerbalken, Report und Plot gelingt in ≤ 3 Zeilen.
- NIST StRD (alle 27 nichtlinearen Datensätze): **100 %** erreichen die zertifizierten Werte (≥ 6 signifikante Stellen) von Startpunkt 1 aus.
- NIST StRD ohne jeden Startwert (nur Automatik): **100 %** der Datensätze mit niedriger und mittlerer Schwierigkeit, **≥ 80 %** derer mit höherer Schwierigkeit. Jeder verbleibende Fall endet mit `ok == False` und einer klaren Warnung.
- Kein Fall in der Testsuite, in dem ein nicht konvergierter Fit ohne Warnung Zahlen liefert (Property-basierte Tests).
- Die Parameterunsicherheiten sind kalibriert: In Monte-Carlo-Tests mit bekannter Wahrheit liegt die Überdeckung der 1σ-Intervalle bei 68 ± 3 %.

**Adoption (nach v0.4):**

- Umfrage unter ≥ 20 Testnutzer:innen (Praktika, Arbeitsgruppen): > 50 % sagen „Ich brauche Origin für Standardfits nicht mehr“.
- Ein Praktikum oder eine Lehrveranstaltung setzt `fitwise` regulär ein.

## 10. Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Startwert-Heuristiken versagen bei exotischen Modellen | Kaskade mit Multi-Start-Fallback; klare Meldung; `p0` jederzeit parameterweise überschreibbar |
| Rollenanalyse ist bei stark gekoppelten Parametern mehrdeutig | Nur Rollen mit hoher Konfidenz verwenden, Rest per Multi-Start; Benchmarksuite mit 30+ realen Modellen |
| „Noch ein Fit-Wrapper“ – geringe Sichtbarkeit | Positionierung über Report und Origin-Vergleich; Galerie mit Beispielen aus realen Praktika; Vorträge in der eigenen Community |
| Abhängigkeit von lmfit-API-Änderungen | Backend-Protokoll; CI gegen minimale und neueste Versionen; Pinning der Obergrenze nur bei bekanntem Bruch |
| `scipy.odr` wird entfernt (deprecated seit SciPy 1.17, Entfernung in 1.19) | ODR von Anfang an über `odrpack` statt `scipy.odr` |
| Nutzer vertrauen der Automatik blind | Jede Automatik im Report sichtbar; Warnungen nur mit explizitem Flag abschaltbar; Doku-Kapitel „Wann man dem Fit nicht trauen sollte“ |
| Scope Creep (GUI, Bayes, Datenimport) | Nicht-Ziele (Abschnitt 5) als verbindliche Referenz in Issue-Diskussionen |

## 11. Entscheidungen

| Frage | Entscheidung | Begründung |
|---|---|---|
| Lizenz | **BSD-3-Clause** | Gleich wie lmfit, scipy, numpy; keine Reibung im Ökosystem |
| String-Modelle | **v0.2**, über `asteval` (kommt mit lmfit) | Keine neue Pflichtabhängigkeit; sympy nur optional für Ableitungen |
| Komposition `+` | **v0.1** | Wird für reale Spektren sofort gebraucht (Peak + Untergrund) |
| Nichtlinearer Solver | **`scipy.optimize.least_squares`** direkt statt lmfit-Minimizer | Nur so 27/27 NIST-Datensätze von beiden Startpunkten (ADR 5); lmfit bleibt für `conf_interval` und `asteval` |
| ODR-Implementierung | **`odrpack`** statt `scipy.odr` | `scipy.odr` ist deprecated; ODRPACK95 unterstützt Bounds |
| `weights`-Argument | **entfällt** | Doppelte Semantik zu `yerr`; Fehlerquelle |
| Kovarianzskalierung | mit `yerr` absolut, ohne `yerr` skaliert, beides im Report | Entspricht der physikalischen Erwartung; kein stilles `absolute_sigma` |
| Name | **`fitwise`** | Frei auf PyPI; Reserve: `fitsmith`, `snapfit` |

## 12. Offene Fragen

- Soll `fit()` auch `pandas`-Spalten und `xarray` direkt akzeptieren (Einheiten/Labels aus Metadaten)?
- Brauchen globale Fits (`shared=`) eine eigene API (`fw.fit_global`) oder reicht `fit_many`?
- Gehört der Kerndatensatz der Doku-Beispiele (Praktikumsversuche) in ein separates Paket `fitwise-data`?
