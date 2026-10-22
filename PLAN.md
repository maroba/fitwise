# Projektplan: `fitwise`

*Stand: 2026-10-22 · Bezug: [`vision_fitwise.md`](vision_fitwise.md)*

Annahmen: eine Person, etwa 50 % Kapazität. Die Aufwände sind in Personentagen (PT) angegeben, die Kalenderwochen bei dieser Kapazität.

| Phase | Version | Aufwand | Kalender | Zieltermin |
|---|---|---|---|---|
| 0 | Setup | 3 PT | 1 Woche | 2026-10-30 |
| 1 | v0.1 MVP | 18 PT | 7 Wochen | 2026-12-18 |
| 2 | v0.2 Automatik | 15 PT | 6 Wochen | 2027-02-12 |
| 3 | v0.3 Unsicherheiten | 12 PT | 5 Wochen | 2027-03-19 |
| 4 | v0.4 Skalierung & Export | 10 PT | 4 Wochen | 2027-04-16 |
| 5 | v1.0 Stabil | 10 PT | 4 Wochen + Feedbackphase | 2027-06 |

Puffer: ca. 20 % sind in den Kalenderwochen enthalten, nicht in den PT.

---

## Phase 0 – Setup (3 PT)

| # | Aufgabe | Ergebnis | Status |
|---|---|---|---|
| 0.1 | Git-Repo, BSD-3-Lizenz, `README.md`, `CHANGELOG.md` | Lokales Repo | ✅ |
| 0.1b | GitHub-Repo `maroba/fitwise` anlegen und pushen, GitHub Pages aktivieren | Öffentliches Repo | ✅ |
| 0.2 | `pyproject.toml` (hatchling), `src/`-Layout, Extras laut Vision §6, Python ≥ 3.11 | `uv sync --all-extras` funktioniert | ✅ |
| 0.3 | Tooling: `uv`, `ruff` (Lint + Format), `mypy --strict` für `src/`, `pre-commit` | `pre-commit run -a` grün | ✅ |
| 0.4 | Tests: `pytest`, `pytest-cov`, `hypothesis`; Marker `slow`, `nist`; Coverage-Ziel 90 % | Suite läuft | ✅ |
| 0.5 | CI (GitHub Actions): Python 3.11–3.14 (neueste Abhängigkeiten) + 3.11 mit Mindestversionen; Lint, Typen, Tests; Release über Trusted Publishing | Workflows liegen in `.github/workflows/` | ✅ |
| 0.6 | Doku-Gerüst: Sphinx + pydata-Theme + MyST + `sphinx-gallery` (ADR 4); ADRs 1–4; Deployment auf GitHub Pages | `sphinx-build -W` grün | ✅ |
| 0.7 | Trusted Publisher auf PyPI einrichten (Pending Publisher) | Release per Tag möglich | ✅ (kein Platzhalter-Release, direkt 0.1.0) |
| 0.8 | Testdaten: 27 NIST-StRD-Datensätze in `tests/data/nist/`, Loader `tests/nist_strd.py` mit Modellfunktionen; Prüfung: zertifizierte Werte reproduzieren die zertifizierte RSS (rel. 1e-8) | `tests/test_nist_fixtures.py` grün | ✅ |

Befund aus 0.8: In `Rat43.dat` stehen 9 Freiheitsgrade, korrekt sind 11 (15 − 4). Die zertifizierte Residuen-Standardabweichung rechnet mit 11. Der Loader liefert deshalb `dof` berechnet und `dof_stated` wie in der Datei.

**Baseline (2026-10-22):** reines `scipy.optimize.least_squares(method="lm")` mit engen Toleranzen, Kriterium ≥ 6 korrekte signifikante Stellen in allen Parametern:

| Schwierigkeit | Start 1 | Start 2 | alle Startwerte = 1 |
|---|---|---|---|
| niedrig | 7/8 | 7/8 | 4/8 |
| mittel | 9/11 | 8/11 | 3/11 |
| hoch | 6/8 | 8/8 | 2/8 |
| **gesamt** | **22/27** | **23/27** | **9/27** |

Fehlschläge von Start 1: Lanczos3, Kirby2, Hahn1, BoxBOD, Bennett5. Das v0.1-Ziel (27/27 von Start 1) braucht also mehr als einen Wrapper, etwa Parameterskalierung (`x_scale="jac"`), Nachiteration und Prüfung der Konvergenzkriterien. Der Wert „alle = 1“ ist der Ausgangspunkt für das v0.2-Ziel ohne Startwerte.

---

## Phase 1 – v0.1 MVP (18 PT)

**Ziel:** `fw.fit(fw.models.gaussian, x, y, yerr=dy)` liefert einen vollständigen, ehrlichen Report und Plot.

### Arbeitspakete

| # | Paket | Inhalt | PT | Abhängig von |
|---|---|---|---|---|
| 1.1 | Kerntypen | `FitProblem`, `BackendResult`, `Parameter` (`value`, `stderr`, `bounds`, `fixed`, `guess`, `guess_strategy`), Backend-Protokoll | 2 | – |
| 1.2 | Eingabevalidierung | Arrays/Listen, Shapes, `yerr` als Skalar/Array, NaN-Policy, `p0`/`bounds`/`fixed` als Dicts mit Namensprüfung und hilfreichen Fehlermeldungen („Unknown parameter 'sigam'. Did you mean 'sigma'?“) | 1.5 | 1.1 |
| 1.3 | Model-Klasse | Signatur-Introspektion (erstes Argument = unabhängige Variable, Defaults → `p0`), Metadaten (Rollen, Einheiten, Linearität), `+`-Komposition mit automatischen Präfixen | 2.5 | 1.1 |
| 1.4 | lmfit-Backend | `leastsq`/`least_squares` je nach Bounds, Fixierung, Konvergenzinfo, Kovarianz-Semantik laut Vision §4.5 | 2 | 1.1, 1.3 |
| 1.5 | Eingebaute Modelle (10) | `constant`, `linear`, `polynomial(n)`, `gaussian`, `lorentzian`, `voigt`, `exponential`, `sigmoid`, `powerlaw`, `damped_sine`; je Funktion, Docstring mit Formel, Rollen, `guess()`, Unit-Tests | 3 | 1.3 |
| 1.6 | Guess für Komposition | Sequenziell: Untergrund vom Rand schätzen, Peaks auf den Residuen | 1 | 1.5 |
| 1.7 | FitResult + Gütemaße | χ², ndf, χ²/ndf mit p-Wert, R², RMSE, AIC/BIC, Korrelationsmatrix, `ok`, `eval(x)`, `residuals`, `raw` | 1.5 | 1.4 |
| 1.8 | Warnungen | `FitWarning`-Hierarchie: `NotConverged`, `ParameterAtBound`, `PoorChiSquare`, `StrongCorrelation`, `SingularCovariance`, `LargeUncertainty`; im Result, im Report und über `warnings` | 1 | 1.7 |
| 1.9 | Text-Report | Layout laut Vision §4.6, Rundung nach PDG-Regel, Unicode/ASCII-Modus, `__repr__`/`__str__` | 1.5 | 1.7, 1.8 |
| 1.10 | Plot | Daten + Fehlerbalken, fein abgetastete Fitkurve, normierte Residuen, Rückgabe `Figure`/`Axes`, lazy matplotlib-Import (Band erst in v0.3) | 1 | 1.7 |
| 1.11 | Serialisierung | `to_dict()`/`from_dict()` mit Schema-Version 1, Daten-Hash (SHA-256 über Arrays), Versionen | 0.5 | 1.7 |
| 1.12 | Doku + Beispiele | Quickstart, Konzept „Unsicherheiten“, 10 Galerie-Beispiele (Spektroskopie-Peak, RC-Entladung, Federpendel gedämpft, Titrationskurve, Arrhenius-Plot, Kennlinie Diode als Potenzgesetz, Zerfall, Kalibriergerade, Resonanzkurve, Peak + Untergrund) | 2 | alle |

### Teststrategie v0.1

- **Referenzvergleich:** Für alle eingebauten Modelle muss das Ergebnis mit direktem `lmfit` und `curve_fit` auf 1e-8 relativ übereinstimmen (Werte und Kovarianz).
- **Synthetische Daten mit bekannter Wahrheit** (`hypothesis`): Parameter zufällig, Rauschen bekannt → Fit ohne `p0` konvergiert in ≥ 99 % der Fälle für eingebaute Modelle.
- **Ehrlichkeit:** Für absichtlich unlösbare Probleme (falsches Modell, zu wenig Punkte, unidentifizierbare Parameter) muss immer `ok == False` gelten oder eine Warnung gesetzt sein.
- **NIST StRD von Startpunkt 1:** alle 27 Datensätze (Marker `nist`).
- **Report-Snapshots:** Golden-File-Tests für das Textlayout.

### Stand (2026-10-22)

Erledigt: 1.1–1.12, Release 0.1.0. Offen: Ankündigung an Testnutzer:innen.

Abweichungen vom Plan:

- **1.4:** Der Risiko-Checkpoint hat gegriffen. Statt des lmfit-Minimizers ruft das Backend
  `scipy.optimize.least_squares` direkt auf (TRF, 3-Punkt-Jacobi, normierte Parameter,
  Neustarts). Nur so werden alle 27 NIST-Probleme von beiden Startpunkten gelöst; Messungen in
  ADR 5.
- **Referenzvergleich:** Gegen `curve_fit` stimmen die Werte auf < 1e-5 Standardfehler und die
  Kovarianz auf 1e-4 relativ überein. Das 1e-8-Ziel ist bei Parametern nahe 0 nicht sinnvoll,
  weil beide Solver an einer Toleranz stoppen (gleiches χ² auf 15 Stellen).
- **Konvergenz ohne Startwerte:** Alle 10 Modelle und 4 typische Kompositionen konvergieren in
  100/100 zufälligen Fällen (`tests/test_models.py`, Ziel war ≥ 99 %).
- **`from_dict()`** entfällt vorerst. Eine Python-Funktion lässt sich nicht aus JSON
  rekonstruieren; `to_dict()` enthält alles, um den Fit mit demselben Modell zu wiederholen.
- **Korrelationswarnung:** Schwelle 0.999 statt 0.95. Die Beispiele zeigten, dass
  Korrelationen von 0.95–0.999 bei Arrhenius, Potenzgesetzen und Geraden fern von x = 0 normal
  sind; eine Warnung dort wäre Rauschen. Korrelationen > 0.5 stehen weiter im Report.
- **Kalibrierung geprüft:** Pulls (Fit − Wahrheit)/σ im Zerfallsbeispiel über 300 Seeds:
  Mittelwert −0.06, Standardabweichung 1.03.
- **Kompatibilität:** Bei scipy 1.12 ignoriert `least_squares(method="lm")` den 3-Punkt-Jacobi
  stillschweigend; fitwise berechnet ihn für LM deshalb selbst.

### Definition of Done v0.1

- [x] Die drei Zeilen aus Vision §2 laufen und liefern Report und Plot
- [x] NIST StRD von Startpunkt 1: 27/27 (auch von Startpunkt 2)
- [x] Coverage ≥ 90 %, `mypy --strict` grün (98 %)
- [x] Doku online mit Quickstart und 10 Beispielen
- [x] Release `0.1.0` auf PyPI
- [ ] Ankündigung an 5–10 Testnutzer:innen mit Feedback-Formular

---

## Phase 2 – v0.2 Automatik (15 PT)

**Ziel:** Eigene Funktionen ohne `p0` fitten, lineare Modelle ohne Startwerte lösen.

| # | Paket | Inhalt | PT |
|---|---|---|---|
| 2.1 | Linearitätserkennung | Numerischer Jacobi-Test an 3 Zufallspunkten, Toleranz relativ zur Skalennorm; Caching pro Modell | 1.5 |
| 2.2 | Linear-Backend | `lstsq` mit Gewichtung, `lsq_linear` bei Bounds, Fixierung durch Spaltenelimination, exakte Kovarianz | 1.5 |

| 2.3 | Rollenanalyse | Störung jedes Parameters an einem Referenzpunkt; Klassifikation (Offset, Amplitude, x-Shift, x-Skala, sonstige) per Korrelation der Ableitung mit Templates; Konfidenzwert | 3 |
| 2.4 | Schätzer pro Rolle | Offset/Amplitude/Shift/Skala aus Datenstatistik (Median, Spannweite, Extremum, FWHM, Wendepunkt) | 2 |
| 2.5 | Linearisierung + Variable Projection | Log-Transformation für exp/power; lineare Parameter bei gegebenen nichtlinearen exakt lösen | 2 |
| 2.6 | Multi-Start | Latin-Hypercube in den Skalen aus 2.3/2.4 bzw. Bounds; `max_starts`, Seed, frühe Terminierung | 1.5 |
| 2.7 | String-Modelle | `"a*exp(-x/tau) + c"` über asteval, Parametererkennung, Whitelist erlaubter Funktionen, optional sympy-Ableitungen | 1.5 |
| 2.8 | Robuste Losses | `loss=` über `least_squares`; Hinweis im Report, dass die Kovarianz dann approximativ ist | 1 |
| 2.9 | Benchmarksuite „freie Funktionen“ | 30+ reale Modelle aus Literatur/Praktika als Testkatalog; Messung der Konvergenzrate ohne `p0` | 1 |

**Stand 2.1/2.2 (2026-10-22): erledigt.** Prüfung zusätzlich gegen die NIST-StRD-Linear-Suite (11 Datensätze, ADR 6): Filip 7.4 Stellen, alle anderen ≥ 7.7 außer Wampler5 (5.7 Stellen bei Standardfehlern von 10³–10⁷ × Wert).

**DoD v0.2:** NIST StRD ohne Startwerte: 100 % niedrige/mittlere, ≥ 80 % höhere Schwierigkeit; Konvergenzrate der Benchmarksuite ≥ 85 %; `result.guess_strategy` für jeden Parameter gesetzt und im Report sichtbar.

**Stand 2.3–2.6, 2.9 (2026-10-22): erledigt, mit Abweichung (ADR 7).** Die Benchmarksuite (2.9, 32 Modelle) wurde zuerst gebaut. Multi-Start mit Variable Projection und datenbasierten Kandidaten (2.5, 2.6) erreicht 99.7 % von 3300 Fits; die Rollenanalyse (2.3, 2.4) wurde deshalb nicht implementiert. DoD v0.2: Benchmark ≥ 85 % ✅ (99.7 %); NIST ohne Startwerte: hoch 8/8 ✅, niedrig 7/8 und mittel 10/11 ❌ (Gauss2, Gauss3: zwei stark überlappende Peaks plus Zerfall als eine Black-Box-Funktion; mit eingebauten Modellen lösbar). `guess_strategy` für jeden Parameter gesetzt ✅.

**Stand 2.7, 2.8 (2026-10-22): erledigt.** String-Modelle mit eigenem `ast`-Parser statt `asteval` (sicherer, schneller, keine Abhängigkeit); `^` als Potenz, Ersetzung vor dem Parsen wegen der Operator-Priorität. Robuste Fits zweistufig (gewöhnlicher Fit, dann TRF mit Loss); Kovarianz der äquivalenten gewichteten Kleinste-Quadrate, Skala ohne `yerr` über 1.4826 · MAD. Kalibrierung per Monte Carlo: Pull-Streuung 0.95–1.09 mit `yerr`, 0.96–1.17 ohne.

**Entscheidungspunkt nach 2.3:** Liefert die Rollenanalyse auf der Benchmarksuite weniger als 60 % korrekte Klassifikationen, wird sie auf Offset/Amplitude reduziert und der Rest über Multi-Start gelöst. Die Abweichung wird in der Vision dokumentiert.

---

## Phase 3 – v0.3 Unsicherheiten (12 PT)

| # | Paket | Inhalt | PT |
|---|---|---|---|
| 3.1 | ODR-Backend | `odrpack` (Extra `[odr]`), Bounds, Mapping auf `BackendResult`, Report „x and y errors“ | 2.5 |
| 3.2 | Bänder | Konfidenzband über Delta-Methode (Jacobi × Kovarianz); Prädiktionsband inkl. `yerr`; Integration in `plot(band=...)` | 2 |
| 3.3 | Bootstrap | Residuen- und Fall-Bootstrap, parallelisierbar, Seed; Perzentil-Intervalle | 2 |
| 3.4 | Profile-Likelihood | Über `lmfit.conf_interval`, asymmetrische Fehler im Report | 1.5 |
| 3.5 | Einheiten | `pint`-Quantities als Eingabe; Einheiten der Parameter aus Modell-Metadaten oder Dimensionsanalyse; Achsenbeschriftung | 2.5 |
| 3.6 | uncertainties-Brücke | `result.params.as_ufloat()` mit Korrelationen (`correlated_values`) | 0.5 |
| 3.7 | Kalibrierungstest | Monte-Carlo-Überdeckung der 1σ-Intervalle für alle Methoden | 1 |

**DoD v0.3:** Überdeckung 68 ± 3 % für Kovarianz, Bootstrap und ODR in Monte-Carlo-Tests; ODR-Ergebnisse stimmen mit `odrpack` direkt überein; Beispiel „x- und y-Fehler“ in der Galerie.

---

## Phase 4 – v0.4 Skalierung & Export (10 PT)

| # | Paket | Inhalt | PT |
|---|---|---|---|
| 4.1 | `fit_many` | Getrennte Fits, `warm_start`, Fortschrittsanzeige (optional `tqdm`), Parallelisierung über `concurrent.futures` | 2 |
| 4.2 | Globale Fits | `shared=[...]`: gemeinsame Parameter über Datensätze, gestapeltes Residuum | 2.5 |
| 4.3 | `FitCollection` | `.to_dataframe()`, Filter nach Warnungen, Sammelplot | 1.5 |
| 4.4 | Export | `to_markdown()`, `to_latex()` (siunitx-kompatibel), `_repr_html_()` | 2 |
| 4.5 | Stil-Backend | `plot(style=...)` als Hook; Adapter für `plotstyle`, sobald verfügbar | 0.5 |
| 4.6 | Performance | Benchmark gegen `curve_fit`; Overhead < 5 ms beim Gaußfit mit 1000 Punkten; Lazy Imports prüfen (`python -X importtime`) | 1.5 |

**DoD v0.4:** 200 Spektren × Gauß + linear in < 10 s auf einem Laptop; Jupyter-Darstellung in der Doku; Umfrage unter Testnutzer:innen gestartet.

---

## Phase 5 – v1.0 Stabil (10 PT)

| # | Paket | Inhalt | PT |
|---|---|---|---|
| 5.1 | API-Review | Alle öffentlichen Namen prüfen, Deprecations aus 0.x entfernen, `__all__` festschreiben | 2 |
| 5.2 | NIST-Suite in CI | Beide Startpunkte plus Automatik, Report als CI-Artefakt | 1 |
| 5.3 | Restliche Modelle | `pseudo_voigt`, `exponential(n)`, `boltzmann`, `hill`, `michaelis_menten`, `arrhenius`, `linear_breakpoint` | 2.5 |
| 5.4 | Doku | Kapitel „Wann man dem Fit nicht trauen sollte“, Migrationsguides von `curve_fit`/Origin, API-Referenz vollständig | 2.5 |
| 5.5 | `curvelib`-Brücke | `result.to_curve()`, sobald `curvelib` eine stabile API hat; sonst auf 1.1 verschieben | 1 |
| 5.6 | Release | Changelog, Ankündigung (Mailingliste, scientific-python discourse), conda-forge-Feedstock | 1 |

**DoD v1.0:** Alle Erfolgskriterien aus Vision §9 (funktional) erfüllt; Semver-Versprechen in der Doku.

---

## Querschnitt

### Konventionen

- **Branching:** `main` ist immer releasefähig; Feature-Branches mit PRs, auch allein (CI als Gate).
- **Commits:** Conventional Commits (`feat:`, `fix:` …) → automatisch erzeugter Changelog.
- **Docstrings:** NumPy-Stil; jedes eingebaute Modell mit Formel (LaTeX), Parametertabelle und Beispiel.
- **Öffentliche API:** nur was in `fitwise/__init__.py` und `fitwise.models` exportiert ist; alles andere `_privat`.
- **Entscheidungen:** kurze ADRs in `docs/adr/` (eine Datei pro Entscheidung, die in Vision §11 steht oder später dazukommt).

### Feedback-Schleifen

- Nach v0.1: 5–10 Testnutzer:innen aus der eigenen Umgebung (Arbeitsgruppen, Praktikumsbetreuung) mit je einem echten Datensatz; strukturiertes Feedback (Was hat gefehlt? Wo hat der Report verwirrt?).
- Nach v0.2: Vergleichsstudie „derselbe Fit in Origin vs. fitwise“ (Zeit, Zeilen, Ergebnisgleichheit) als Doku-Seite und Marketing-Material.
- Nach v0.4: Umfrage für das Adoptionskriterium (≥ 20 Personen).

### Risiko-Checkpoints

| Zeitpunkt | Prüfung | Bei Nichterfüllung |
|---|---|---|
| Ende 1.4 | lmfit-Backend deckt Bounds + Fixierung + Kovarianz-Semantik sauber ab | Direkt `least_squares` nutzen, lmfit nur für `conf_interval` → **eingetreten**, siehe ADR 5 |
| Ende 2.3 | Rollenanalyse ≥ 60 % korrekt | Umfang reduzieren (siehe Phase 2) |
| Ende 3.1 | `odrpack` stabil und auf PyPI für alle Plattformen verfügbar | ODR mit eigener Implementierung über `least_squares` auf erweitertem Residuum |
| v0.4 | ≥ 5 aktive externe Nutzer:innen | Positionierung überdenken, bevor v1.0-Aufwand investiert wird |

---

## Nächste Schritte

1. Ankündigung von 0.1.0 an 5–10 Testnutzer:innen mit je einem echten Datensatz; Feedback sammeln.
2. Phase 2 (v0.2): Linearitätserkennung und direkte Lösung (2.1, 2.2), dann die Startwert-Kaskade für freie Funktionen (2.3–2.6).
