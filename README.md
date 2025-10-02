# 🏬 SEIKI — Analyse des déplacements dans un centre commercial

Ce projet fournit une chaîne complète pour analyser et visualiser les déplacements des visiteurs dans un centre commercial à partir des données de comptage de beacons.

Il répond à un **test technique d’Analytics Engineer** en combinant :
- prétraitement des données brutes (horaires, étages, directions),
- détection des chemins les plus empruntés,
- analyse des flux entre étages,
- visualisation dynamique (GIF et HTML interactif).

---

## 📂 Structure du projet
├── heatmap_evolutive.py # Animation GIF (heatmap ou segments) sur une période
├── heatmap_interactive.py # Visualisation interactive HTML multi-journée
├── analyze_paths.py # Analyse des flux et détection des chemins dominants
├── beacons_world_template.csv # Template beacons (id, x/y mètres, étage)
├── segments_template.csv # Template segments (x1,y1,x2,y2,floor)
├── porta_di_roma_counts.csv # Données de comptage (beacons, dates, heures, directions)
└── README.md # Documentation du projet

---

## 🔧 Données d’entrée

### `beacons_world_template.csv`
| id        | x_m  | y_m  | floor |
|-----------|------|------|-------|
| B001      | 10.5 |  4.2 | 0     |
| B002      | 15.8 | 12.0 | 1     |

- **id** : identifiant unique de chaque beacon  
- **x_m, y_m** : coordonnées en mètres (plan 2D)
- **floor** : étage (entier, ex : -1, 0, 1)

### `segments_template.csv`
| segment_id | x1_m | y1_m | x2_m | y2_m | floor |
|------------|------|------|------|------|-------|
| S001       | 10.0 | 5.0  | 15.0 | 5.0  | 0     |

- Définit les **segments physiques** (couloirs, passages)

### `porta_di_roma_counts.csv`
| beacon | date_it    | hour | count | direction_out | direction_in |
|--------|-----------|------|-------|---------------|--------------|
| B001   | 2024-06-01 | 10   | 12    | 0             | outdoor      |
| B002   | 2024-06-01 | 11   | 8     | 1             | 0            |

- `beacon` : identifiant du beacon où est mesuré le passage  
- `date_it` : date (AAAA-MM-JJ)  
- `hour` : heure entière de comptage  
- `count` : nombre de passages  
- `direction_in` / `direction_out` : direction du mouvement (`-1`, `0`, `1`, `outdoor`)

---

## 🚀 Scripts & Modélisations

### 1️⃣ `heatmap_evolutive.py` — GIF évolutif

Permet de créer une **heatmap dynamique** ou une **animation des segments** sur une plage temporelle.

**Usage :**

# 1) Visualisation rapide (GIF)
python heatmap_evolutive.py --beacons beacons_world_template.csv --segments segments_template.csv \
  --counts porta_di_roma_counts.csv --floor 0 --viz segments --hour-min 10 --hour-max 22 \
  --segment-radius 5 --out heatmap_floor0.gif -v

# 2) Dashboard interactif par jour
python heatmap_interactive.py --beacons beacons_world_template.csv --segments segments_template.csv \
  --counts porta_di_roma_counts.csv --floor 0 --aggregate-by day --normalize global \
  --hour-min 10 --hour-max 22 --out heatmap_interactive_floor0.html -v

# 3) Analyse des flux & rapport complet
python analyze_paths.py --beacons beacons_world_template.csv --segments segments_template.csv \
  --counts porta_di_roma_counts.csv --hour-min 10 --hour-max 22 --segment-radius 5 \
  --topk 20 --outdir analysis_out -v


Interprétation attendue du HTML
- Top segments : couloirs ou axes à trafic fort.
- Matrice inter-étages : parts relatives de flux montants/descendants et vers l’extérieur.
- Heatmaps : zones de concentration selon l’heure et la date.
- Sankey : schéma clair des échanges entre niveaux.


🧩 Améliorations possibles
- Ajouter des comparaisons temporelles (week-end vs semaine).
- Analyser les pics horaires par zone.
- Créer un mapping enrichi si les directions deviennent plus détaillées.
