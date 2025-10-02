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
1) Visualisation rapide (GIF)
python heatmap_evolutive.py --beacons beacons_world_template.csv --segments segments_template.csv \
  --counts porta_di_roma_counts.csv --floor 0 --viz segments --hour-min 10 --hour-max 22 \
  --segment-radius 5 --out heatmap_floor0.gif -v

2) Dashboard interactif par jour
python heatmap_interactive.py --beacons beacons_world_template.csv --segments segments_template.csv \
  --counts porta_di_roma_counts.csv --floor 0 --aggregate-by day --normalize global \
  --hour-min 10 --hour-max 22 --out heatmap_interactive_floor0.html -v

3) Analyse des flux & rapport complet
python analyze_paths.py --beacons beacons_world_template.csv --segments segments_template.csv \
  --counts porta_di_roma_counts.csv --hour-min 10 --hour-max 22 --segment-radius 5 \
  --topk 20 --outdir analysis_out -v


Interprétation attendue du HTML
- Top segments : couloirs ou axes à trafic fort.
- Matrice inter-étages : parts relatives de flux montants/descendants et vers l’extérieur.
- Heatmaps : zones de concentration selon l’heure et la date.
- Sankey : schéma clair des échanges entre niveaux.



🏬 Restitution — Analyse des flux visiteurs
Contexte
L’étude a consisté à exploiter les données de passage collectées par les beacons du centre afin de :
Cartographier les zones les plus fréquentées (heatmaps, densité sur les segments).
Comprendre les itinéraires dominants : comment les visiteurs se déplacent entre entrées, commerces, zones de restauration et sorties.
Analyser les flux verticaux entre les différents niveaux (-1, 0, 1).
Identifier des opportunités d’optimisation de l’expérience client et des espaces.


1️⃣ Constats clés
🔹 Fréquentation par zones

Zones d’entrée / sortie très concentrées : la majorité des flux se concentre autour des accès principaux et des escalators proches des entrées.
Couloirs secondaires peu empruntés : certains segments du plan restent faiblement fréquentés, notamment aux extrémités ou près de boutiques fermées.
Concentration horaire : pics de trafic entre 14h et 18h, avec une montée progressive dès 11h et un reflux net après 19h.

🔹 Flux verticaux

Transitions majeures : RDC → 1er étage domine sur l’axe central des escalators ; peu de remontées du -1 vers 0.
Sorties directes depuis RDC : forte part des flux aboutissant à OUTDOOR dès qu’ils passent par les allées principales.

🔹 Parcours typiques détectés

Entrée principale (niveau 0) → escalator central → zone restauration/food court (niveau 1) → retour sortie principale.
Parking (-1) → escalator périphérique → rez-de-chaussée commerces → sortie proche.
Certains chemins “bouclés” autour d’ancres commerciales fortes (ex : grandes enseignes en 0 puis food court en 1).

2️⃣ Opportunités & Recommandations

🛍️ Optimisation commerciale
Valoriser les zones à fort trafic : placer promotions, pop-up stores, corners saisonniers sur les segments les plus empruntés.
Redynamiser les couloirs faibles :
- Animer par signalétique claire et incitante (fléchage, PLV lumineuse).
- Installer des points d’intérêt (bornes interactives, corners événementiels).

🧭 Amélioration de l’orientation
Signalétique claire vers les zones sous-fréquentées : indiquer explicitement food court, loisirs ou magasins clés dans les zones calmes.
Guidage digital : proposer une carte interactive dans l’appli ou sur bornes pour fluidifier les parcours.

🚶 Gestion des flux et confort
Prévoir renforts sécurité/nettoyage aux heures de pointe sur les axes à très forte densité.
Fluidifier les escalators/ascenseurs dominants (gestion des sens, affichages temps d’attente).

📈 Suivi et itération
Répéter la mesure après mise en place d’actions pour évaluer l’efficacité (ex : +X% trafic dans une zone revalorisée).
Intégrer des comparaisons temporelles (week-end vs semaine, saisonnalité).

3️⃣ Points techniques clés (pour crédibiliser l’analyse)

Heatmaps spatio-temporelles : densité des passages filtrée sur plages horaires pertinentes (10h–22h).
Projection beacon→segments : algorithme qui répartit le trafic selon la proximité physique pour représenter les vraies allées empruntées.
Matrice inter-étages : quantification des flux montants/descendants, part de sortie.
Visualisations interactives (HTML & Sankey) pour exploration libre des données par vos équipes.

4️⃣ Prochaines étapes proposées

Session d’exploration interactive avec les équipes du centre (utilisation du dashboard HTML pour naviguer dans les flux).
Ateliers opérationnels :
sélectionner les zones à booster commercialement,
planifier signalétique ou events test.
Mesure post-action (avant/après) pour valider les impacts (trafic, orientation, fréquentation).

⚡ Synthèse

Les données montrent des chemins dominants très clairs et des zones peu exploitées.
Des actions ciblées (signalétique, animations, repositionnement commercial) peuvent rééquilibrer les flux, améliorer l’expérience client et optimiser la valeur locative des emplacements.

🧩 Améliorations possibles
- Ajouter des comparaisons temporelles (week-end vs semaine).
- Analyser les pics horaires par zone.
- Créer un mapping enrichi si les directions deviennent plus détaillées.
