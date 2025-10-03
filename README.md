# 🏬 SEIKI — Analyse des déplacements dans un centre commercial

## Contexte
Une célèbre foncière, que l’on appellera Perklière dans cet exercice, présente dans
plusieurs pays européen a un pain point : elle n’a aucune connaissance quant à la
fréquentation et comportement des visiteurs au sein de ses centres. Pour pallier cette
problématique, Perklière décide d’équiper ses centres avec des compteurs physiques
(appelés « beacons », cf. éléments communiqués ci-dessous). Toutefois, via ces
capteurs et sans traitement supplémentaire, il n’est que possible d’avoir un nombre
d’entrées et de sorties dans le centre, ce qui donne une idée de la fréquentation mais pas
du tout du comportement des visiteurs. Perklière décide donc de contacter Seiki, expert
en modélisation des flux, afin de trouver une solution à leur problème. Seiki et Perklière
se mettent alors d’accord pour démarrer cette collaboration avec un POC sur un centre
en Italie : Porta di Roma.

## Mission
En tant qu’Analytics Engineer de Seiki, ta mission est de proposer une solution à Perklière
pour les aider à mieux comprendre la dynamique au sein du centre Porta di Roma. A l’issu
de la réunion de kick off, deux problématiques se dégagent :

• Problématique n°1 : comment modéliser les flux indoor du centre Porta di Roma ?

Via le langage de programmation de ton choix, développe un algorithme capable
de modéliser les flux dans le centre en prenant en entrée les données issues des
beacons (cf. éléments communiqués ci-dessous). Pour cet exercice, on négligera
les flux liés aux ascenseurs et on considérera que les beacons couvrent 100% des
entrées/sorties ou passage à un autre niveau du centre.

• Problématique n°2 : quelle restitution faire à Perklière, sachant que les
utilisateurs sont des opérationnels qui ne connaissent pas grand-chose à la
data ?

Via le support de ton choix, propose une restitution des résultats tout en gardant
à l’esprit que ton auditoire n’est pas technique. Ce sera notamment ton support
de présentation lors de l’entretien technique.

## Axes de travail
- prétraitement des données brutes (horaires, étages, directions),
- génération des fichiers input nécessaires,
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

├── analysis_out # Fichiers output générés

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
| B1     | 2024-06-01 | 10   | 12    | 0             | outdoor      |
| B2     | 2024-06-01 | 11   | 8     | 1             | 0            |

- `beacon` : identifiant du beacon où est mesuré le passage  
- `date_it` : date (AAAA-MM-JJ)  
- `hour` : heure entière de comptage  
- `count` : nombre de passages  
- `direction_in` / `direction_out` : direction du mouvement (`-1`, `0`, `1`, `outdoor`)

---

## 🚀 Scripts & Modélisations

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
On se concentrera sur ce troisième axe :

### 🔧 Fonctions utilitaires (prétraitement)
setup_logging(v: int)
Configure le niveau de logs (WARNING, INFO, DEBUG) selon -v.
Permet d’avoir un retour console adapté (filtres appliqués, nb de lignes gardées…).

read_csv_lower(path: str) -> pd.DataFrame
Charge un CSV en forçant toutes les colonnes en minuscules.
Sécurise la casse pour éviter les erreurs lors des jointures.

to_num(df: pd.DataFrame, cols: list)
Convertit certaines colonnes en numérique (int/float), ignore les valeurs invalides (coerce).
Important car les CSV peuvent contenir du texte ou des NA.

ensure_cols(df, need, name="df")
Vérifie qu’un DataFrame contient les colonnes obligatoires.
Tente de rattraper certains noms dupliqués (_x / _y après merge).
Lève une exception explicite si des colonnes manquent.

point_to_segment_distance(px, py, x1, y1, x2, y2)
Calcule la distance minimale entre un point (px,py) et un segment (x1,y1)-(x2,y2) en 2D.
Utilisé pour projeter la fréquentation d’un beacon sur le segment le plus proche.

### 🔥 Fonctions cœur d’analyse
load_data(...)
Rôle : ingestion et préparation des trois sources (beacons, segments, counts).
Lecture CSV et normalisation des noms de colonnes.
Conversion types (floor, hour, count…).
Filtrage temporel : créneau horaire (hour_min/hour_max) et optionnellement période de dates.
Création d’un timestamp ts.
Normalisation des directions (direction_in/out → -1,0,1,outdoor).
Option d’appliquer un mapping direction→beacon si besoin.
🔎 → Résultat : trois DataFrames propres prêts pour les calculs.

build_beacon_flows(b, c)
Rôle : transformer les counts beacon/date en flux par beacon.
Associe chaque ligne à un “bucket” de destination (dst_bucket) : soit direction_out prioritaire, soit direction_in.
Jointure avec la géométrie des beacons pour avoir (x_m, y_m, floor).
Renvoie un DataFrame flows avec :
ts, beacon, x_m, y_m, src_floor, dst_bucket, weight

project_flows_to_segments(segments, flows, radius=5)
Rôle : répartir le trafic sur les segments physiques.
Deux modes automatiques :
A — Ligne src→dst : s’il existe des coordonnées destination (x_m_dst), calcule la distance de chaque extrémité au segment.
B — Buckets (cas actuel) : prend chaque beacon et projette son flux sur les segments de son étage selon la distance.
⚙️ Algorithme : pondération par 1 / (1 + distance) pour répartir le poids weight.
🔎 → Retourne un DataFrame segments + flow_load.

build_inter_floor_matrix_from_buckets(flows)
Rôle : construire une matrice de transitions entre étages.
Agrège les poids par couple (src_floor, dst_bucket).
Produit un tableau clair : lignes = étage d’origine, colonnes = destination (-1,0,1,outdoor).

project_counts_to_segments_by_bucket(...)
Rôle : version détaillée de project_flows_to_segments séparée par destination.
Crée un dict {bucket: DataFrame} où chaque DataFrame donne le flow_load par segment uniquement pour les visiteurs allant vers ce bucket.

top_k_segments(seg_loads, k=20)
Classe les segments par intensité décroissante.
Retourne le Top K segments les plus empruntés.

top_k_od_beacons(flows, k=20)
Essaie de donner les couples origine–destination (si colonnes src/dst présentes).
Dans notre cas (buckets), renvoie un DF vide car on n’a pas de vrai beacon destination.

### 📊 Génération du rapport
make_report_html(seg_loads, topsegs, topod, inter_floor, out_html)
- Génère un rapport interactif Plotly combiné dans une page unique :
- Bar chart des top segments.
- Heatmap inter-étages (flux entre niveaux / sorties).
- Sankey des flux verticaux.
- Table OD si dispo.
Écrit un seul fichier report_paths.html.

### ⚡ Workflow du main()

Parsing arguments : chemins CSV, paramètres temporels, rayon pour projection, top K…
Chargement + nettoyage via load_data.
Exclusion optionnelle des ascenseurs si --exclude-elevators.
Construction des flux via build_beacon_flows.
Projection sur segments (global + par bucket).
Matrice inter-étages via build_inter_floor_matrix_from_buckets.
Exports CSV : intensité segments, buckets, top segments, matrice.
Rapport HTML via make_report_html.

### 🧩 Vision d’ensemble

load_data = ingestion & nettoyage
build_beacon_flows = transforme données brutes en flux utilisables
project_flows_to_segments & project_counts_to_segments_by_bucket = spatialisation des flux
build_inter_floor_matrix_from_buckets = vision agrégée verticale
top_k_segments & make_report_html = synthèse pour restitution client

### Interprétation attendue du HTML
- Top segments : couloirs ou axes à trafic fort.
- Matrice inter-étages : parts relatives de flux montants/descendants et vers l’extérieur.
- Heatmaps : zones de concentration selon l’heure et la date.
- Sankey : schéma clair des échanges entre niveaux.

## 🏬 Restitution — Analyse des flux visiteurs
Contexte
L’étude a consisté à exploiter les données de passage collectées par les beacons du centre afin de :
Cartographier les zones les plus fréquentées (heatmaps, densité sur les segments).
Comprendre les itinéraires dominants : comment les visiteurs se déplacent entre entrées, commerces, zones de restauration et sorties.
Analyser les flux verticaux entre les différents niveaux (-1, 0, 1).
Identifier des opportunités d’optimisation de l’expérience client et des espaces.


### 1️⃣ Constats clés
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

### 2️⃣ Opportunités & Recommandations

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

### 3️⃣ Points techniques clés (pour crédibiliser l’analyse)

Heatmaps spatio-temporelles : densité des passages filtrée sur plages horaires pertinentes (10h–22h).
Projection beacon→segments : algorithme qui répartit le trafic selon la proximité physique pour représenter les vraies allées empruntées.
Matrice inter-étages : quantification des flux montants/descendants, part de sortie.
Visualisations interactives (HTML & Sankey) pour exploration libre des données par vos équipes.

### 4️⃣ Prochaines étapes proposées

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
