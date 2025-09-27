# seiki
Use Case Porta di Roma

Effectuer la commande 

python heatmap_evolutive.py \
  --beacons beacons_world_template.csv \
  --segments segments_template.csv \
  --counts porta_di_roma_counts.csv \
  --out heatmap_floor0.gif \
  --fps 6 \
  --bins 120 \
  --smooth-kernel 3 \
  --smooth-passes 1 \
  --floor 0


--floor all pour générer un rendu par étage (suffixe _floorX).

--out accepte .gif ou .mp4 (MP4 tentée d’abord, fallback en GIF).

--bins contrôle la définition de la grille.

--smooth-kernel / --smooth-passes appliquent un lissage léger (sans SciPy).

Ce que fait le script

Joint counts ↔ beacons (sur beacon, floor), crée un timestamp ts = date_it + hour.

Agrège et projette les intensités sur une grille 2D (mètres), par timestamp et étage.

Normalise l’échelle de couleurs sur toute l’animation (comparabilité temporelle).

Dessine les segments (segments_template.csv) en overlay pour situer les flux.

Exporte l’animation finale.


Notes techniques
- Le filtrage horaire est appliqué juste après le parsing, puis les timestamps sont reconstruits (ts = date_it + hour).
- Les intensités par segment sont calculées en pré-associant les beacons proches de chaque segment (distance point-segment ≤ segment-radius) et en moyennant les counts pondérés par 1/(1+d).
- Les échelles de couleur et l’épaisseur des segments sont normalisées globalement sur toute l’animation pour une lecture cohérente dans le temps.
