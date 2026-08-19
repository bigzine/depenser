# Stratégie de stockage des données de production

## Historique de la décision (évolution du raisonnement)

Deux approches ont été évaluées successivement pour ce projet :

1. **Version initiale : fichier JSON Lines + export planifié.** Les logs
   étaient écrits dans un fichier à l'intérieur du conteneur (stockage
   éphémère de Hugging Face Spaces), avec un export vers le dépôt git via
   un workflow GitHub Actions déclenché quotidiennement.
2. **Version retenue : écriture directe en base PostgreSQL.** Après
   réflexion sur l'exigence d'automatisation du monitoring, cette
   deuxième approche a été retenue à la place.

**Raison du changement** : la version 1 introduisait une fenêtre de perte
de données entre deux exécutions de l'export planifié (jusqu'à 24h de logs
perdus si le conteneur redémarrait entre deux exports), et nécessitait de
maintenir un workflow GitHub Actions dédié uniquement à cette tâche. Écrire
directement en base à chaque requête élimine ce problème à la racine : les
données sont persistées **immédiatement**, sans étape intermédiaire, sans
planification, et sans fenêtre de perte possible.

## Stratégie retenue

1. **Génération et persistance immédiate** : chaque appel à `/predict`
   (succès ou erreur) est journalisé **directement en base de données**
   (PostgreSQL, hébergée sur Neon — tier gratuit) via `api/db.py`, au
   moment même de la requête. Chaque évènement contient : timestamp,
   identifiant client, features en entrée (`inputs`), sortie du modèle
   (probabilité, décision, seuil), latence d'inférence, nombre de features
   manquantes, et détail d'erreur le cas échéant. C'est l'implémentation
   minimale requise par le cahier des charges : *« logs d'appels, inputs,
   outputs, et temps d'exécution »*.

2. **Export pour analyse** : un endpoint dédié et protégé, `GET
   /admin/logs`, permet de récupérer l'intégralité des logs stockés en
   base, au format JSON Lines (pour compatibilité avec les notebooks et le
   dashboard existants). Un second endpoint, `GET /admin/logs/stats`,
   donne un aperçu rapide (nombre d'évènements, taux d'erreur) sans tout
   télécharger. **Contrairement à la version précédente, cet export peut
   être déclenché à tout moment sans risque de perte** : les données sont
   déjà en sécurité en base au moment de l'export, celui-ci ne fait que
   les récupérer pour analyse locale.

3. **Protection** : ces deux endpoints exigent un header `x-admin-key`
   correspondant à la variable d'environnement `ADMIN_API_KEY`, définie
   comme secret sur le Space Hugging Face. La chaîne de connexion à la
   base (`DATABASE_URL`) est elle aussi gérée comme secret, jamais commitée
   dans le code (voir `.gitignore`).

4. **Portabilité du code** : `api/db.py` utilise SQLAlchemy Core plutôt
   qu'un client PostgreSQL spécifique, ce qui permet au même code de
   fonctionner avec SQLite en local/tests (`DATABASE_URL` non définie) et
   PostgreSQL en production (`DATABASE_URL` pointant vers Neon) — sans
   dupliquer la logique.

## Pourquoi PostgreSQL (Neon) plutôt qu'un fichier plat ?

Le raisonnement initial en faveur d'un fichier JSON Lines restait valable
sur plusieurs points (simplicité, zéro coût, zéro dépendance) — mais
l'exigence d'automatisation a fait pencher la balance :

| Critère | Fichier JSON Lines + export planifié | PostgreSQL (Neon) |
|---|---|---|
| Persistance | Différée (jusqu'au prochain export) | **Immédiate, à chaque requête** |
| Fenêtre de perte de données | Jusqu'à la fréquence de l'export (ex. 24h) | **Aucune** |
| Automatisation | Nécessite un workflow planifié dédié | **Native** (écriture directe) |
| Requêtage | Impossible sans tout charger en mémoire | Possible directement en SQL |
| Coût | Gratuit | Gratuit (tier Neon) |
| Complexité de mise en place | Faible | Faible-moyenne (compte Neon, secret de connexion) |
| Dépendances ajoutées | Aucune | `sqlalchemy`, `psycopg2-binary` |

Le tier gratuit de Neon (PostgreSQL serverless, 0.5 Go) suffit largement
au volume de ce projet, et évite les inconvénients du fichier plat sans
sacrifier la simplicité de mise en place — un compte gratuit et une chaîne
de connexion suffisent, sans configuration serveur à gérer.

### Limites assumées (PoC)

- Le tier gratuit de Neon a des limites de stockage et de temps de calcul
  (suffisantes pour ce projet, mais à surveiller si le volume de trafic
  augmentait significativement).
- Pas de chiffrement applicatif des données au-delà de ce que fournit Neon
  par défaut (connexion chiffrée en transit via `sslmode=require`).
- En production réelle avec de vrais clients, une politique de rétention
  et d'anonymisation des données personnelles serait nécessaire — voir les
  limites RGPD déjà documentées dans le notebook d'analyse de drift.

## Adéquation avec les exigences du monitoring

| Exigence | Réponse apportée |
|---|---|
| Logs d'appels | Une ligne en base par appel à `/predict`, horodatée |
| Inputs | Colonne `inputs` (JSON) — dictionnaire complet des features reçues |
| Outputs | Colonnes `probability_default`, `decision`, `threshold_used` |
| Temps d'exécution | Colonne `inference_time_ms`, mesurée côté serveur |
| Accès sécurisé | Clé d'administration + chaîne de connexion gérées en secrets |
| Automatisation / pas de perte de données | Écriture synchrone à chaque requête, sans étape d'export intermédiaire |

## Captures d'écran

### 1. Preuve que les données sont effectivement stockées

_Résultat de `GET /admin/logs/stats` sur l'API déployée, montrant le nombre
d'évènements accumulés._

`[Capture d'écran à insérer ici]`

```
curl https://ediagabate-scoring-credit-api.hf.space/admin/logs/stats \
  -H "x-admin-key: <votre_clé>"
```

### 2. Tableau de bord Neon montrant les données en base

_Console Neon (onglet "Tables" ou "SQL Editor") montrant la table
`predictions` avec des lignes réelles — preuve directe de la persistance
en base, indépendamment de l'API._

`[Capture d'écran à insérer ici]`

### 3. Format des données exportées

_Extrait de `monitoring/production_logs.jsonl` (résultat d'un export via
`/admin/logs`) ouvert dans un éditeur, montrant la structure des
évènements journalisés (inputs, outputs, latence)._

`[Capture d'écran à insérer ici]`

### 4. Endpoints d'export documentés

_Page `/docs` (Swagger) montrant les endpoints `/admin/logs` et
`/admin/logs/stats`, avec l'exigence d'authentification visible._

`[Capture d'écran à insérer ici]`

### 5. Gestion sécurisée des secrets

_Page "Settings → Variables and secrets" du Space Hugging Face montrant
`ADMIN_API_KEY` et `DATABASE_URL` configurés (valeurs masquées)._

`[Capture d'écran à insérer ici]`
