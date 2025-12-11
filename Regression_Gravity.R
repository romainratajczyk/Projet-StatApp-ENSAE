library(haven)
library(tidyverse)
library(lmtest)
library(sandwich)
library(glmnet)
library(car)
library(dplyr)
library(estimatr)

# Cmd+Entrée avec curseur sur une ligne pour exécuter la ligne

file_path <- "ProjetStat/reg_gravity_CEPII.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)


# le modèle de gravité c'est la loi de la gravitation ! donc multiplicatif, prendre le log

# probleme math de base : ln(0) = -infty. 
# Solution A: exclure les zéros
# Solution B: migrants+=1 pour tous. Pb: pourquoi +1 et pas +0.001 ? 
# Solution A gardée. Faire un PPML plus tard 

df_pos <- subset(df, migrantCount > 0) #enlève 100 000 lignes


df_pos$l_migrants <- log(df_pos$migrantCount)
df_pos$l_dist     <- log(df_pos$distw_harmonic) # distance harmonique pondérée de toutes les grandes villes
df_pos$l_pop_o    <- log(df_pos$pop_o)
df_pos$l_pop_d    <- log(df_pos$pop_d)
df_pos$l_gdpcap_o <- log(df_pos$gdp_o) - df_pos$l_pop_o
df_pos$l_gdpcap_d <- log(df_pos$gdp_d) - df_pos$l_pop_d #PIB par tete, sinon PIB et pop , pop va capturer l'effet de PIB. Plus economique de regarder PIB par tete
# attention à la colinéarité, ne pas rajouter gdp_o ou gdp_d
df_pos$l_LA_o <- log(df_pos$LA_o)
df_pos$l_LA_d <- log(df_pos$LA_d)
# intégrer les variables t-2000 linéaire et quadratique, pour capter l'effet de la date. 
# cela résout le probleme de 'year' mélangées. 
# hypothèse: la physique de la migration est constante dans le temps (%), seul le VOLUME change.
df_pos$time_trend <- df_pos$year - 2000
df_pos$time_trend_sq <- df_pos$time_trend^2

model_ols_robust <- lm_robust(l_migrants ~ l_dist + l_pop_o + l_pop_d + l_gdpcap_o + l_gdpcap_d + l_LA_o + l_LA_d + 
                  contig + comlang_off + col_dep_ever + LL_o + LL_d + time_trend + time_trend_sq, 
                data = df_pos)

# 5. Résultats
summary(model_ols_robust)

# on a supposé l'homoscédasticité. On a les bons coeff de beta, mais les p-value sont fausses! 

#-----------------------------REGRESSION SUR DONNEES ISHAGH-------------------------------------------

 #---Importation du fichier----
file_path <- "ProjetStat/data_final/FINAL_GRAVITY_TRAINING_MATRIX.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)
head(df)
str(df)
# On supprime les 14 dernières colonnes car elles sont déjà transformées en Log
df <- df[, 1:(ncol(df) - 14)]

# On convertit les colonnes d'intérêt en Log
should_log <- function(x) {
  is.numeric(x) && 
  length(unique(x)) > 2 && 
  min(x, na.rm = TRUE) > 0
}

df_ready <- df %>%
  dplyr::select(-iso3_o, -iso3_d) %>%
  mutate(across(
    .cols = where(should_log), 
    .fns = log,
    .names = "log_{.col}" # Renomme les colonnes transformées
  )) %>%
  dplyr::select(starts_with("log_"),, everything())

# Derniers ajustements
df <- df_ready[, 1:(ncol(df_ready) - 14)]
df$d_2000 <- df$year - 2000
df$d2_2000 <- (df$year - 2000)**2
df <- subset(df, select = -c(migrantCount,distcap,log_pop_15_64_pct_y,log_pop_15_64_pct_x,year, log_year))
# Vérification du dataframe final
str(df)

#------------Regression sans penalisation------------------
formule_finale <- as.formula(
  paste("log_migrantCount ~", paste(colnames(df[-1]), collapse = " + "))
)

print(formule_finale)

modele_final <- lm_robust(
  formula = formule_finale,
  data = df,
  se_type = "stata" 
)

summary(modele_final)



# glmnet a besoin de matrices, pas de dataframes

# Variable cible (Y) : On garde le flux brut (avec les zéros !)
Y <- as.matrix(df[["log_migrantCount"]]) # Remplace var_cible par "flow" ou ton nom

# Variables explicatives (X) : Tout sauf le flux
# On convertit en matrice numérique
X <- as.matrix(df %>% dplyr::select(-log_migrantCount))

# Lancement du modèle pénalisé
# --- LANCEMENT DU LASSO POISSON ---

# alpha = 1 -> C'est le Lasso (0 serait Ridge)
set.seed(123) # Pour que tes résultats soient fixes

print("Lancement de la validation croisée...")
cv_lasso <- cv.glmnet(
  x = X, 
  y = Y, 
  family = "poisson",  # CRUCIAL pour le modèle de gravité
  alpha = 1,           # Lasso
  standardize = TRUE   # Important pour mettre les variables à la même échelle
)

# --- RÉSULTATS ---

# 1. Le graphique de l'erreur (à mettre dans ton rapport)
plot(cv_lasso)
title("Sélection du Lambda optimal (Lasso Poisson)", line = 2.5)

# 2. Le meilleur lambda
best_lambda <- cv_lasso$lambda.min
print(paste("Meilleur Lambda :", best_lambda))

# 3. Les variables sélectionnées
# On regarde les coefficients pour le meilleur lambda
# Si un coeff est "." ou 0, la variable est rejetée
coefs <- coef(cv_lasso, s = "lambda.min")

# Astuce pour afficher proprement les variables retenues (non nulles)
# On convertit en matrice pour filtrer
coefs_matrix <- as.matrix(coefs)
vars_retenues <- coefs_matrix[coefs_matrix != 0, ]
print("Variables sélectionnées par le modèle :")
print(vars_retenues)


#---------------------------------Regression linéaire finale-------------------------

# Installation des packages nécessaires si besoin
# install.packages(c("estimatr", "texreg"))
library(estimatr) # Pour lm_robust
library(texreg)   # Pour afficher de beaux tableaux (type publication)
library(dplyr)
# --- ÉTAPE 1 : RÉCUPÉRATION AUTOMATIQUE DES VARIABLES DU LASSO ---

# On reprend l'objet 'cv_lasso' de l'étape précédente
# On extrait les coefficients pour le lambda optimal

coeffs <- coef(cv_lasso, s = "lambda.min")

# On récupère les noms des variables dont le coefficient n'est PAS zéro
# as.matrix convertit l'objet sparse matrix en matrice standard pour manipuler
coeffs_mat <- as.matrix(coeffs)
vars_selectionnees <- rownames(coeffs_mat)[coeffs_mat != 0]

# On retire "(Intercept)" de la liste car la fonction lm() l'ajoute automatiquement
vars_selectionnees <- vars_selectionnees[vars_selectionnees != "(Intercept)"]

# Vérification : quelles variables ont survécu ?
print("Variables retenues par le Lasso :")
print(vars_selectionnees)

# --- ÉTAPE 2 : CONSTRUCTION DE LA FORMULE ---

# On crée une formule dynamique : log(flow) ~ var1 + var2 + ...
# Note : Comme on fait un lm_robust, on doit prédire le LOG du flux
formule_finale <- as.formula(
  paste("log_migrantCount ~", paste(vars_selectionnees, collapse = " + "))
)

print(formule_finale)

# --- ÉTAPE 3 : PRÉPARATION DES DONNÉES POUR LM_ROBUST ---

# lm_robust (OLS) ne gère pas les zéros dans la variable cible (car log(0) = infini)
# On doit filtrer les flux > 0 et créer la colonne log(flow)
df_post_lasso <- df %>%
  filter(log_migrantCount  > 0)

# --- ÉTAPE 4 : ENTRAÎNEMENT DU MODÈLE ROBUSTE ---

# se_type = "stata" (ou "HC1") est le standard en économétrie pour l'hétéroscédasticité
modele_final <- lm_robust(
  formula = formule_finale,
  data = df_post_lasso,
  se_type = "stata" 
)

# --- ÉTAPE 5 : AFFICHAGE DU TABLEAU DE RÉSULTATS ---

# Affichage console simple avec p-valeurs et IC
summary(modele_final)

# Affichage PRO (type publication) avec texreg
# Cela te donne les étoiles de significativité, les erreurs types entre parenthèses, etc.
screenreg(
  list(modele_final),
  stars = c(0.01, 0.05, 0.1),
  digits = 3,
  caption = "Modèle de Gravité Post-Lasso (Erreurs Types Robustes)",
  custom.model.names = c("OLS Robuste")
)
