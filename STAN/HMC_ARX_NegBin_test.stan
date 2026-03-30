// Point1: but de ce code: dans HMC_ARX_v4 la variance est calculée comme sigma_d[d] = fmax(sigma_cluster[...] * exp(...), 1e-4). Ne dépend pas de la magnitude du flux.
// on pense que c'est pour cela que la distribution des erreurs est positivement corrélée à la magnitude du flux pour les flux > 15 000. 

// Point 2: Aussi, le but premier est de remplacer la loi log-normale par une loi Negative Binomiale, qui gère nativement la nature discrète des micro-flux et converge asymptotiqueemnt vers une allure gaussienne qui ressemble à la log-normale précédente pour les macro-flux. 
// l'hétéroscédasticité est aussi native dans la loi NegBin! le point 1 est donc redondant. 

// Point3: Enfin, traiter de manière hiérarchique (comme les autres paramètres) le coefficient beta_lag du Hurdle. 