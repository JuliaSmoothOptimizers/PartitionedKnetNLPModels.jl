
Execution of
```
ges_plse = PK.PLS(pknet_nlp_plse; max_time, printing, α)
```
for minibatch of size:
- 20 : 4,.. seconds by iteration
  3000 iteration needed for an epoch
  - 12 000 sec by epoch
  - en 4h30 de calcul:
    - 5728 iteration pour minibtach de taille 20
    - 4107 iteration pour minibtach de taille 100
- PSNET_SPLDP : Adam  
  - acc 2 epochs: 0.8409 ; 0.8703; 0.8375; 0.7715; 0.0000
  - acc 3 epochs: 0.9168 ; 0.8737; 0.8291; 0.7970; 0.8173
  - acc 4 epochs: 0.9342 ; 0.9156; 0.8871; 0.0000; 
  - acc 5 epochs: 0.0000 ; 0.0000; 0.9079; 0.0000; 

- erreur dans le script de résultat (3e colonne).
  - seul minbatch a 100 observable (le minibtach taille 20 et parti du résultat minibtach taille 100).
  Là où Adam a trouvé 0.8375, 0.8291, 0.8871, 0.9079
    PLSE à partir de l'entraînement des deux première époques à trouvé (0.8375, 0.8291,) 0.9203, 0.9516

- 4 eme colonne:
  - PLSE sur 100 minibtach, basé sur les poids étant arrivés à 0.797
  - observation d'un 0.91 à l'itération 24, 0.9225à la 65ème
  - 

- minbatch100, PLSE seul, permiers signe d'augmentation au bout de 2 epoques et pas avant