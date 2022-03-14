## remarks on preprocessing
* all patients are in patients.csv (one unique row per patient) and 2 patients only have medications and no visits (will probably be dropped later)
* in visits, more 100 features related to joints, more than 80 to pain
* das283bsr_score is most available das score (when it is missing, the other das scores are missing at least 95%)

## 

## questions
what tolerance to use to link medications with visits ? -> +- 3 months
radai5 df has many missing uid_num/visit_dates (2500 out of 7000) is it something patients measure at home ? if yes, use recording_time

## preprocessing specific to adaptivenet
* weird categories for dose (use as continuous value instead?)

## todos
* for unit columns, check which have several units and try to homogeneize
* for feature selection, which best techniques to select/aggregate very similar features (e.g. joints, pain)

## meeting 04.03.22
* use all patients, but with different targets as das28. Idea : compare the disease evolutions with an unified score (improvement/worsening)

## notes
# brDMARD = biologic DMARD
