
def preprocess_medication_table(tables: dict):

    med_cat = tables['m'].groupby('m.medication_drug_classification')['m.patient_id'].apply(list)

    # Union of the two different br/bs and cs/ts
    bDMARD_patients = list(set(med_cat[0] + med_cat[1]))
    cDMARD_patients = list(set(med_cat[2] + med_cat[4]))

    # Intersection of patients having both br/vs and cs/ts medication
    Treatment_switched_patients = list(set(bDMARD_patients) & set(cDMARD_patients))
    print(f"{len(Treatment_switched_patients)} patients received both bDMARD and cDMARD treatment")

    subset_patient = tables['m'][tables['m']['m.patient_id'].isin(Treatment_switched_patients)]

    # Complete the categorical labelling of drugs:
    t = subset_patient
    drugs_per_cat = t.groupby(by='m.medication_drug_classification')['m.medication_drug'].apply(list)
    drug_to_category = dict()

    for i in range(len(drugs_per_cat)):
        list_of_subcat_drug = list(set(drugs_per_cat[i]))
        for j in list_of_subcat_drug:
            drug_to_category[j] = drugs_per_cat.index[i]

    # not labelled drugs
    not_labelled_drugs = list(set(t['m.medication_drug'].to_list()) - set(list(temp_dict.keys())))
    print(not_labelled_drugs)

    print(t['m.medication_drug_classification'].unique())
    # TODO do it without for loop
    for i in range(len(t)):
        if t['m.medication_drug_classification'][i] is None:
            if t['m.medication_drug'][i] not in not_labelled_drugs:
                t['m.medication_drug_classification'][i] = drug_to_category[t['m.medication_drug'][i]]

    print(t['m.medication_drug_classification'].unique())

    # 'Visualization of patient drug history'
    # Select specific patient
    r = t[t['m.patient_id'] ==t['m.patient_id'].unique()[6]]

    #subset of medication table
    m_columns = ['m.medication_drug','m.medication_drug_classification','m.medication_start_date','m.medication_end_date','m.recording_time']
    r = r[m_columns].sort_values(by='m.medication_start_date', ascending=True)



    #### VISULATIZATION tool
    from datetime import date

    today = date.today()
    type(today)

    figure(figsize=(10, 6), dpi=80)

    color = ['black', 'blue', 'green', 'red', 'purple']

    categories = temp.index.to_list()
    data_dict = dict()
    for i in range(len(categories)):
        data_dict[categories[i]] = i

    for i in range(len(r)):
        start = r['m.medication_start_date'][i]
        if start is not None:
            date_start = datetime.strptime(start, '%Y-%m-%d')
        else:
            error('start date is none')

        end = r['m.medication_end_date'][i]
        if end is None:
            date_end = today
        else:
            date_end = datetime.strptime(end, '%Y-%m-%d')

        cat = r['m.medication_drug_classification'][i]

        if (cat is not None):
            y = data_dict[cat]
            plt.plot((date_start, date_end), (y, y), 'o-', color=color[y])

    _ = plt.yticks(range(len(categories)), list(data_dict))
    _ = plt.xticks(rotation='vertical')