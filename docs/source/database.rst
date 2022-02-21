Database
========

.. _installation:

Data source
------------
The data has been acquired from the SCQM swiss ....

Data content
------------

Data dump has been performed on the 20210701 (snapshot). The data received is structured in an R-list containing following tables:

.. csv-table::
   :header: "Table", "Content"
   :widths: 15, 15

    Patients, "Patient characteristics (most recent available information per patient)"
    Visits, "Clinical measurements by the treating physician (and versioned changes in patient characteristics)"
    Medications, "Treatment information (note that there might be multiple lines per treatment, if e.g. medication doses have been adjusted)"
    Health_issues, "Comorbidities and other health issues"
    Mnyc_scoring, "Modified new york criterium scoring (sacroiliac joint) X-ray scoring"
    Ratingen_scoring, "Rau or Ratingen X-ray scoring"
    Sonar_as, "Sonography: sonography of axSpA-patients"
    Sonar_ra, "Sonography of RA-patients"


Axial Spondyloarthritis (axSpA) is a chronic, inflammatory rheumatic disease that affects the axial skeleton, causing severe pain, stiffness and fatigue. The disease typically starts in early adulthood, a critical period in terms of education and beginning a career path.
Many patients often experience other manifestations such as enthesitis, peripheral arthritis, dactylitis, uveitis, psoriasis, inflammatory bowel disease and osteoporosis.

Rheumatoid arthritis can cause pain, swelling and deformity. As the tissue that lines your joints (synovial membrane) becomes inflamed and thickened, fluid builds up and joints erode and degrade. Rheumatoid arthritis is a chronic inflammatory disorder that can affect more than just your joints.

.. csv-table:: Patient reported outcomes (PRO)
   :header: "Table", "Content"
   :widths: 15, 15

    asas, "`ASAS Health Index <https://www.asas-group.org/instruments/asas-health-index/>`_"
    basfi, "`Bath Ankylosing Spondylitis Functional Index (BASFI) <https://basdai.com/BASFI.php>`_"
    basdai, "`Bath Ankylosing Spondylitis Disease Activity Index <https://basdai.com>`_"
    dlqi, "`DERMATOLOGY LIFE QUALITY INDEX (DLQI) <https://www.bad.org.uk/shared/get-file.ashx?id=1653&itemtype=document>`_"
    euroquol, "The EQ-5D-5L - descriptive system - comprises five dimensions: mobility, self-care, usual activities, pain/discomfort and anxiety/depression"
    haq, "Measures of functional status and quality of life in rheumatoid arthritis: Health Assessment Questionnaire Disability Index (HAQ) `Article  <https://onlinelibrary.wiley.com/doi/full/10.1002/acr.20620>`_"
    psada, "psoriatic arthritis (PsA)"
    radai5, "Rheumatoid Arthritis Disease Activity Index"
    sf_12, "Physical and emotional health"
    socioeco, "Socio-economic status of the people - based on education, working status, pension, smoking ..."

.. note::
    * all tables either link over patient_id or over the numeric visit_uid
    * tables have a prefix that consists of 1-3 letters followed by a ".", these prefixes unambiguously distinct column names in different tables
    * data to the clinical research data can be linked over the p.research_id_usz column
    * patients were selected according to the so-called "loose informed consent criterium", this means there exists somewhere an informed consent, but it never arrived in the SCQM office

Structure medications table research DB3:

General structure:
Whenever a new medication is recorded in the online DB, a new medication object is created. This object contains:
* the name of the medication
* the initial application form
* the initial dosing regimen
* the start date, and, if rmakeecorded, the stop date.

In the research database (RDB) this information (together with the drug classification) is captured in the first line for each medication.
For the start and stop date the variable names are stored_start_date and stored_end_date.
Upon adjusting the dosing regimen or changing the application form a sub object of the medication object is created.
This sub object contains the name of the medication, the application form, the dosing regimen, and the adjustment date (the date at which dosing/application was adjusted).

In RDB’s medications table this sub object is captured with an additional row for the medication id.
For this sub object row we have entries in columns identifying the medication (but not in medication_drug_classification), in dosing columns, and the column regarding the application route.
The adjustment date is added to the medication_start_date column for that row and to the medication_end_date column in the previous row.
Each subsequent adjustment leads to a new medication sub object and a further row in the medications table.
Upon discontinuation of the medication, the stop date is recorded on the medication object level, the reasons for discontinuation, however, on the level of the active (last) sub object.
In RDB’s medications table we see the medication stop date as entry of stored_end_date in the first row (along the entry for stored_start_date) and of medication_end_date in the last row (corresponding to the active sub object at time of discontinuation). The reasons for discontinuation are also shown in the last row.
The medication information that we mostly use is thus to be found in the first row (where stored_start_date and, if recorded, stored_end_date are provided) except the information about discontinuation. Please be aware that the ordering must not always be correct, i.e., the row referring to the medication object must not in all cases be the first row displayed for a given medication.

Information on creation and last change:
Information on when objects (including sub objects) were created are found in the column recording_time of RDB’s medications table.
For capturing information on the last change we have two columns in RDB’s medications table:
Column last_change: Captures the date of the last change done in the medication object. In absence of changes in the medication object it equals the recording time. WARNING: If patients do mySCQM and answer compliance questions, then this alters last_change as well as the compliance questions belong to the medication object!
Column last_medication_change: Captures the date of last i) change in the medication object excluding compliance (!), ii) change in any existing sub object, iii) addition of new sub objects.
Since the changes captured by last_change are not a subset of the changes captured by last_medication_change nor the reverse it does neither hold that last_change >= last_medication_change nor last_change <= last_medication_change.
Last_medication_change was requested by us to allow us to capture the date of the last change/addition done to a medication via the Online DB’s medication interface.

.. note::
    * br/bs not allowed to give it alone <- something to check
    * ts can be prescribe alone
    * steroid mostly used for a short period of time (seriousness of the disease could be inferred

Abbreviations:
R.A. = rheumta

DAS28 score is used to assess R.A
<3.2 low activity
>5.1 high activity


ASAS & ASDAS scores for axSpa and psa

ASDAS:
<1.3 inactive
3.5> high activity

Response to treatment:
R.A:
~3 months intervals between assessment (2-6 months)
if das28>3.7 change does not matter -> considered no response
if delta > 1.2 then the change is significant:q

https://nras.org.uk/resource/the-das28-score/

# number of unique patients per table

Number of Patients in each table
________________________________

.. csv-table:: Patients per table
   :header: "Table", "Number of patients"
   :widths: 15, 15
   :alignment: l, m

    Patients, "1429"
    Visits, "1427"
    Medications, "1174"
    Health issues, ""
    Mnyc scoring, ""
    Ratingen scoring, ""
    Sonar as, ""
    Sonar ra, ""
    asas, ""
    basfi, ""
    basdai, ""
    dlqi, ""
    euroquol, ""
    haq, ""
    psada, ""
    radai5, ""
    sf_12, ""
    socioeco, ""
