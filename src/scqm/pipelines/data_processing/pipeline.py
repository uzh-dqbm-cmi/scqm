from kedro.pipeline import Pipeline, node

from .nodes import filter_relevant_columns, remove_unique_value_columns

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(func=remove_unique_value_columns(),
                 inputs="patients",
                 outputs="filtered_patients",
                 name="patients_without_uniquevalue_columns"),
            node(func=remove_unique_value_columns(),
                 inputs="visits",
                 outputs="filtered_visits",
                 name="visits_without_uniquevalue_columns"),
        ]
    )

#[patients,visits,medications,healthissues,modifiednewyorkxrayscore,ratingenscore,sonaras,sonarra,
         #           asas,basfi,basdai,dlqi,euroquol,haq,psada,radai5,sf12,socioeco] scqmdatatables
