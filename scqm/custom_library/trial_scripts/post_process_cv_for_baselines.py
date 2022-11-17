import pickle
import copy

if __name__ == "__main__":
    with open("/cluster/work/medinfmk/scqm/tmp/final_model/saved_cv.pickle", "rb") as f:
        cv = pickle.load(f)
    cv_post = copy.deepcopy(cv)
    dataset = cv_post.dataset
    dataset.post_process_joint_df()
    dataset.create_dfs()
    dataset.transform_to_numeric_adanet()
    cv_post.dataset = dataset
    with open(
        "/cluster/work/medinfmk/scqm/tmp/final_model/saved_cv_baselines.pickle", "wb"
    ) as f:
        pickle.dump(cv_post, f)
