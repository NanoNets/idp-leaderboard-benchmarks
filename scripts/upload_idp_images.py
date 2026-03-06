from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./idp_images",
    repo_id="shhdwi/idp-leaderboard-results",
    repo_type="dataset",
    path_in_repo="images/idp",
    commit_message="Replace IDP images with correct Nanobench source images",
)
print("Done")
