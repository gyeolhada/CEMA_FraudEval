# Output Folder

This folder contains the results generated from executing Step 04 of the CEMA-based adversarial attack pipeline on the fraud dialogue dataset. The results correspond to four victim models, with each CSV file containing the predictions on the test set after applying the selected adversarial examples.

Files included:
- test_with_cema_selected_Bert.csv – Results for the BERT-base victim model.
- test_with_cema_selected_DistillBert.csv – Results for the DistilBERT victim model.
- test_with_cema_selected_Long.csv – Results for the Longformer-110M victim model.
- test_with_cema_selected_Qwen.csv – Results for the Qwen-Turbo victim model.

Notes:

Each CSV file contains the original test samples along with the corresponding predictions after adversarial attacks.

These results can be used to calculate attack success rates, analyze model robustness, and compare the impact of different attack strategies.
