# Public Streamlit deployment

CalibraNeo is prepared for Streamlit Community Cloud deployment from GitHub.

## Deploy

1. Sign in at <https://share.streamlit.io/>.
2. Select the `Benjaminotuya1/CalibraNeo` repository.
3. Select branch `main`.
4. Set the main file to `app.py`.
5. Click **Deploy**.

The application requires the tracked public demo artifacts:

- `models/calibraneo_stage1_bundle.pkl`
- `data_processed/MASTER_CALIBRANEO_DATASET.csv`
- `data_processed/nature_2317_9tools_benchmark.csv`

The large RNA context model is intentionally not required for startup. Raw
archives, manuscripts, PDFs, and patient files are not part of the deployed
application.

## Public-use boundary

The app is a computational research demonstration. Do not upload identifiable
patient data. Uploaded files are processed by the running application and
should be treated according to the hosting provider's data policy. The app
does not provide clinical recommendations, therapeutic decisions, or validated
predictions of tumour killing or cytokine safety.

## Troubleshooting

If deployment reports a missing model or feature table, confirm that the latest
`main` commit has finished building and that the deployment is using branch
`main`. The app shows a startup error rather than silently switching to an
unsupported model.
