mkdir tests/output

python RoSCA.py \
    --X "./tests/thdmi/thdmi_feature-table_filtered_samples_features.biom" \
    --Y "./tests/thdmi/nutrients_data_no_cal.csv" \
    --out-directory "tests/thdmi/output" 
    