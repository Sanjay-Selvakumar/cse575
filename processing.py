import pandas as pd
import os

# Load the files
mri_df = pd.read_csv("Originals/MRI_Original.csv")
pet_df = pd.read_csv("Originals/PET_Original.csv")

os.makedirs("MRI", exist_ok=True)
os.makedirs("PET", exist_ok=True)

# Drop specified columns
mri_drop_cols = ['PHASE', 'PTID', 'PROCESSDATE', 'SLICE_THICKNESS', 'T2_QC_COMMENT', 'update_stamp']
pet_drop_cols = ['ORIGPROT', 'COLPROT', 'RUNDATE', 'STATUS', 'MODALITY', 'update_stamp']

mri_df = mri_df.drop(columns=[col for col in mri_drop_cols if col in mri_df.columns])
pet_df = pet_df.drop(columns=[col for col in pet_drop_cols if col in pet_df.columns])

# Sort and save cleaned MRI dataset
mri_df["EXAMDATE"] = pd.to_datetime(mri_df["EXAMDATE"], errors='coerce')
mri_df = mri_df.sort_values(by=["RID", "EXAMDATE"])
mri_df = mri_df.drop(columns=["VISCODE"])
mri_df = mri_df.rename(columns={"VISCODE2": "VISCODE"})
mri_df.to_csv("MRI/MRI_Final.csv", index=False)

# Sort and save cleaned PET dataset
pet_df["EXAMDATE"] = pd.to_datetime(pet_df["EXAMDATE"], errors='coerce')
pet_df = pet_df.sort_values(by=["RID", "EXAMDATE"])
pet_df = pet_df.drop(columns=["VISCODE"])
pet_df = pet_df.rename(columns={"VISCODE2": "VISCODE"})
pet_df.to_csv("PET/PET_Final.csv", index=False)

# Load ADNIMERGE data
adnimerge_df = pd.read_csv("Originals/ADNIMERGE.csv")
adnimerge_df["EXAMDATE"] = pd.to_datetime(adnimerge_df["EXAMDATE"], errors='coerce')
adnimerge_df = adnimerge_df.sort_values(by=["RID", "EXAMDATE"])


def process_cohort(rid_list, prefix):
    
    diagnosis_df = adnimerge_df[adnimerge_df['RID'].astype(str).str.strip().isin(rid_list)]
    diagnosis_df = diagnosis_df[['RID', 'VISCODE', 'EXAMDATE', 'DX']]
    diagnosis_df = diagnosis_df.sort_values(by=["RID", "EXAMDATE"])
    diagnosis_df.to_csv(f"{prefix}/Diagnosis_by_Visit.csv", index=False)

    progression_events = []
    non_progression_events = []

    for rid in rid_list:
        group = adnimerge_df[adnimerge_df["RID"].astype(str).str.strip() == rid]
        group_diag = group[["DX", "EXAMDATE", "VISCODE"]].dropna(subset=["DX", "EXAMDATE"])
        if group_diag.empty:
            continue

        first_mci = None
        progressed = False
        for _, row in group_diag.iterrows():
            dx = row["DX"]
            if dx == "MCI" and first_mci is None:
                first_mci = row
            elif dx in ["Dementia", "AD"] and first_mci is not None:
                progression_events.append({
                    "RID": int(rid),
                    "From_VISCODE": first_mci["VISCODE"],
                    "From_DX": "MCI",
                    "From_EXAMDATE": first_mci["EXAMDATE"],
                    "To_VISCODE": row["VISCODE"],
                    "To_DX": dx,
                    "To_EXAMDATE": row["EXAMDATE"],
                    "PROG_TIME": (row["EXAMDATE"] - first_mci["EXAMDATE"]).days / 365
                })
                progressed = True
                break

        if not progressed:
            dxs = group_diag["DX"].tolist()
            if dxs and all(d == dxs[0] for d in dxs):
                visits = adnimerge_df[adnimerge_df["RID"].astype(str).str.strip() == rid].dropna(subset=["DX", "EXAMDATE"])
                min_date = visits["EXAMDATE"].min()
                max_date = visits["EXAMDATE"].max()
                prog_time = (max_date - min_date).days / 365
                if prog_time > 0.0:
                    non_progression_events.append({
                        "RID": int(rid),
                        "DX": dxs[0],
                        "From_VISCODE": visits["VISCODE"].iloc[visits["EXAMDATE"].argmin()],
                        "From_EXAMDATE": min_date,
                        "To_VISCODE": visits["VISCODE"].iloc[visits["EXAMDATE"].argmax()],
                        "To_EXAMDATE": max_date,
                        "PROG_TIME": prog_time
                    })

    valid_progressors = [event["RID"] for event in progression_events]
    pd.DataFrame({"RID": valid_progressors}).astype({"RID": int}).sort_values(by="RID").to_csv(f"{prefix}/Progressors.csv", index=False)
    pd.DataFrame(progression_events).astype({"RID": int}).sort_values(by=["RID", "From_EXAMDATE"]).to_csv(f"{prefix}/Progression_Events.csv", index=False)
    print(f"\n[{prefix}] Subjects who progressed from MCI to Dementia/AD: {len(valid_progressors)}")
    print(f"[{prefix}] Saved progression events to Progression_Events.csv")

    valid_non_progressors = [event["RID"] for event in non_progression_events]
    pd.DataFrame({"RID": valid_non_progressors}).astype({"RID": int}).sort_values(by="RID").to_csv(f"{prefix}/Non_Progressors.csv", index=False)
    pd.DataFrame(non_progression_events).sort_values(by="RID").to_csv(f"{prefix}/Non_Progression_Events.csv", index=False)
    print(f"[{prefix}] Subjects with no progression (stable diagnosis): {len(non_progression_events)}")
    print(f"[{prefix}] Saved non-progression events to Non_Progression_Events.csv")

# Process MRI-based cohort
process_cohort(list(mri_df['RID'].astype(str).str.strip().unique()), "MRI")

# Process PET-based cohort
process_cohort(list(pet_df['RID'].astype(str).str.strip().unique()), "PET")


def find_closest_scan(scan_df, rid, target_date, max_days=180):
    subset = scan_df[scan_df["RID"] == rid].copy()
    if subset.empty:
        return None
    subset["date_diff"] = (subset["EXAMDATE"] - target_date).abs()
    return subset.sort_values("date_diff").head(1)

def generate_features(modality, scan_df):
    prog_rids = pd.read_csv(f"{modality}/Progression_Events.csv")["RID"].unique().tolist()
    non_prog_rids = pd.read_csv(f"{modality}/Non_Progression_Events.csv")["RID"].unique().tolist()
    valid_rids = set(prog_rids + non_prog_rids)
    scan_df = scan_df[scan_df["RID"].isin(valid_rids)].copy()

    prog_rows = []
    non_prog_rows = []

    events_df = pd.read_csv(f"{modality}/Progression_Events.csv", parse_dates=["From_EXAMDATE", "To_EXAMDATE"])
    prog_rids = events_df["RID"].unique()
    prog_scan_df = scan_df[scan_df["RID"].isin(prog_rids)].copy()
    
    for _, row in events_df.iterrows():
        for stage in ["From", "To"]:
            scan = find_closest_scan(prog_scan_df, row["RID"], row[f"{stage}_EXAMDATE"])
            if scan is not None:
                record = scan.drop(columns=["date_diff"]).iloc[0].to_dict()
                record.update({
                    "Stage": stage,
                    "Target_EXAMDATE": row[f"{stage}_EXAMDATE"],
                    "SCAN_OFFSET_YEARS": (record["EXAMDATE"] - row[f"{stage}_EXAMDATE"]).days / 365
                })
                prog_rows.append(record)

    events_df = pd.read_csv(f"{modality}/Non_Progression_Events.csv", parse_dates=["From_EXAMDATE", "To_EXAMDATE"])
    non_prog_rids = events_df["RID"].unique()
    non_prog_scan_df = scan_df[scan_df["RID"].isin(non_prog_rids)].copy()
    
    for _, row in events_df.iterrows():
        for stage in ["From", "To"]:
            scan = find_closest_scan(non_prog_scan_df, row["RID"], row[f"{stage}_EXAMDATE"])
            if scan is not None:
                record = scan.drop(columns=["date_diff"]).iloc[0].to_dict()
                record.update({
                    "Stage": stage,
                    "Target_EXAMDATE": row[f"{stage}_EXAMDATE"],
                    "SCAN_OFFSET_YEARS": (record["EXAMDATE"] - row[f"{stage}_EXAMDATE"]).days / 365
                })
                non_prog_rows.append(record)

    progressors_df = pd.DataFrame(prog_rows)
    progressors_df = progressors_df.sort_values(by=["RID", "EXAMDATE"])

    non_progressors_df = pd.DataFrame(non_prog_rows)
    non_progressors_df = non_progressors_df.sort_values(by=["RID", "EXAMDATE"])

    progressors_df = progressors_df[
        ((progressors_df["Stage"] == "From") & (progressors_df["SCAN_OFFSET_YEARS"] <= 0.5)) |
        ((progressors_df["Stage"] == "To") & (progressors_df["SCAN_OFFSET_YEARS"] >= -0.5))
    ]

    non_progressors_df = non_progressors_df[
        ((non_progressors_df["Stage"] == "From") & (non_progressors_df["SCAN_OFFSET_YEARS"] <= 0.5)) |
        ((non_progressors_df["Stage"] == "To") & (non_progressors_df["SCAN_OFFSET_YEARS"] >= -0.5))
    ]

    progressors_df = progressors_df.groupby("RID").filter(lambda x: x["EXAMDATE"].nunique() == 2)
    non_progressors_df = non_progressors_df.groupby("RID").filter(lambda x: x["EXAMDATE"].nunique() == 2)

    base_cols = ["RID", "VISCODE", "EXAMDATE", "Stage", "Target_EXAMDATE", "SCAN_OFFSET_YEARS"]
    remaining_cols = [col for col in progressors_df.columns if col not in base_cols]
    ordered_cols = base_cols + remaining_cols

    progressors_df = progressors_df[ordered_cols]
    non_progressors_df = non_progressors_df[ordered_cols]

    print("")
    for stage in ["From", "To"]:
        stage_df = progressors_df[progressors_df["Stage"] == stage]
        print(f"[{modality}] Progressors ({stage}) SCAN_OFFSET_YEARS - min: {stage_df['SCAN_OFFSET_YEARS'].min():.2f}, max: {stage_df['SCAN_OFFSET_YEARS'].max():.2f}, mean: {stage_df['SCAN_OFFSET_YEARS'].mean():.2f}, std: {stage_df['SCAN_OFFSET_YEARS'].std():.2f}")
    print(f"[{modality}] Generating Progressors_Features.csv with {len(progressors_df)} rows")
    progressors_df.to_csv(f"{modality}/Progressors_Features.csv", index=False)

    print("")
    for stage in ["From", "To"]:
        stage_df = non_progressors_df[non_progressors_df["Stage"] == stage]
        print(f"[{modality}] Non-Progressors ({stage}) SCAN_OFFSET_YEARS - min: {stage_df['SCAN_OFFSET_YEARS'].min():.2f}, max: {stage_df['SCAN_OFFSET_YEARS'].max():.2f}, mean: {stage_df['SCAN_OFFSET_YEARS'].mean():.2f}, std: {stage_df['SCAN_OFFSET_YEARS'].std():.2f}")
    print(f"[{modality}] Generating Non_Progressors_Features.csv with {len(non_progressors_df)} rows")
    non_progressors_df.to_csv(f"{modality}/Non_Progressors_Features.csv", index=False)

# Generate MRI and PET features
generate_features("MRI", mri_df)
generate_features("PET", pet_df)

print("")