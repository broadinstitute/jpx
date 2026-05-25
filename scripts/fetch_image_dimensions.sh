#!/usr/bin/env bash
# Fetch image dimensions for each JUMP source using ImageMagick
# Usage: ./scripts/fetch_image_dimensions.sh > data/interim/image_dimensions.csv

set -euo pipefail

# One image path per source (extracted from load_data_with_illum.csv via DuckDB)
declare -A IMAGES=(
    ["source_1"]="s3://cellpainting-gallery/cpg0016-jump/source_1/images/Batch1_20221004/images/UL000109__2022-10-05T06_35_06-Measurement1/Images/r01c02f01p01-ch6sk1fk1fl1.tiff"
    ["source_2"]="s3://cellpainting-gallery/cpg0016-jump/source_2/images/20210607_Batch_2/images/1053601756/1053601756_A01_T0001F001L01A06Z01C06.tif"
    ["source_3"]="s3://cellpainting-gallery/cpg0016-jump/source_3/images/CP59/images/BR5867a3__2022-04-29T01_02_16-Measurement 1/Images/r01c01f01p01-ch3sk1fk1fl1.tiff"
    ["source_4"]="s3://cellpainting-gallery/cpg0016-jump/source_4/images/2021_04_26_Batch1/images/BR00117035__2021-05-02T16_02_51-Measurement1/Images/r01c01f01p01-ch7sk1fk1fl1.tiff"
    ["source_5"]="s3://cellpainting-gallery/cpg0016-jump/source_5/images/JUMPCPE-20210623-Run01_20210624_003152/images/P01_ADMJUM001/P01_ADMJUM001_A01_T0001F001L01A06Z01C06.tif"
    ["source_6"]="s3://cellpainting-gallery/cpg0016-jump/source_6/images/p210824CPU2OS48hw384exp022JUMP/images/110000293081/110000293081_A01_T0001F001L01A04Z01C06.tif"
    ["source_7"]="s3://cellpainting-gallery/cpg0016-jump/source_7/images/20210719_Run1/images/CP1-SC1-01/CP1-SC1-01_A01_T0001F001L01A02Z01C03.tif"
    ["source_8"]="s3://cellpainting-gallery/cpg0016-jump/source_8/images/J1/images/A1170383/Images/HTS_A01_s1_w33EFC51B3-6F57-4273-AFE3-BAA0781C8ACF.tif"
    ["source_9"]="s3://cellpainting-gallery/cpg0016-jump/source_9/images/20210824-Run5/images/GR00003381/Images/r01c01f01p01-ch5sk1fk1fl1.tiff"
    ["source_10"]="s3://cellpainting-gallery/cpg0016-jump/source_10/images/2021_05_31_U2OS_48_hr_run1/images/Dest210531-152149/Dest210531-152149_A01_T0001F001L01A04Z01C06.tif"
    ["source_11"]="s3://cellpainting-gallery/cpg0016-jump/source_11/images/Batch1/images/EC000001__2021-05-27T17_24_29-Measurement1/Images/r01c01f01p01-ch6sk1fk1fl1.tiff"
    ["source_13"]="s3://cellpainting-gallery/cpg0016-jump/source_13/images/20220914_Run1/images/CP-CC9-R1-01/CP-CC9-R1-01_A01_T0001F001L01A02Z01C03.tif"
    ["source_15"]="s3://cellpainting-gallery/cpg0016-jump/source_15/images/2021_12_03_Batch1/images/PEC00001783__2021-12-06T17_22_31-Measurement1/Images/r01c01f01p01-ch7sk1fk1fl1.tiff"
)

echo "source,width,height"

for source in $(echo "${!IMAGES[@]}" | tr ' ' '\n' | sort); do
    img_path="${IMAGES[$source]}"
    # Stream image from S3 and get dimensions with ImageMagick identify
    dims=$(aws s3 cp "$img_path" - --no-sign-request 2>/dev/null | identify -format "%w,%h" - 2>/dev/null) || dims=","
    echo "${source},${dims}"
done
