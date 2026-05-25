-- Add interim tables to jump_metadata.duckdb
-- Creates augmented database in data/processed/ with additional target annotation tables
-- Run from jump_production directory: duckdb data/processed/jump_metadata_augmented.duckdb < scripts/augment_metadata_db.sql

-- ============================================
-- 1. TABLE DEFINITIONS WITH CONSTRAINTS
-- ============================================

-- ChEMBL protein targets table
CREATE TABLE chembl_protein_targets (
    Metadata_JCP2022 VARCHAR,
    Metadata_Uniprot_target VARCHAR,
    FOREIGN KEY (Metadata_JCP2022) REFERENCES compound(Metadata_JCP2022)
);

COMMENT ON TABLE chembl_protein_targets IS 'ChEMBL protein targets for JUMP compounds linking JCP IDs to UniProt targets';
COMMENT ON COLUMN chembl_protein_targets.Metadata_JCP2022 IS 'JUMP Compound ID';
COMMENT ON COLUMN chembl_protein_targets.Metadata_Uniprot_target IS 'Pipe-separated list of UniProt target IDs from ChEMBL';

-- Drug Repurposing Hub annotations table
CREATE TABLE repurposing_hub_annotations (
    Metadata_JCP2022 VARCHAR,
    Metadata_repurposing_name VARCHAR,
    Metadata_repurposing_clinical_phase VARCHAR,
    Metadata_repurposing_moa VARCHAR,
    Metadata_repurposing_target VARCHAR,
    Metadata_repurposing_disease_area VARCHAR,
    Metadata_repurposing_indication VARCHAR,
    FOREIGN KEY (Metadata_JCP2022) REFERENCES compound(Metadata_JCP2022)
);

COMMENT ON TABLE repurposing_hub_annotations IS 'Drug Repurposing Hub annotations for JUMP compounds';
COMMENT ON COLUMN repurposing_hub_annotations.Metadata_JCP2022 IS 'JUMP Compound ID';
COMMENT ON COLUMN repurposing_hub_annotations.Metadata_repurposing_name IS 'Compound name from Drug Repurposing Hub';
COMMENT ON COLUMN repurposing_hub_annotations.Metadata_repurposing_clinical_phase IS 'Clinical development phase';
COMMENT ON COLUMN repurposing_hub_annotations.Metadata_repurposing_moa IS 'Mechanism of action';
COMMENT ON COLUMN repurposing_hub_annotations.Metadata_repurposing_target IS 'Gene targets';
COMMENT ON COLUMN repurposing_hub_annotations.Metadata_repurposing_disease_area IS 'Therapeutic disease area';
COMMENT ON COLUMN repurposing_hub_annotations.Metadata_repurposing_indication IS 'Clinical indication';

-- Chemical probes matched to JUMP compounds - single table ready to join
CREATE TABLE chemical_probes (
    Metadata_JCP2022 VARCHAR,
    Metadata_chmprb_target_genes VARCHAR,  -- pipe-separated gene targets
    Metadata_chmprb_pdid VARCHAR,
    Metadata_chmprb_name VARCHAR,
    Metadata_chmprb_experimental_probe INTEGER,
    Metadata_chmprb_pd_approved INTEGER,
    Metadata_chmprb_available INTEGER,
    Metadata_chmprb_approved_drug INTEGER,
    Metadata_chmprb_protac INTEGER,
    Metadata_chmprb_covalent_binder INTEGER,
    Metadata_chmprb_biased_gpcr_ligand INTEGER,
    Metadata_chmprb_num_targets DOUBLE,
    Metadata_chmprb_qed DOUBLE,
    Metadata_chmprb_has_quality_alerts INTEGER,
    Metadata_chmprb_is_high_quality INTEGER,
    Metadata_chmprb_in_jump INTEGER,
    FOREIGN KEY (Metadata_JCP2022) REFERENCES compound(Metadata_JCP2022)
);

COMMENT ON TABLE chemical_probes IS 'High-quality small-molecule probes from Probes & Drugs Portal matched to JUMP compounds for target validation studies';
COMMENT ON COLUMN chemical_probes.Metadata_JCP2022 IS 'JUMP compound ID for direct joining with compound table';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_target_genes IS 'Pipe-separated list of gene targets from Probes & Drugs Portal';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_pdid IS 'Probes & Drugs Portal unique identifier';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_name IS 'Chemical probe name/identifier';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_experimental_probe IS '1 if compound from published probe characterization experiments with consistent profiling data';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_pd_approved IS '1 if P&D probe-likeness score >70% (based on potency, selectivity, and other criteria)';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_available IS '1 if member of commercial compound sets or in stock at MolPort/mcule';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_approved_drug IS '1 if approved by FDA, EMA or other agencies (from ChEMBL, DrugBank, etc.)';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_protac IS '1 if proteolysis targeting chimera compound';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_covalent_binder IS '1 if forms covalent bond with target protein';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_biased_gpcr_ligand IS '1 if GPCR ligand with functional selectivity towards specific G proteins or β-arrestins';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_num_targets IS 'Number of unique gene targets for this probe';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_qed IS 'Quantitative Estimate of Drug-likeness score (0-1 scale, higher is better)';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_has_quality_alerts IS '1 if has PAINS filters, aggregator alerts, or marked as obsolete/unsuitable';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_is_high_quality IS '1 if meets high-quality criteria: experimental probe, P&D approved, QED>=0.5, no alerts';
COMMENT ON COLUMN chemical_probes.Metadata_chmprb_in_jump IS '1 if probe matched to JUMP compound library via InChIKey';

-- Cell counts per well from jump-profiling-recipe
CREATE TABLE cell_counts (
    Metadata_Source VARCHAR,
    Metadata_Plate VARCHAR,
    Metadata_Well VARCHAR,
    Metadata_Batch VARCHAR,
    Metadata_Count_Cells INTEGER
);

COMMENT ON TABLE cell_counts IS 'Cell counts per well from CellProfiler, used for QC and normalization';
COMMENT ON COLUMN cell_counts.Metadata_Source IS 'JUMP data source identifier';
COMMENT ON COLUMN cell_counts.Metadata_Plate IS 'Plate barcode';
COMMENT ON COLUMN cell_counts.Metadata_Well IS 'Well position (e.g., A01)';
COMMENT ON COLUMN cell_counts.Metadata_Batch IS 'Processing batch identifier';
COMMENT ON COLUMN cell_counts.Metadata_Count_Cells IS 'Number of cells segmented in the well';

-- Chemical probes detailed target information
CREATE TABLE chemical_probes_targets (
    Metadata_chmprb_pdid VARCHAR,
    Metadata_chmprb_probe_name VARCHAR,
    Metadata_chmprb_gene_name VARCHAR,
    Metadata_chmprb_target_name VARCHAR,
    Metadata_chmprb_target_type VARCHAR,
    Metadata_chmprb_moa VARCHAR,
    Metadata_chmprb_activity_biochemical VARCHAR
);

COMMENT ON TABLE chemical_probes_targets IS 'Detailed probe-target relationships from Probes & Drugs Portal';
COMMENT ON COLUMN chemical_probes_targets.Metadata_chmprb_pdid IS 'Probes & Drugs Portal ID';
COMMENT ON COLUMN chemical_probes_targets.Metadata_chmprb_probe_name IS 'Chemical probe name';
COMMENT ON COLUMN chemical_probes_targets.Metadata_chmprb_gene_name IS 'Gene symbol (may include comma-separated complexes)';
COMMENT ON COLUMN chemical_probes_targets.Metadata_chmprb_target_name IS 'Full target protein name';
COMMENT ON COLUMN chemical_probes_targets.Metadata_chmprb_target_type IS 'Type of target (e.g., primary, secondary)';
COMMENT ON COLUMN chemical_probes_targets.Metadata_chmprb_moa IS 'Mechanism of action';
COMMENT ON COLUMN chemical_probes_targets.Metadata_chmprb_activity_biochemical IS 'Biochemical activity measure';

-- Kinase probe sets (KCGS and PKIS) matched to JUMP compounds
-- KCGS: ~295 selective kinase inhibitors from SGC-UNC (Zenodo DOI: 10.5281/zenodo.7830352)
-- PKIS: ~367 kinase inhibitors from GSK via ChEMBL (document CHEMBL2303647)
CREATE TABLE kinase_probes (
    Metadata_JCP2022 VARCHAR,
    Metadata_kinase_probe_set VARCHAR,
    Metadata_kinase_probe_id VARCHAR,
    Metadata_kinase_probe_name VARCHAR,
    Metadata_kinase_probe_inchikey VARCHAR,
    Metadata_kinase_selectivity_s10 DOUBLE,
    Metadata_kinase_is_original_kcgs INTEGER,
    Metadata_kinase_target_info VARCHAR,
    FOREIGN KEY (Metadata_JCP2022) REFERENCES compound(Metadata_JCP2022)
);

COMMENT ON TABLE kinase_probes IS 'Curated kinase probe sets (KCGS and PKIS) matched to JUMP compounds for kinase inhibitor analysis';
COMMENT ON COLUMN kinase_probes.Metadata_JCP2022 IS 'JUMP compound ID for direct joining with compound table';
COMMENT ON COLUMN kinase_probes.Metadata_kinase_probe_set IS 'Probe set name: KCGS (Kinase Chemogenomic Set) or PKIS (Published Kinase Inhibitor Set)';
COMMENT ON COLUMN kinase_probes.Metadata_kinase_probe_id IS 'Original probe identifier (compound name for KCGS, ChEMBL ID for PKIS)';
COMMENT ON COLUMN kinase_probes.Metadata_kinase_probe_name IS 'Probe common name';
COMMENT ON COLUMN kinase_probes.Metadata_kinase_probe_inchikey IS 'Standardized InChIKey from jump_smiles';
COMMENT ON COLUMN kinase_probes.Metadata_kinase_selectivity_s10 IS 'KCGS selectivity metric S10(1µM): fraction of kinases with >90% inhibition (lower = more selective)';
COMMENT ON COLUMN kinase_probes.Metadata_kinase_is_original_kcgs IS '1 if member of original KCGS v1.0 (most validated compounds)';
COMMENT ON COLUMN kinase_probes.Metadata_kinase_target_info IS 'KCGS target annotation from Discoverx/Nanosyn screening';

-- Compound-gene annotations from MOTIVE dataset (aggregated from 8 databases)
-- Source: BioKG, DGIdb, DrugRep, Hetionet, OpenBioLink, OpenTargets, PharmeBiNet, PrimeKG
-- Format: one row per (JCP2022, rel_type, database) with genes pre-aggregated as pipe-separated list
CREATE TABLE motive_annotations (
    Metadata_JCP2022 VARCHAR,
    Metadata_rel_type VARCHAR,
    Metadata_database VARCHAR,
    Metadata_motive_gene VARCHAR,
    Metadata_n_genes INTEGER,
    FOREIGN KEY (Metadata_JCP2022) REFERENCES compound(Metadata_JCP2022)
);

COMMENT ON TABLE motive_annotations IS 'Curated compound-gene interactions from the MOTIVE dataset (8 drug-target databases). One row per (compound, rel_type, database) with pre-aggregated gene lists.';
COMMENT ON COLUMN motive_annotations.Metadata_JCP2022 IS 'JUMP compound ID';
COMMENT ON COLUMN motive_annotations.Metadata_rel_type IS 'Relationship type (targets, binds, upregulates, downregulates, inhibitor, agonist, etc.)';
COMMENT ON COLUMN motive_annotations.Metadata_database IS 'Source database (biokg, dgidb, drugrep, hetionet, openbiolink, opentargets, pharmebinet, primekg)';
COMMENT ON COLUMN motive_annotations.Metadata_motive_gene IS 'Pipe-separated gene symbols for this (compound, rel_type, database) combination';
COMMENT ON COLUMN motive_annotations.Metadata_n_genes IS 'Number of genes in Metadata_motive_gene';

-- EPA ToxCast bioactivity annotations (invitrodb v4.3)
-- Long format: one row per (compound, assay endpoint)
-- Source: US EPA ToxCast program https://doi.org/10.23645/epacomptox.6062623.v14
CREATE TABLE toxcast_annotations (
    Metadata_JCP2022 VARCHAR,
    Metadata_txcst_assay VARCHAR,
    Metadata_txcst_hitcall DOUBLE,
    Metadata_txcst_active INTEGER,
    Metadata_txcst_ac50 DOUBLE,
    Metadata_txcst_acc DOUBLE,
    FOREIGN KEY (Metadata_JCP2022) REFERENCES compound(Metadata_JCP2022)
);

COMMENT ON TABLE toxcast_annotations IS 'EPA ToxCast high-throughput screening bioactivity data (invitrodb v4.3). One row per compound-assay pair with hitcall and AC50 potency values.';
COMMENT ON COLUMN toxcast_annotations.Metadata_JCP2022 IS 'JUMP compound ID';
COMMENT ON COLUMN toxcast_annotations.Metadata_txcst_assay IS 'ToxCast assay endpoint name (e.g., ATG_Ahr_CIS, TOX21_NFkB_BLA_agonist)';
COMMENT ON COLUMN toxcast_annotations.Metadata_txcst_hitcall IS 'Continuous hit call value (0-1 scale, values >= 0.9 considered active)';
COMMENT ON COLUMN toxcast_annotations.Metadata_txcst_active IS 'Binary activity flag (1 if hitcall >= 0.9, 0 otherwise)';
COMMENT ON COLUMN toxcast_annotations.Metadata_txcst_ac50 IS 'Activity concentration at 50% response in µM (null if not active)';
COMMENT ON COLUMN toxcast_annotations.Metadata_txcst_acc IS 'Activity concentration at cutoff in µM';

-- MitoTox mitochondrial toxicity annotations
-- Source: https://www.mitotox.org/ (CC BY-NC 4.0)
CREATE TABLE mitotox_annotations (
    Metadata_JCP2022 VARCHAR,
    Metadata_mitotox_label VARCHAR,
    Metadata_mitotox_toxic INTEGER,
    Metadata_mitotox_mechanisms VARCHAR,
    FOREIGN KEY (Metadata_JCP2022) REFERENCES compound(Metadata_JCP2022)
);

COMMENT ON TABLE mitotox_annotations IS 'MitoTox mitochondrial toxicity labels with functional mechanism categories. Data from https://www.mitotox.org/ (CC BY-NC 4.0).';
COMMENT ON COLUMN mitotox_annotations.Metadata_JCP2022 IS 'JUMP compound ID';
COMMENT ON COLUMN mitotox_annotations.Metadata_mitotox_label IS 'Toxicity classification: "toxic" or "non-toxic"';
COMMENT ON COLUMN mitotox_annotations.Metadata_mitotox_toxic IS '1 if toxic, 0 if non-toxic (binary for filtering)';
COMMENT ON COLUMN mitotox_annotations.Metadata_mitotox_mechanisms IS 'Pipe-separated affected mitochondrial functions: Transmembrane potential, Function of mitochondria, Organization of mitochondria, Movement of mitochondria, Oxidative stress, Mitochondrial DNA, Cell death, Signaling';

-- Physicochemical properties computed from SMILES using RDKit
-- Source: pixi run -e cheminformatics python -m jump_production.processing.compound_properties properties
-- Fingerprints stored separately in data/interim/compound_featurization/morgan_fp.npz
CREATE TABLE compound_properties (
    Metadata_JCP2022 VARCHAR,
    Metadata_MW DOUBLE,
    Metadata_LogP DOUBLE,
    Metadata_TPSA DOUBLE,
    Metadata_HBD INTEGER,
    Metadata_HBA INTEGER,
    Metadata_RotatableBonds INTEGER,
    Metadata_NumRings INTEGER,
    Metadata_NumHeavyAtoms INTEGER,
    Metadata_NumAromaticRings INTEGER,
    Metadata_FractionCSP3 DOUBLE,
    Metadata_Lipinski_Violations INTEGER,
    Metadata_QED DOUBLE,
    Metadata_MurckoScaffold VARCHAR,
    Metadata_HasPAINS BOOLEAN,
    Metadata_ValidMol BOOLEAN,
    FOREIGN KEY (Metadata_JCP2022) REFERENCES compound(Metadata_JCP2022)
);

COMMENT ON TABLE compound_properties IS 'Physicochemical properties computed from SMILES using RDKit. Fingerprints stored separately in compound_featurization/morgan_fp.npz.';
COMMENT ON COLUMN compound_properties.Metadata_JCP2022 IS 'JUMP compound identifier (join key)';
COMMENT ON COLUMN compound_properties.Metadata_MW IS 'Molecular weight in Daltons';
COMMENT ON COLUMN compound_properties.Metadata_LogP IS 'Calculated octanol-water partition coefficient (Wildman-Crippen)';
COMMENT ON COLUMN compound_properties.Metadata_TPSA IS 'Topological polar surface area in Å²';
COMMENT ON COLUMN compound_properties.Metadata_HBD IS 'Number of hydrogen bond donors';
COMMENT ON COLUMN compound_properties.Metadata_HBA IS 'Number of hydrogen bond acceptors';
COMMENT ON COLUMN compound_properties.Metadata_RotatableBonds IS 'Number of rotatable bonds';
COMMENT ON COLUMN compound_properties.Metadata_NumRings IS 'Total number of rings';
COMMENT ON COLUMN compound_properties.Metadata_NumHeavyAtoms IS 'Number of heavy (non-hydrogen) atoms';
COMMENT ON COLUMN compound_properties.Metadata_NumAromaticRings IS 'Number of aromatic rings';
COMMENT ON COLUMN compound_properties.Metadata_FractionCSP3 IS 'Fraction of sp3 carbons (saturation)';
COMMENT ON COLUMN compound_properties.Metadata_Lipinski_Violations IS 'Number of Lipinski Rule-of-5 violations (0-4)';
COMMENT ON COLUMN compound_properties.Metadata_QED IS 'Quantitative Estimate of Drug-likeness (0-1, higher = more drug-like)';
COMMENT ON COLUMN compound_properties.Metadata_MurckoScaffold IS 'Murcko scaffold as SMILES string';
COMMENT ON COLUMN compound_properties.Metadata_HasPAINS IS 'True if compound matches PAINS structural alerts';
COMMENT ON COLUMN compound_properties.Metadata_ValidMol IS 'True if RDKit could parse the SMILES';

-- Toxicity and pharmacokinetic annotations (DILI, DICT, PK)
-- Sources:
-- - DILI: DILIrank v2 (https://pubmed.ncbi.nlm.nih.gov/41005561/)
-- - DICT: DICTrank (https://srijitseal.com/DICTrank_Predictor/datasets.html)
-- - PK: PKSmart (https://srijitseal.com/PKSmart/datasets.html)
CREATE TABLE toxicity_pk_annotations (
    Metadata_JCP2022 VARCHAR,
    -- DILI (Drug-Induced Liver Injury) columns
    Metadata_dili_concern VARCHAR,
    Metadata_dili_positive INTEGER,
    -- DICT (Drug-Induced CardioToxicity) columns
    Metadata_dict_concern VARCHAR,
    Metadata_dict_positive INTEGER,
    -- PK (Pharmacokinetic) columns
    Metadata_pk_vdss_l_kg DOUBLE,
    Metadata_pk_cl_ml_min_kg DOUBLE,
    Metadata_pk_fup DOUBLE,
    Metadata_pk_mrt_h DOUBLE,
    Metadata_pk_thalf_h DOUBLE,
    FOREIGN KEY (Metadata_JCP2022) REFERENCES compound(Metadata_JCP2022)
);

COMMENT ON TABLE toxicity_pk_annotations IS 'Toxicity and pharmacokinetic annotations from FDA DILIrank v2, FDA DICTrank, and PKSmart datasets. Matched to JUMP compounds via InChIKey14 prefix (stereoisomer-tolerant).';
COMMENT ON COLUMN toxicity_pk_annotations.Metadata_JCP2022 IS 'JUMP compound ID';
COMMENT ON COLUMN toxicity_pk_annotations.Metadata_dili_concern IS 'FDA DILIrank v2 concern level based on FDA labeling review: most=withdrawn/boxed warning/severe DILI in warnings, less=mild DILI in warnings or adverse reactions, no=no DILI in labeling, ambiguous=insufficient causality evidence';
COMMENT ON COLUMN toxicity_pk_annotations.Metadata_dili_positive IS '1 if DILI concern is "most" (positive class for binary classification)';
COMMENT ON COLUMN toxicity_pk_annotations.Metadata_dict_concern IS 'FDA DICTrank concern level based on FDA labeling review: most=withdrawn/boxed warning/severe-moderate cardiotoxicity, less=mild in warnings or adverse reactions, no=no cardiotoxicity in labeling, ambiguous=insufficient information';
COMMENT ON COLUMN toxicity_pk_annotations.Metadata_dict_positive IS '1 if DICT concern is "most" (positive class for binary classification)';
COMMENT ON COLUMN toxicity_pk_annotations.Metadata_pk_vdss_l_kg IS 'Volume of distribution at steady state (L/kg) - extent of drug distribution from plasma to tissues (PKSmart)';
COMMENT ON COLUMN toxicity_pk_annotations.Metadata_pk_cl_ml_min_kg IS 'Clearance (mL/min/kg) - rate of drug elimination from the body (PKSmart)';
COMMENT ON COLUMN toxicity_pk_annotations.Metadata_pk_fup IS 'Fraction unbound in plasma (0-1) - proportion of drug not bound to plasma proteins (PKSmart)';
COMMENT ON COLUMN toxicity_pk_annotations.Metadata_pk_mrt_h IS 'Mean residence time (hours) - average time drug molecules spend in the body (PKSmart)';
COMMENT ON COLUMN toxicity_pk_annotations.Metadata_pk_thalf_h IS 'Elimination half-life (hours) - time for plasma concentration to decrease by 50% (PKSmart)';

-- Normalized gene table (union of gene info from ORF and CRISPR)
-- One row per unique NCBI_Gene_ID with consolidated gene metadata
CREATE TABLE gene (
    Metadata_NCBI_Gene_ID VARCHAR PRIMARY KEY,
    Metadata_Symbol VARCHAR,
    Metadata_Gene_Description VARCHAR,
    Metadata_Taxon_ID VARCHAR
);

COMMENT ON TABLE gene IS 'Normalized gene metadata from ORF and CRISPR perturbations. One row per unique NCBI Gene ID.';
COMMENT ON COLUMN gene.Metadata_NCBI_Gene_ID IS 'NCBI Gene ID (primary key for gene annotations)';
COMMENT ON COLUMN gene.Metadata_Symbol IS 'Gene symbol (e.g., TP53, EGFR)';
COMMENT ON COLUMN gene.Metadata_Gene_Description IS 'Full gene name/description';
COMMENT ON COLUMN gene.Metadata_Taxon_ID IS 'NCBI Taxonomy ID (e.g., 9606 for human)';

-- ============================================
-- 2. DATA IMPORT
-- ============================================

-- Import ChEMBL protein targets (filter to valid compound JCP IDs)
INSERT INTO chembl_protein_targets
SELECT c.* FROM read_csv_auto('data/interim/chembl_protein_targets_processed.csv') c
WHERE c.Metadata_JCP2022 IN (SELECT Metadata_JCP2022 FROM compound);

-- Import Drug Repurposing Hub annotations (filter to valid compound JCP IDs)
INSERT INTO repurposing_hub_annotations
SELECT r.* FROM read_csv_auto('data/interim/repurposing_hub_annotations_processed.tsv', delim='\t') r
WHERE r.Metadata_JCP2022 IN (SELECT Metadata_JCP2022 FROM compound);

-- Import chemical probes - now includes all probes with Metadata_chmprb_ prefixed columns
-- Only those with Metadata_JCP2022 will be matched to JUMP compounds
INSERT INTO chemical_probes
SELECT
    Metadata_JCP2022,
    Metadata_chmprb_target_genes,
    Metadata_chmprb_pdid,
    Metadata_chmprb_name,
    Metadata_chmprb_experimental_probe,
    Metadata_chmprb_pd_approved,
    Metadata_chmprb_available,
    Metadata_chmprb_approved_drug,
    Metadata_chmprb_protac,
    Metadata_chmprb_covalent_binder,
    Metadata_chmprb_biased_gpcr_ligand,
    Metadata_chmprb_num_targets,
    Metadata_chmprb_qed,
    Metadata_chmprb_has_quality_alerts,
    Metadata_chmprb_is_high_quality,
    Metadata_chmprb_in_jump
FROM read_csv_auto('data/interim/chemical_probes_processed.csv')
WHERE Metadata_chmprb_in_jump = 1;  -- Only import probes that are in JUMP

-- Import detailed probe-target relationships
INSERT INTO chemical_probes_targets
SELECT
    Metadata_chmprb_pdid,
    Metadata_chmprb_probe_name,
    Metadata_chmprb_gene_name,
    Metadata_chmprb_target_name,
    Metadata_chmprb_target_type,
    Metadata_chmprb_moa,
    Metadata_chmprb_activity_biochemical
FROM read_csv_auto('data/interim/chemical_probes_targets.csv');

-- Import kinase probe sets (KCGS and PKIS)
INSERT INTO kinase_probes
SELECT
    Metadata_JCP2022,
    Metadata_kinase_probe_set,
    Metadata_kinase_probe_id,
    Metadata_kinase_probe_name,
    Metadata_kinase_probe_inchikey,
    Metadata_kinase_selectivity_s10,
    Metadata_kinase_is_original_kcgs,
    Metadata_kinase_target_info
FROM read_csv_auto('data/interim/kinase_probes.csv')
WHERE Metadata_JCP2022 IN (SELECT Metadata_JCP2022 FROM compound);

-- Import cell counts for compound wells
INSERT INTO cell_counts
SELECT
    Metadata_Source,
    Metadata_Plate,
    Metadata_Well,
    Metadata_Batch,
    Metadata_Count_Cells
FROM read_csv_auto('data/external/compound_cell_counts.csv.gz');

-- Import compound-gene annotations from MOTIVE dataset
-- Format: one row per (JCP2022, rel_type, database) with genes already pipe-aggregated
INSERT INTO motive_annotations
SELECT
    Metadata_JCP2022,
    Metadata_rel_type,
    Metadata_database,
    Metadata_motive_gene,
    Metadata_n_genes
FROM read_csv_auto('data/interim/motive_annotations.csv')
WHERE Metadata_JCP2022 IN (SELECT Metadata_JCP2022 FROM compound);

-- Import ToxCast bioactivity annotations (filter to valid JCP IDs)
INSERT INTO toxcast_annotations
SELECT
    Metadata_JCP2022,
    Metadata_txcst_assay,
    Metadata_txcst_hitcall,
    Metadata_txcst_active,
    Metadata_txcst_ac50,
    Metadata_txcst_acc
FROM read_parquet('data/interim/toxcast_processed.parquet')
WHERE Metadata_JCP2022 IN (SELECT Metadata_JCP2022 FROM compound);

-- Import MitoTox annotations (filter to valid JCP IDs)
INSERT INTO mitotox_annotations
SELECT
    Metadata_JCP2022,
    Metadata_mitotox_label,
    Metadata_mitotox_toxic,
    Metadata_mitotox_mechanisms
FROM read_csv_auto('data/interim/mitotox_processed.csv')
WHERE Metadata_JCP2022 IN (SELECT Metadata_JCP2022 FROM compound);

-- Import compound physicochemical properties (computed with RDKit)
-- Source: pixi run -e cheminformatics python -m jump_production.processing.compound_properties properties
INSERT INTO compound_properties
SELECT
    Metadata_JCP2022,
    Metadata_MW,
    Metadata_LogP,
    Metadata_TPSA,
    Metadata_HBD,
    Metadata_HBA,
    Metadata_RotatableBonds,
    Metadata_NumRings,
    Metadata_NumHeavyAtoms,
    Metadata_NumAromaticRings,
    Metadata_FractionCSP3,
    Metadata_Lipinski_Violations,
    Metadata_QED,
    Metadata_MurckoScaffold,
    Metadata_HasPAINS,
    Metadata_ValidMol
FROM read_parquet('data/interim/compound_featurization/properties.parquet')
WHERE Metadata_JCP2022 IN (SELECT Metadata_JCP2022 FROM compound);

-- Import toxicity and pharmacokinetic annotations (DILI, DICT, PK)
-- Source: pixi run -e jump-smiles python -m jump_production.processing.toxicity_pk
INSERT INTO toxicity_pk_annotations
SELECT
    Metadata_JCP2022,
    Metadata_dili_concern,
    Metadata_dili_positive,
    Metadata_dict_concern,
    Metadata_dict_positive,
    Metadata_pk_vdss_l_kg,
    Metadata_pk_cl_ml_min_kg,
    Metadata_pk_fup,
    Metadata_pk_mrt_h,
    Metadata_pk_thalf_h
FROM read_csv_auto('data/interim/toxicity_pk_processed.csv')
WHERE Metadata_JCP2022 IN (SELECT Metadata_JCP2022 FROM compound);

-- Populate gene table from ORF and CRISPR tables
-- ORF has more columns (Gene_Description, Taxon_ID), CRISPR only has Symbol
-- Same gene can have multiple entries (different transcripts), so deduplicate
INSERT INTO gene
WITH orf_genes AS (
    -- Get first occurrence of each gene from ORF (has most metadata)
    SELECT DISTINCT ON (Metadata_NCBI_Gene_ID)
        Metadata_NCBI_Gene_ID,
        Metadata_Symbol,
        Metadata_Gene_Description,
        Metadata_Taxon_ID
    FROM orf
    WHERE Metadata_NCBI_Gene_ID IS NOT NULL
),
crispr_genes AS (
    -- Get genes from CRISPR that aren't in ORF
    SELECT DISTINCT
        Metadata_NCBI_Gene_ID,
        Metadata_Symbol,
        NULL as Metadata_Gene_Description,
        NULL as Metadata_Taxon_ID
    FROM crispr
    WHERE Metadata_NCBI_Gene_ID IS NOT NULL
      AND Metadata_NCBI_Gene_ID NOT IN (SELECT Metadata_NCBI_Gene_ID FROM orf_genes)
)
SELECT * FROM orf_genes
UNION ALL
SELECT * FROM crispr_genes;

-- Create filtered views for MOTIVE targets by database
-- Each view provides targets from a single database - simple filter, no aggregation needed
-- Add more views as needed for different database/rel_type combinations

-- Targets from BioKG (~1,600 compounds)
CREATE VIEW motive_targets_biokg AS
SELECT Metadata_JCP2022,
       Metadata_motive_gene AS Metadata_motive_gene_biokg,
       Metadata_n_genes AS Metadata_n_genes_biokg
FROM motive_annotations
WHERE Metadata_rel_type = 'targets' AND Metadata_database = 'biokg';

COMMENT ON VIEW motive_targets_biokg IS 'MOTIVE gene targets from BioKG database (targets relationship only)';

-- Note: DrugRep view removed - use repurposing_hub_annotations instead (better coverage)

-- Targets from OpenTargets (~800 compounds)
CREATE VIEW motive_targets_opentargets AS
SELECT Metadata_JCP2022,
       Metadata_motive_gene AS Metadata_motive_gene_opentargets,
       Metadata_n_genes AS Metadata_n_genes_opentargets
FROM motive_annotations
WHERE Metadata_rel_type = 'targets' AND Metadata_database = 'opentargets';

COMMENT ON VIEW motive_targets_opentargets IS 'MOTIVE gene targets from OpenTargets database (targets relationship only)';

-- Targets from PrimeKG (~100 compounds)
CREATE VIEW motive_targets_primekg AS
SELECT Metadata_JCP2022,
       Metadata_motive_gene AS Metadata_motive_gene_primekg,
       Metadata_n_genes AS Metadata_n_genes_primekg
FROM motive_annotations
WHERE Metadata_rel_type = 'targets' AND Metadata_database = 'primekg';

COMMENT ON VIEW motive_targets_primekg IS 'MOTIVE gene targets from PrimeKG database (targets relationship only)';

-- ToxCast active assays aggregated per compound (for consistency analysis)
-- Format matches other annotation views: one row per compound, pipe-separated assays
CREATE VIEW toxcast_active_assays AS
SELECT
    Metadata_JCP2022,
    STRING_AGG(Metadata_txcst_assay, '|' ORDER BY Metadata_txcst_assay) as Metadata_txcst_active_assays,
    COUNT(*) as Metadata_txcst_n_active_assays
FROM toxcast_annotations
WHERE Metadata_txcst_active = 1
GROUP BY Metadata_JCP2022;

COMMENT ON VIEW toxcast_active_assays IS 'Aggregated ToxCast active assays per compound for consistency analysis. One row per compound with pipe-separated active assay names.';

-- ============================================
-- 3. VIEWS
-- ============================================

-- Pre-joined compound metadata view for efficient querying
-- Aggregates multi-target compounds with pipe-separated values
--
-- Design: All metadata joins are centralized here rather than in Python code.
-- This allows load_profiles() to do a single "SELECT * FROM compound_metadata"
-- query, with DuckDB optimizing the join plan. To add new metadata columns
-- (e.g., moa, toxcast), update this view definition and rebuild the database.
-- See: src/jump_production/io.py::load_profiles()
CREATE VIEW compound_metadata AS
SELECT
    c.Metadata_JCP2022,
    c.Metadata_InChIKey,
    c.Metadata_InChI,
    c.Metadata_SMILES,
    r.Metadata_repurposing_name,
    r.Metadata_repurposing_target,
    r.Metadata_repurposing_moa,
    r.Metadata_repurposing_disease_area,
    r.Metadata_repurposing_clinical_phase,
    r.Metadata_repurposing_indication,
    ch.Metadata_Uniprot_target,
    tx.Metadata_txcst_active_assays,
    cp.Metadata_chmprb_target_genes,
    mb.Metadata_motive_gene_biokg,
    mo.Metadata_motive_gene_opentargets,
    mp.Metadata_motive_gene_primekg,
    cc.Metadata_median_cell_count,
    -- Physicochemical properties (from compound_properties)
    pr.Metadata_MW,
    pr.Metadata_LogP,
    pr.Metadata_TPSA,
    pr.Metadata_HBD,
    pr.Metadata_HBA,
    pr.Metadata_RotatableBonds,
    pr.Metadata_NumRings,
    pr.Metadata_NumHeavyAtoms,
    pr.Metadata_NumAromaticRings,
    pr.Metadata_FractionCSP3,
    pr.Metadata_Lipinski_Violations,
    pr.Metadata_QED,
    pr.Metadata_MurckoScaffold,
    pr.Metadata_HasPAINS,
    pr.Metadata_ValidMol
FROM compound c
LEFT JOIN (
    -- Aggregate repurposing hub annotations per compound
    SELECT
        Metadata_JCP2022,
        STRING_AGG(DISTINCT Metadata_repurposing_name, '|') as Metadata_repurposing_name,
        STRING_AGG(Metadata_repurposing_target, '|') as Metadata_repurposing_target,
        STRING_AGG(Metadata_repurposing_moa, '|') as Metadata_repurposing_moa,
        STRING_AGG(Metadata_repurposing_disease_area, '|') as Metadata_repurposing_disease_area,
        STRING_AGG(DISTINCT Metadata_repurposing_clinical_phase, '|') as Metadata_repurposing_clinical_phase,
        STRING_AGG(Metadata_repurposing_indication, '|') as Metadata_repurposing_indication
    FROM repurposing_hub_annotations
    GROUP BY Metadata_JCP2022
) r ON c.Metadata_JCP2022 = r.Metadata_JCP2022
LEFT JOIN (
    -- Aggregate ChEMBL targets per compound
    SELECT
        Metadata_JCP2022,
        STRING_AGG(Metadata_Uniprot_target, '|') as Metadata_Uniprot_target
    FROM chembl_protein_targets
    GROUP BY Metadata_JCP2022
) ch ON c.Metadata_JCP2022 = ch.Metadata_JCP2022
LEFT JOIN (
    -- ToxCast active assays (already aggregated in view)
    SELECT Metadata_JCP2022, Metadata_txcst_active_assays
    FROM toxcast_active_assays
) tx ON c.Metadata_JCP2022 = tx.Metadata_JCP2022
LEFT JOIN (
    -- Chemical probes target genes
    SELECT Metadata_JCP2022, Metadata_chmprb_target_genes
    FROM chemical_probes
) cp ON c.Metadata_JCP2022 = cp.Metadata_JCP2022
LEFT JOIN (
    -- MOTIVE targets from BioKG
    SELECT Metadata_JCP2022, Metadata_motive_gene_biokg
    FROM motive_targets_biokg
) mb ON c.Metadata_JCP2022 = mb.Metadata_JCP2022
LEFT JOIN (
    -- MOTIVE targets from OpenTargets
    SELECT Metadata_JCP2022, Metadata_motive_gene_opentargets
    FROM motive_targets_opentargets
) mo ON c.Metadata_JCP2022 = mo.Metadata_JCP2022
LEFT JOIN (
    -- MOTIVE targets from PrimeKG
    SELECT Metadata_JCP2022, Metadata_motive_gene_primekg
    FROM motive_targets_primekg
) mp ON c.Metadata_JCP2022 = mp.Metadata_JCP2022
LEFT JOIN (
    -- Median cell count per compound (via well table)
    SELECT
        w.Metadata_JCP2022,
        CAST(MEDIAN(cc.Metadata_Count_Cells) AS INTEGER) as Metadata_median_cell_count
    FROM cell_counts cc
    JOIN well w ON cc.Metadata_Source = w.Metadata_Source
               AND cc.Metadata_Plate = w.Metadata_Plate
               AND cc.Metadata_Well = w.Metadata_Well
    GROUP BY w.Metadata_JCP2022
) cc ON c.Metadata_JCP2022 = cc.Metadata_JCP2022
LEFT JOIN (
    -- Physicochemical properties (no aggregation needed - one row per compound)
    SELECT *
    FROM compound_properties
) pr ON c.Metadata_JCP2022 = pr.Metadata_JCP2022;

COMMENT ON VIEW compound_metadata IS 'Pre-joined compound metadata with aggregated targets and all physicochemical properties for efficient loading into AnnData';

-- Pre-joined gene metadata view for ORF and CRISPR perturbations
-- Analogous to compound_metadata but for gene perturbations
-- One row per JCP2022 (perturbation), joining normalized gene info
CREATE VIEW gene_metadata AS
SELECT
    p.Metadata_JCP2022,
    p.Metadata_perturbation_modality,
    g.Metadata_NCBI_Gene_ID,
    g.Metadata_Symbol,
    g.Metadata_Gene_Description,
    g.Metadata_Taxon_ID,
    -- ORF-specific columns (NULL for CRISPR)
    o.Metadata_Transcript,
    o.Metadata_Vector,
    o.Metadata_Insert_Length,
    o.Metadata_Prot_Match
    -- Future: pathway annotations, GO terms, etc. will be added here
FROM perturbation p
LEFT JOIN orf o ON p.Metadata_JCP2022 = o.Metadata_JCP2022
LEFT JOIN crispr c ON p.Metadata_JCP2022 = c.Metadata_JCP2022
LEFT JOIN gene g ON COALESCE(o.Metadata_NCBI_Gene_ID, c.Metadata_NCBI_Gene_ID) = g.Metadata_NCBI_Gene_ID
WHERE p.Metadata_perturbation_modality IN ('orf', 'crispr');

COMMENT ON VIEW gene_metadata IS 'Pre-joined gene metadata for ORF and CRISPR perturbations. One row per JCP2022 with gene info and modality-specific columns.';

-- ============================================
-- 4. MICROSCOPE CONFIG AUGMENTATION
-- ============================================

-- Add image dimension columns to microscope_config
-- Data from: scripts/fetch_image_dimensions.sh > data/interim/image_dimensions.csv
ALTER TABLE microscope_config ADD COLUMN IF NOT EXISTS Metadata_Image_Width_Pixels INTEGER;
ALTER TABLE microscope_config ADD COLUMN IF NOT EXISTS Metadata_Image_Height_Pixels INTEGER;

COMMENT ON COLUMN microscope_config.Metadata_Image_Width_Pixels IS 'Image width in pixels (from TIFF headers via fetch_image_dimensions.sh)';
COMMENT ON COLUMN microscope_config.Metadata_Image_Height_Pixels IS 'Image height in pixels (from TIFF headers via fetch_image_dimensions.sh)';

-- Update microscope_config with image dimensions from CSV
UPDATE microscope_config
SET Metadata_Image_Width_Pixels = dims.width,
    Metadata_Image_Height_Pixels = dims.height
FROM read_csv_auto('data/interim/image_dimensions.csv') dims
WHERE microscope_config.Metadata_Source = dims.source;

-- Create view with computed field of view and total well area
CREATE VIEW microscope_config_fov AS
SELECT
    *,
    ROUND(Metadata_Image_Width_Pixels * Metadata_Pixel_Size_Microns, 1) as Metadata_FOV_Width_Microns,
    ROUND(Metadata_Image_Height_Pixels * Metadata_Pixel_Size_Microns, 1) as Metadata_FOV_Height_Microns,
    ROUND(Metadata_Image_Width_Pixels * Metadata_Pixel_Size_Microns *
          Metadata_Image_Height_Pixels * Metadata_Pixel_Size_Microns, 1) as Metadata_FOV_Area_Microns2,
    ROUND(Metadata_Image_Width_Pixels * Metadata_Pixel_Size_Microns *
          Metadata_Image_Height_Pixels * Metadata_Pixel_Size_Microns *
          Metadata_Sites_Per_Well, 1) as Metadata_Well_Area_Microns2
FROM microscope_config;

COMMENT ON VIEW microscope_config_fov IS 'Microscope config with computed field of view and total imaged well area (FOV × sites_per_well)';

-- ============================================
-- 5. INDEXES
-- ============================================

-- Create indexes for query performance
CREATE INDEX idx_chembl_protein_targets_jcp ON chembl_protein_targets(Metadata_JCP2022);
CREATE INDEX idx_repurposing_hub_annotations_jcp ON repurposing_hub_annotations(Metadata_JCP2022);
CREATE INDEX idx_chemical_probes_jcp ON chemical_probes(Metadata_JCP2022);
CREATE INDEX idx_chemical_probes_quality ON chemical_probes(Metadata_chmprb_is_high_quality);
CREATE INDEX idx_chemical_probes_in_jump ON chemical_probes(Metadata_chmprb_in_jump);
CREATE INDEX idx_chemical_probes_targets_pdid ON chemical_probes_targets(Metadata_chmprb_pdid);
CREATE INDEX idx_chemical_probes_targets_gene ON chemical_probes_targets(Metadata_chmprb_gene_name);
CREATE INDEX idx_kinase_probes_jcp ON kinase_probes(Metadata_JCP2022);
CREATE INDEX idx_kinase_probes_set ON kinase_probes(Metadata_kinase_probe_set);
CREATE INDEX idx_cell_counts_well ON cell_counts(Metadata_Source, Metadata_Plate, Metadata_Well);
CREATE INDEX idx_motive_annotations_jcp ON motive_annotations(Metadata_JCP2022);
CREATE INDEX idx_motive_annotations_rel ON motive_annotations(Metadata_rel_type);
CREATE INDEX idx_motive_annotations_db ON motive_annotations(Metadata_database);
CREATE INDEX idx_motive_annotations_rel_db ON motive_annotations(Metadata_rel_type, Metadata_database);
CREATE INDEX idx_toxcast_annotations_jcp ON toxcast_annotations(Metadata_JCP2022);
CREATE INDEX idx_toxcast_annotations_assay ON toxcast_annotations(Metadata_txcst_assay);
CREATE INDEX idx_toxcast_annotations_active ON toxcast_annotations(Metadata_txcst_active);
CREATE INDEX idx_compound_properties_jcp ON compound_properties(Metadata_JCP2022);
CREATE INDEX idx_mitotox_jcp ON mitotox_annotations(Metadata_JCP2022);
CREATE INDEX idx_mitotox_toxic ON mitotox_annotations(Metadata_mitotox_toxic);
CREATE INDEX idx_toxicity_pk_jcp ON toxicity_pk_annotations(Metadata_JCP2022);
CREATE INDEX idx_toxicity_pk_dili ON toxicity_pk_annotations(Metadata_dili_concern);
CREATE INDEX idx_toxicity_pk_dict ON toxicity_pk_annotations(Metadata_dict_concern);
CREATE INDEX idx_gene_symbol ON gene(Metadata_Symbol);

-- ============================================
-- 6. VERIFICATION
-- ============================================

-- Show all tables
SHOW TABLES;

-- Row counts for augmented tables
SELECT 'chembl_protein_targets' as table_name, COUNT(*) as rows FROM chembl_protein_targets
UNION ALL SELECT 'repurposing_hub_annotations', COUNT(*) FROM repurposing_hub_annotations
UNION ALL SELECT 'chemical_probes', COUNT(*) FROM chemical_probes
UNION ALL SELECT 'chemical_probes_targets', COUNT(*) FROM chemical_probes_targets
UNION ALL SELECT 'kinase_probes', COUNT(*) FROM kinase_probes
UNION ALL SELECT 'cell_counts', COUNT(*) FROM cell_counts
UNION ALL SELECT 'motive_annotations', COUNT(*) FROM motive_annotations
UNION ALL SELECT 'toxcast_annotations', COUNT(*) FROM toxcast_annotations
UNION ALL SELECT 'compound_properties', COUNT(*) FROM compound_properties
UNION ALL SELECT 'toxicity_pk_annotations', COUNT(*) FROM toxicity_pk_annotations
UNION ALL SELECT 'mitotox_annotations', COUNT(*) FROM mitotox_annotations
UNION ALL SELECT 'gene', COUNT(*) FROM gene;

-- Compound coverage by annotation source
SELECT
    'chembl_protein_targets' as source,
    COUNT(DISTINCT Metadata_JCP2022) as compounds
FROM chembl_protein_targets
UNION ALL SELECT 'repurposing_hub_annotations', COUNT(DISTINCT Metadata_JCP2022) FROM repurposing_hub_annotations
UNION ALL SELECT 'chemical_probes', COUNT(DISTINCT Metadata_JCP2022) FROM chemical_probes
UNION ALL SELECT 'chemical_probes (high quality)', COUNT(DISTINCT Metadata_JCP2022) FROM chemical_probes WHERE Metadata_chmprb_is_high_quality = 1
UNION ALL SELECT 'kinase_probes', COUNT(DISTINCT Metadata_JCP2022) FROM kinase_probes
UNION ALL SELECT 'kinase_probes (KCGS)', COUNT(DISTINCT Metadata_JCP2022) FROM kinase_probes WHERE Metadata_kinase_probe_set = 'KCGS'
UNION ALL SELECT 'kinase_probes (PKIS)', COUNT(DISTINCT Metadata_JCP2022) FROM kinase_probes WHERE Metadata_kinase_probe_set = 'PKIS'
UNION ALL SELECT 'motive_annotations', COUNT(DISTINCT Metadata_JCP2022) FROM motive_annotations
UNION ALL SELECT 'toxcast_annotations', COUNT(DISTINCT Metadata_JCP2022) FROM toxcast_annotations
UNION ALL SELECT 'toxcast_annotations (active)', COUNT(DISTINCT Metadata_JCP2022) FROM toxcast_annotations WHERE Metadata_txcst_active = 1
UNION ALL SELECT 'compound_properties', COUNT(DISTINCT Metadata_JCP2022) FROM compound_properties
UNION ALL SELECT 'compound_properties (valid)', COUNT(DISTINCT Metadata_JCP2022) FROM compound_properties WHERE Metadata_ValidMol = true
UNION ALL SELECT 'toxicity_pk (any)', COUNT(DISTINCT Metadata_JCP2022) FROM toxicity_pk_annotations
UNION ALL SELECT 'toxicity_pk (DILI)', COUNT(DISTINCT Metadata_JCP2022) FROM toxicity_pk_annotations WHERE Metadata_dili_concern IS NOT NULL
UNION ALL SELECT 'toxicity_pk (DICT)', COUNT(DISTINCT Metadata_JCP2022) FROM toxicity_pk_annotations WHERE Metadata_dict_concern IS NOT NULL
UNION ALL SELECT 'toxicity_pk (PK)', COUNT(DISTINCT Metadata_JCP2022) FROM toxicity_pk_annotations WHERE Metadata_pk_vdss_l_kg IS NOT NULL
UNION ALL SELECT 'mitotox_annotations', COUNT(DISTINCT Metadata_JCP2022) FROM mitotox_annotations
UNION ALL SELECT 'mitotox_annotations (toxic)', COUNT(DISTINCT Metadata_JCP2022) FROM mitotox_annotations WHERE Metadata_mitotox_toxic = 1;

-- MOTIVE annotation details by database and rel_type
SELECT
    Metadata_database,
    Metadata_rel_type,
    COUNT(*) as records,
    COUNT(DISTINCT Metadata_JCP2022) as compounds
FROM motive_annotations
GROUP BY Metadata_database, Metadata_rel_type
ORDER BY compounds DESC
LIMIT 20;

-- MOTIVE views compound coverage
SELECT 'motive_targets_biokg' as view_name, COUNT(*) as compounds FROM motive_targets_biokg
UNION ALL SELECT 'motive_targets_opentargets', COUNT(*) FROM motive_targets_opentargets
UNION ALL SELECT 'motive_targets_primekg', COUNT(*) FROM motive_targets_primekg
UNION ALL SELECT 'toxcast_active_assays', COUNT(*) FROM toxcast_active_assays;

-- Gene metadata coverage by modality
SELECT
    Metadata_perturbation_modality as modality,
    COUNT(*) as perturbations,
    COUNT(DISTINCT Metadata_NCBI_Gene_ID) as unique_genes
FROM gene_metadata
GROUP BY Metadata_perturbation_modality;

-- ToxCast annotation summary (top assays by active compound count)
SELECT
    Metadata_txcst_assay as assay,
    COUNT(DISTINCT Metadata_JCP2022) as compounds_tested,
    SUM(Metadata_txcst_active) as compounds_active,
    ROUND(100.0 * SUM(Metadata_txcst_active) / COUNT(*), 1) as pct_active
FROM toxcast_annotations
GROUP BY Metadata_txcst_assay
ORDER BY compounds_active DESC
LIMIT 20;

-- Toxicity/PK annotation summary
SELECT
    'DILI' as source,
    Metadata_dili_concern as concern_level,
    COUNT(*) as compounds
FROM toxicity_pk_annotations
WHERE Metadata_dili_concern IS NOT NULL
GROUP BY Metadata_dili_concern
UNION ALL
SELECT
    'DICT' as source,
    Metadata_dict_concern as concern_level,
    COUNT(*) as compounds
FROM toxicity_pk_annotations
WHERE Metadata_dict_concern IS NOT NULL
GROUP BY Metadata_dict_concern
ORDER BY source, compounds DESC;

-- PK parameter statistics
SELECT
    'VDss (L/kg)' as parameter,
    COUNT(*) as n_compounds,
    ROUND(MIN(Metadata_pk_vdss_l_kg), 2) as min_val,
    ROUND(AVG(Metadata_pk_vdss_l_kg), 2) as avg_val,
    ROUND(MAX(Metadata_pk_vdss_l_kg), 2) as max_val
FROM toxicity_pk_annotations WHERE Metadata_pk_vdss_l_kg IS NOT NULL
UNION ALL SELECT 'CL (mL/min/kg)', COUNT(*), ROUND(MIN(Metadata_pk_cl_ml_min_kg), 2), ROUND(AVG(Metadata_pk_cl_ml_min_kg), 2), ROUND(MAX(Metadata_pk_cl_ml_min_kg), 2) FROM toxicity_pk_annotations WHERE Metadata_pk_cl_ml_min_kg IS NOT NULL
UNION ALL SELECT 'fup', COUNT(*), ROUND(MIN(Metadata_pk_fup), 3), ROUND(AVG(Metadata_pk_fup), 3), ROUND(MAX(Metadata_pk_fup), 3) FROM toxicity_pk_annotations WHERE Metadata_pk_fup IS NOT NULL
UNION ALL SELECT 'MRT (h)', COUNT(*), ROUND(MIN(Metadata_pk_mrt_h), 2), ROUND(AVG(Metadata_pk_mrt_h), 2), ROUND(MAX(Metadata_pk_mrt_h), 2) FROM toxicity_pk_annotations WHERE Metadata_pk_mrt_h IS NOT NULL
UNION ALL SELECT 't1/2 (h)', COUNT(*), ROUND(MIN(Metadata_pk_thalf_h), 2), ROUND(AVG(Metadata_pk_thalf_h), 2), ROUND(MAX(Metadata_pk_thalf_h), 2) FROM toxicity_pk_annotations WHERE Metadata_pk_thalf_h IS NOT NULL;

-- Microscope config with image dimensions, FOV, and total well area
SELECT
    Metadata_Source,
    Metadata_Microscope_Name,
    Metadata_Sites_Per_Well,
    Metadata_Image_Width_Pixels,
    Metadata_Image_Height_Pixels,
    Metadata_FOV_Area_Microns2,
    Metadata_Well_Area_Microns2
FROM microscope_config_fov
ORDER BY Metadata_Source;
