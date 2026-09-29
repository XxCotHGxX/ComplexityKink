# Croissant with Responsible-AI fields: build, merge, validate

NeurIPS 2026 E&D requires a public dataset by camera-ready plus a validated
Croissant file that includes the minimal RAI properties `rai:dataLimitations`,
`rai:dataBiases`, `rai:personalSensitiveInformation`, `rai:dataUseCases`,
`rai:dataSocialImpact`, `rai:hasSyntheticData`, `prov:wasDerivedFrom`, and
`prov:wasGeneratedBy`. Hugging Face and Harvard Dataverse both auto-generate a
core Croissant file but do not add these RAI fields, so they are merged in
afterwards.

Files in this directory:

| File | Purpose |
|---|---|
| `README.md` | Hugging Face dataset card (copied to the release root by the build). |
| `croissant_rai_fields.json` | Draft RAI and provenance properties. Resolve every `TODO(author):` first. |
| `merge_croissant_rai.py` | Standard-library script that merges the RAI file into any Croissant JSON-LD. |
| `LICENSE_NOTES.md` | Source license and per-model terms to review before choosing a license. |
| `build_report.json` | Sanity checks and sanitization scan from the last build. |

## 0. Build the release locally

```powershell
cd D:\ProgD\ComplexityKinkResearch\.publish_worktrees\camera-ready
D:\ProgD\ComplexityKinkResearch\.venv\Scripts\python.exe scripts\build_public_release.py --data-root D:\ProgD\ComplexityKinkResearch
```

This writes `D:\ProgD\ComplexityKinkResearch\data\public_release\` (Parquet
configs, `README.md`, `docs\`, and a locally built `croissant.json` that already
contains the RAI fields), refreshes the field tables in `release\README.md`, and
writes `release\build_report.json`. Rebuild after editing `README.md` or
`croissant_rai_fields.json`.

## 1. Resolve placeholders before upload

- `croissant_rai_fields.json`: every `TODO(author):` string.
- `README.md`: the Hugging Face repo id in the quick start, license, title,
  author list, and the HTML `TODO(author)` comments.
- `scripts/build_public_release.py`, `write_croissant()`: `url`, `creator`,
  `citeAs`, `license`, and `datePublished` for the local Croissant file.

List what remains:

```powershell
Select-String -Path release\*.json, release\*.md -Pattern "TODO\(author\)"
```

## 2. Upload (done by the authors, not by the build)

- **Hugging Face**: upload the contents of `data\public_release\` to the root of a
  new public dataset repository (for example with `hf upload <org>/<name>
  D:\ProgD\ComplexityKinkResearch\data\public_release . --repo-type dataset`).
  `README.md` must be at the repository root; its `configs:` block maps each
  config to `<config>/test-*.parquet`.
- **Harvard Dataverse**: files must stay below 2.5 GB each (the largest file is
  about 33 MB). Dataverse flattens folders unless directory labels are kept;
  uploading a single `.zip` of `public_release\` lets Dataverse unpack it and
  keep each file's folder as its directory label. Parquet files are stored as
  is (they are not ingested as tabular data).

## 3. Download the platform-generated Croissant file

- **Hugging Face**: on the dataset page, click **Croissant** under the dataset
  name and choose **Download Croissant metadata**, or fetch
  `https://huggingface.co/api/datasets/<org>/<name>/croissant` and save it as
  `hf_croissant.json`.
- **Harvard Dataverse**: open the dataset, click **Metadata**, then **Export
  Metadata**, and select **Croissant**. The dataset must be published or shared
  through a preview link. Save it as `dataverse_croissant.json`.

## 4. Merge the RAI fields

```powershell
D:\ProgD\ComplexityKinkResearch\.venv\Scripts\python.exe release\merge_croissant_rai.py `
    --croissant hf_croissant.json `
    --out complexity_kink_hf_croissant_rai.json
```

The script adds the `rai`, `prov`, and `dct` prefixes to `@context`, copies every
RAI and provenance property into the dataset node, keeps any property the
platform already set unless `--overwrite` is given, checks that all eight
NeurIPS-required properties are present, and refuses to write while any
`TODO(author):` placeholder remains (`--allow-todo` for dry runs). Optional:
`--add-rai-conformance` also declares `http://mlcommons.org/croissant/RAI/1.0`
in `conformsTo`. Repeat with `dataverse_croissant.json` for the Dataverse copy.

## 5. Validate

Install `mlcroissant` in a separate virtual environment (not the project
`.venv` and not the global Python):

```powershell
py -3.13 -m venv $env:TEMP\croissant-venv
& $env:TEMP\croissant-venv\Scripts\python.exe -m pip install mlcroissant pandas pyarrow
& $env:TEMP\croissant-venv\Scripts\mlcroissant.exe validate --jsonld complexity_kink_hf_croissant_rai.json
```

`merge_croissant_rai.py --validate` runs the same check when it is executed with
that environment's Python. Then check the file with the online Croissant checker
linked from the NeurIPS hosting guidelines
(<https://huggingface.co/spaces/JoaquinVanschoren/croissant-checker>), which also
tests that the data files are reachable, and optionally review the RAI fields in
the online RAI editor linked from the same page.

## 6. Local Croissant for the Dataverse copy

The build writes `data\public_release\croissant.json`: a Croissant 1.1 file with
one `FileObject` (relative `contentUrl`, SHA-256) and one `RecordSet` per config,
typed fields with descriptions, keys and cross-config references, and the RAI
fields merged in. It was validated locally with `mlcroissant` 1.1.0 (0 errors, 0
warnings; the first records of all 15 record sets load). Because its
`contentUrl` values are relative paths, it validates against the local copy. For
the published Dataverse record, either use the Dataverse export merged in step 4
or replace each `contentUrl` with the Dataverse file-access URL
(`https://dataverse.harvard.edu/api/access/datafile/<file id>`) and revalidate.

## 7. Submit

Upload the merged and validated Croissant file(s) with the camera-ready
materials as the track instructs, and put the Hugging Face URL and the
Dataverse DOI in the paper and in `rai:dataReleaseMaintenancePlan`.
