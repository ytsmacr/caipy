'''
Convert DEVAS-format dataset to PyHAT format.
(c) Cai Ytsma, January 2026
cai@caiconsulting.co.uk

** RUN IN caipy ANACONDA ENVIRONMENT (see README.md) **

DEVAS format:
- spectra file = .csv with first column 'wave' and subsequent columns as spectra IDs
- metadata file = .csv with first column 'pkey' column matching spectra IDs

PyHAT format:
one file with:
- the first row is the header type or either 'meta', 'comp', or 'wvl'
- the second row is the name of each header (e.g., pkey, Zr (ppm), 304.508)
- the metadata file comes first, then the transposed spectra file with a row of wavelength intensities per sample

The PyHAT header will use default determinations of:
- meta
    - pkey
    - Sample_Name or Sample Name
    - Laser_Power or Laser Power
    - Atmosphere
- comp
    - all columns with units of:
        - wt %
        - wt%
        - ppm
- wvl
    - all wavelengths from the spectra file

Any metadata headers that don't match will be manually input by the user.
'''

from tkinter import Tk, filedialog
import pandas as pd

Tk().withdraw()  # hide root window

print('\nSelect the DEVAS-format metadata file:')
meta_correct = False
while not meta_correct:
    meta_path = filedialog.askopenfilename(
        title="Select metadata file",
        filetypes=[("CSV files", "*.csv")]
    )

    meta = pd.read_csv(meta_path)

    if meta.columns[0] != 'pkey':
        print("First column of metadata file must be 'pkey'. Please check or select a different file.")
        continue
    else:
        meta_correct = True
print(meta_path)

print('\nSelect the corresponding DEVAS-format spectra file:')
spectra_correct = False
while not spectra_correct:
    spectra_path = filedialog.askopenfilename(
        title="Select spectra file",
        filetypes=[("CSV files", "*.csv")]
    )

    spectra = pd.read_csv(spectra_path)
    if spectra.columns[0] != 'wave':
        print("First column of spectra file must be 'wave'. Please check or select a different file.")
        continue
    else:
        spectra_correct = True
print(spectra_path)

print('\nChoose the location and name of the resulting PyHAT-format file:')
outpath = filedialog.asksaveasfilename(
    title="Save PyHAT output file",
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")]
)
print(outpath)

# make sure the spectra and metadata pkeys match
if not all(spectra.columns[1:] == meta['pkey'].astype(str).values):
    print("Spectra columns do not match metadata pkeys. Please check your files.")
    exit(1)

# determine metadata columns
meta_header_types = {
    'pkey': 'meta',
    'Sample_Name': 'meta',
    'Sample Name': 'meta',
    'Laser_Power': 'meta',
    'Laser Power': 'meta',
    'Atmosphere': 'meta',
    'ppm': 'comp',
    'wt %': 'comp',
    'wt%': 'comp'
}

def replace_if_value_in_cell(s, mapping):
    for k, v in mapping.items():
        if k in s:
            return v
    return None

meta_headers = pd.DataFrame({'header':meta.columns})
meta_headers["header_type"] = meta_headers["header"].astype(str).apply(
    lambda s: replace_if_value_in_cell(s, meta_header_types)
)

# second step to catch the Folds columns
meta_headers["header_type"] = meta_headers.apply(
    lambda row: 'meta' if 'Folds' in row["header"] else row["header_type"],
    axis=1
)

# manually assign any that aren't already mapped
ALLOWED_META_TYPES = ['meta','comp']
unmapped_mask = meta_headers["header_type"].isna()
unmapped = meta_headers.loc[unmapped_mask, "header"].tolist()

if unmapped:
    print("\nUnmapped headers found. Please assign a type from", ", ".join(ALLOWED_META_TYPES))
    print("Press Enter to keep as None (or you can type 'skip') - these columns will be dropped.\n")

    manual_assignments = {}
    for h in unmapped:
        while True:
            user_in = input(f"Type for header '{h}': ").strip().lower()
    
            if user_in in ("", "skip"):
                manual_assignments[h] = None
                break

            if user_in in ALLOWED_META_TYPES:
                manual_assignments[h] = user_in
                break

            print(f"Invalid input: '{user_in}'. Please enter one of {sorted(ALLOWED_META_TYPES)} or 'skip'.")

    # apply manual assignments
    meta_headers["header_type"] = meta_headers.apply(
        lambda row: manual_assignments.get(row["header"], row["header_type"])
        if pd.isna(row["header_type"])
        else row["header_type"],
        axis=1
    )

else:
    print("\nAll headers mapped automatically.")

# SPECTRA
spectra_t = spectra.T.reset_index()
spectra_t.columns = spectra_t.iloc[0]
spectra_t = spectra_t.iloc[1:]

# check the columns are numerical values
bad_cols = pd.to_numeric(spectra_t.columns[1:], errors="coerce").isna()
assert not bad_cols.any(), f"Non-float column names: {spectra_t.columns[1:][bad_cols].tolist()}"

spectra_headers = pd.DataFrame({'header':spectra_t.columns[1:], 'header_type':'wvl'})

# MERGE FILES
pyhat_df = meta.merge(spectra_t, how='inner', left_on='pkey', right_on='wave').drop(columns='wave')
print('\nData merged.')

all_headers = pd.concat([meta_headers, spectra_headers], ignore_index=True)
all_headers_t = all_headers.T
all_headers_t.columns = all_headers_t.loc['header']
all_headers_t = all_headers_t.loc[['header_type','header']]
# add pyhat header
pyhat_df = pd.concat([all_headers_t, pyhat_df])

# EXPORT
pyhat_df.to_csv(outpath, header=False, index=False)
print('\nPyHAT-format file exported.\n')