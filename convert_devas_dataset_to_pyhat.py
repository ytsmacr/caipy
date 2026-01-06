'''
Convert DEVAS-format dataset to PyHAT format.
(c) Cai Ytsma, January 2026
cai@caiconsulting.co.uk

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

Any metadata headers that don't match will be manually input by the user.s
'''

from tkinter import Tk, filedialog
import pandas as pd

Tk().withdraw()  # hide root window

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

# spectra_correct = False
# while not spectra_correct:
#     spectra_path = filedialog.askopenfilename(
#         title="Select spectra file",
#         filetypes=[("CSV files", "*.csv")]
#     )

#     spectra = pd.read_csv(spectra_path)
#     if spectra.columns[0] != 'wave':
#         print("First column of spectra file must be 'wave'. Please check or select a different file.")
#         continue
#     else:
#         spectra_correct = True

# # make sure the spectra and metadata pkeys match
# if not all(spectra.columns[1:] == meta['pkey'].astype(str).values):
#     print("Spectra columns do not match metadata pkeys. Please check your files.")
#     exit(1)

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
meta_headers = meta[['pkey']].copy()
meta_headers['header_type'] = meta_headers['pkey'].str.replace(meta_header_types)

display(meta_headers)