# Data

The full reconstructed boiling HTC database is not redistributed in this repository because it was reconstructed from multiple previously published experimental studies and may be subject to the original publication licenses.

To run the code, prepare an Excel workbook following the schema used in the JSON configuration files. The required input columns are organized into three domains:

- Thermal-property inputs: `density_l`, `density_v`, `dynamic viscosity_l`, `dynamic viscosity_v`, `thermal conductivity_l`, `thermal conductivity_v`, `specific heat_l`, `specific heat_v`, `Surface tension`, `Latent heat`
- Flow inputs reconstructed from raw columns: `G`, `Xmean`, `qw`, `density_l`, `density_v`
- PHE geometry inputs: `corrugation angle`, `corrugation depth`, `pitch`, `effective hydraulic diameter`
- Target: `h`

The workbook should contain refrigerant-specific worksheets whose names match the `groups_by_pressure` entries in the JSON configuration files.
