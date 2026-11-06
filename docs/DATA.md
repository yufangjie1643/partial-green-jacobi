# Data inventory

`data/releases.json` gives the release URLs, exact sizes and SHA-256 checksums. Each tarball contains `data/manifests/<asset>.json`, listing the path, exported checksum, byte count and original-archive checksum of every file. Numeric arrays are unchanged. Workstation prefixes in textual metadata are replaced by `${ARCHIVE_ROOT}` or `${HOME}`.

| Asset | Contents | Principal purpose |
|---|---|---|
| `development-data.tar.gz` | The 172 density/coefficient/warm-start states used in calibration; archived trajectory metadata, scan attempts and analysis tables | Inspect alpha scans and reproduce the fitted exponent table |
| `independent-test-data.tar.gz` | All 90 independent frozen states, 18 reference trajectories, initial fields, paired replay records and diagnostic records | Re-run the independent comparisons and inspect failures |
| `workflow-data.tar.gz` | Study I original/resident/candidate/scaling/scenario/native result records; published checkpoint fields and terminal design densities | Inspect complete workflow timing, per-load certificates and native-solver comparisons |

For Study I original trajectories, the release selects checkpoints 0, 49, 62 and 179, including those used in the native comparison. Intermediate displacement snapshots for the `128^3` scaling run are omitted; terminal densities and all per-step records are included. Deterministic drivers and seeds permit fresh generation. Thus the release is a paper-focused data selection, not an export of every exploratory workspace file.

The development archive also retains scan records from the broader exploratory study, while its array files are limited to the 172 calibration states. Only those states define the present calibration dataset. Retained exploratory records are not additional independent tests.

## Array conventions

* Scalar fields have shape `(n,n,n)` in `(z,y,x)` order.
* `a` is the scalar stiffness multiplier; `rho` or `physical` is the filtered/projected density; `x` is the design density.
* `u0`/`previous` is the warm start, with load-major displacement vectors, three components per node. Study II has six engineering-strain loads; the Study I shear design has three.
* `solution` stores certified displacement when included. A missing key means that field was not saved in that particular archive.
* Load order is the Voigt order `xx, yy, zz, yz, xz, xy`; engineering shear strains have tensor off-diagonals equal to half their Voigt entry.

Read `.npz` with `numpy.load(path, allow_pickle=False)`. Raw JSON records contain thresholds, residuals, iteration counts, statuses and timing components. Failed attempts remain in the dataset; successful subsets of a failing method must not be used for an overall speedup claim.

CPU result reconstruction uses the small source tables already in Git. Large assets are only needed for numerical replay or inspection of density/displacement fields. Downloaded assets can be reused across clean checkouts with `--archive-dir`.
