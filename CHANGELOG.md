## [2.5.0] - 2026-03-23

### 🚀 Features

- WIP notebook compatible debug viewer
- Demo notebook

### 🐛 Bug Fixes

- Remove absolute machine-specific paths
- Show annotation widget after cropping in notebooks
- Notebook Google Colab
- Type hinting issues

### 🔨 Build

- Pillow >=12.1.1 to fix security issue (CVE-2026-25990)
- Tornado transient dep >=6.5.5 for security (CVE-2026-31958)
- Update rerun version for lz4_flex security issue
- Fonttools>=4.60.2 for CVE-2026-31958
- Add CI for ruff and ty checking

### ⚙️ Miscellaneous Tasks

- Add CODEOWNERS
- Add prek.toml for pre-commit checks
## [2.4.0] - 2026-03-12

### 🚀 Features

- TauluConfig; data class that can be loaded from TOML, with TOML schema
- Default 'akaze' matcher (faster!)
- TableIndexer.highlight_all_cells accepts paths too

### 🐛 Bug Fixes

- DeepConvNet.load type hint
## [2.3.1] - 2026-03-11

### 🚀 Features

- Right_offset is a property of TableGrid
- Add sift matcher and fix ty warnings
- Fix some ty warnings & errors
- Cell cropping with individual margins
- Reduce unnecessary logs
- Disable cutting by default
- Add parallelogram-based grid extrapolation
- Add rerun debug visualization for parallelogram extrapolation
- Enumerate all L-shapes in parallelogram extrapolation
- Prefer regression over parallelogram, use parallelogram as fallback
- Iterative grid smoothing with configurable strength and passes
- Expose smooth_grid parameters from Taulu class
- Change grid smoothing defaults
- Wip notebook annotation
- Ipynb
- First working implementation of notebook functionality
- Prep for notebook view of clickable cell viewer
- Add AKAZE matcher for header alignment
- Add alignment_scale parameter for independent header alignment downscaling

### 🐛 Bug Fixes

- Use correct slope count in region_aware_fit running average
- Align gaussian weights to approx point near image edges
- Preserve seed row corners during cut phase
- Prevent self-inclusion in neighbour lookups via saturating_sub
- Use || instead of && for insufficient-points early return
- Guard against zero derivative in Newton's method
- Use total_cmp for EdgeCandidate ordering and reject NaN
- Prevent usize underflow in find_best_corner_match_flat for out-of-bounds points
- Reject parallelogram predictions outside image bounds
- X-y reversed
- Ty warnings and errors
- Remove wrong assertion

### 📚 Documentation

- Update Taulu documentation
- Add documentation for TableGrid methods and Rust code

### ⚡ Performance

- Shell-based iteration and sqrt weighting for parallelogram extrapolation
## [2.2.0] - 2026-01-15

### 🚀 Features

- Write hashmap and binaryheap benchmarks
- Use binaryheap based edge_queue
- Update Taulu class typing
- Deduplicate code by extracting newton's method into function
- Allow skipping astar (17x speedup)
- Pop_max_if for edge_queue
- Cutting the grid a couple of times
- Separate log functions to their own impl
- Implement grid cutting for less greedy results
- Update python-side to reflect new cutting mechanism

### 🐛 Bug Fixes

- Make example up to date
- Update Taulu class typing

### 📚 Documentation

- Update Rust documentation

### ⚙️ Miscellaneous Tasks

- Increase minimum python version
- Version bump
## [2.1.0] - 2026-01-12

### 🚀 Features

- Use full top row of template as initiation, not just top left!

### 🔨 Build

- Future-proof action (maturin upload is deprecated)
- Generate release notes

### ⚙️ Miscellaneous Tasks

- Version update
## [2.0.7] - 2026-01-12

### 🐛 Bug Fixes

- Allow joining two tables with different row counts

### 📚 Documentation

- Standardize credits section

### 🔨 Build

- Enable build cache for version pushes
- Use macos15, dont use macos13 in build action
- Don't use macos15

### ⚙️ Miscellaneous Tasks

- Update documentation
- Update docs workflow
- Add LICENSE
- Version update
## [2.0.6] - 2025-11-17

### 🐛 Bug Fixes

- Don't modify search region each time...
## [2.0.5] - 2025-11-17

### 🚀 Features

- *(torch)* Randomly distributed negative line points
- *(torch)* Density based point sampling

### ⚙️ Miscellaneous Tasks

- Version updates
## [2.0.4] - 2025-11-14

### 🚀 Features

- Grid smoothing is optional
## [2.0.3] - 2025-11-13

### 🚀 Features

- Version bump
## [2.0.2] - 2025-11-13

### 🐛 Bug Fixes

- Daulu model convolutions use padding
## [2.0.1] - 2025-11-13

### 🚀 Features

- Improve table growing speed 5%
- Remove unnecessary method
- Limit python version for build times
- Remove debug logging from torch/run.py
## [2.0.0] - 2025-10-17

### 🚀 Features

- Allow passing externally filtered image
- Gpu enabled DeepConvNet for kernel filtering
- Export gpu functionality when feature is enabled
- Apply kernel can take opencv image
- Convert heatmap to integer array
- Better gpu feature export
- Rename gpu opt dependency to torch

### 🐛 Bug Fixes

- Pillow import as PIL
- Actually export gpu modules when available
- Allow loading gpu model on cpu and vice versa
- Log.debug instead of printing kernel tile progress

### 📚 Documentation

- Update documentation link
- Add python docstrings for torch feature
- New html docs

### ⚙️ Miscellaneous Tasks

- Format + version update
## [1.2.0] - 2025-10-13

### 🚀 Features

- Allow splitted parameters in Taulu init

### 📚 Documentation

- Update documentation

### ⚙️ Miscellaneous Tasks

- Version v1.2.0
## [1.1.0] - 2025-09-29

### 🚀 Features

- Add rerun dependency for visualization
- Threshold before template matching by default
- Less aggressive astar cost function
- Check signals periodically
- Better rerun paths and fixed radii
- Rerun should connect to external grpc instance
- Extrapolation with local tabular direction

### 🐛 Bug Fixes

- Saturating sub for adding Step to Coord
- Allow for grow_threshold to be zero

### 🚜 Refactor

- Tiny change

### 🔨 Build

- Rerun as optional debug dependency through debug-tools feature
- Uv.lock update

### ⚙️ Miscellaneous Tasks

- Docs pages workflow
- Version v1.1.0
## [1.0.1] - 2025-09-23

### 🐛 Bug Fixes

- Documentation output in docs/
- Smoothing shouldn't introduce None's

### 📚 Documentation

- Favicon

### ⚙️ Miscellaneous Tasks

- Update README
## [1.0.0] - 2025-09-22

### 🚀 Features

- Implement all directions for astar
- Table annealing post processing step
- Initial implementation
- Clean up TableGrower implementation
- *(table_grower)* Internalize loop in rust for performance
- *(table_grower)* Parallellize
- If visual, show grown points
- Regression based table completion
- Take informed step guesses based on neighbours
- Split up table_grower module and cleanup some
- Grid smoothing based on local regression

### 🐛 Bug Fixes

- *(aligner)* Correctly rescale homography
- Fix clippy warnings

### 📚 Documentation

- Pdoc documentation generation

### ⚙️ Miscellaneous Tasks

- Version 1.0.0
## [0.8.2] - 2025-09-12

### 🐛 Bug Fixes

- Don't show images when A* fails
## [0.8.1] - 2025-09-12

### 🚀 Features

- Implement Taulu class with high level functionality

### 🐛 Bug Fixes

- Dont save header with annotation lines

### ⚙️ Miscellaneous Tasks

- Update README and example.py
- Documentation of Taulu class
- Version 0.8.1
## [0.8.0] - 2025-09-10

### 🚀 Features

- Remove unnecessary rust function & python binding
- Add logging decorator and test config
- Refactor grid point detector
- Add logging to aligner and template
- Use A* for growing column lines too

### ⚙️ Miscellaneous Tasks

- Add gif of segmentation
- Update README
## [0.7.5] - 2025-06-04

### 🚀 Features

- Utility functions for table indexer class
## [0.7.4] - 2025-05-28

### 🚀 Features

- Allow perpendicular movement in astar

### 🔨 Build

- Automatic release on github
## [0.7.3] - 2025-05-28

### 🚀 Features

- Template.cell_heights() for variable cell heights

### ⚙️ Miscellaneous Tasks

- Update README
- Update README
- Update README
- Add CITATION.cff file
## [0.7.2] - 2025-05-26

### 🚀 Features

- Allow variable row heights
## [0.7.1] - 2025-05-26

### 🚀 Features

- Draw astar paths on visual=True
## [0.7.0] - 2025-05-26

### 🚀 Features

- Implement astar based grid detection
- Implement astar as a rust python extension module
- Use _core extension module in grid
- Tests only run if files exist
- Add simple benchmark for profiling rust
- Improve build actions

### 🐛 Bug Fixes

- Benchmark calls existing function
- Remove perf_counters from grid.py and fix profile-astar script

### 🔨 Build

- Remove old workflow
- Generate import lib for windows dist

### ⚙️ Miscellaneous Tasks

- Version 0.7.0
## [0.6.9] - 2025-05-21

### 🚀 Features

- GridDetector distance_penalty parameter

### ⚙️ Miscellaneous Tasks

- Version v0.6.9
## [0.6.8] - 2025-05-21

### 🚀 Features

- Sauvola thresholding for header alignment

### ⚙️ Miscellaneous Tasks

- Version v0.6.8
## [0.6.7] - 2025-05-16

### 🚀 Features

- Better tree search

### ⚙️ Miscellaneous Tasks

- Version v0.6.7
## [0.6.6] - 2025-05-13

### 🐛 Bug Fixes

- Out of bounds find_nearest call

### ⚙️ Miscellaneous Tasks

- Version v0.6.6
## [0.6.5] - 2025-05-13

### 🐛 Bug Fixes

- Tree growing algorithm out of bounds error
- Remove 'WHAT' print

### ⚙️ Miscellaneous Tasks

- Version v0.6.5
## [0.6.4] - 2025-05-08

### 🚀 Features

- Tree based search algorithm

### 🐛 Bug Fixes

- Set guassian penalty to default
- Spelling
- Remove print tree statement

### ⚙️ Miscellaneous Tasks

- V0.6.4
- Documentation fixes
- Version v0.6.4
## [0.6.3] - 2025-04-30

### ⚙️ Miscellaneous Tasks

- Add publish step to action
## [0.6.2] - 2025-04-30

### 🐛 Bug Fixes

- *(tablegrid)* Get actual points from saved json

### ⚙️ Miscellaneous Tasks

- Version v0.6.2
## [0.6.1] - 2025-04-29

### 🚀 Features

- Allow removing annotated lines
- Header cropping based on annotation
- TableGrid save and from_saved methods

### 🐛 Bug Fixes

- Crop first, then annotate

### ⚙️ Miscellaneous Tasks

- Version v0.6.1
## [0.6.0] - 2025-04-24

### 🚀 Features

- More general region cropping
## [0.5.0] - 2025-04-22

### 🚀 Features

- Repr implementation for Split
- Join two tablecrosses & rename to grid

### ⚙️ Miscellaneous Tasks

- Version v0.5.0
## [0.4.0] - 2025-04-18

### 🚀 Features

- Add_top_row function

### ⚙️ Miscellaneous Tasks

- Version 0.4.0
## [0.3.0] - 2025-04-18

### 🐛 Bug Fixes

- Remove exit(1) from region test

### 🔨 Build

- Only on v* tags

### ⚙️ Miscellaneous Tasks

- Version 0.3.0
## [0.2.0] - 2025-04-16

### 🚀 Features

- Improve the PageCropper class for production
- Update HeaderAligner for production
- More generic typing hints for aligner
- Update HeaderTemplate for production
- Changes :)
- Document CornerFilter for production
- Update CornerFilter for production
- Change magic number
- More forgiving typing
- Use opencv matchTemplate function for finding table corners
- Add parameter for table cell height
- Rename tabular to taulu
- Add crop_cell functionality
- Add gaussian weighting for nearest point selection
- Remove unnecessary print statement
- Update corner filter test
- Update examples/run.bash
- Add window option for img_util.show
- Add type annotation for crop_split
- Export Split and implement call indirection on it
- Add window arguments for external control

### 🐛 Bug Fixes

- *(cornerfiler)* Fix offset & out of bounds error
- Examples is not a workspace member
- Text_presence score update for sauvola

### 🎨 Styling

- Format code with black

### 🧪 Testing

- Add example image for tests
- Add tests and a 'visual' mark for interaction
- Add tests for HeaderAligner
- Update tests
- Add dummy header template file
- Add test for CornerFilter
- Visualize detected table points

### ⚙️ Miscellaneous Tasks

- Add justfile for convenience
- Justfile paramters for testing
- Add README
- Update README
- Update README
- Update README
- *(README)* Add kernel visualization
- Update README
- Update README
- Update README
- Update README
- Update README
- Update README
- Update README
- Add example images
- README: move examples up
- Update README diagram
- Version 0.2.0
