# Contributing to LCSO

## Filing issues

When filing an issue, make sure to answer these five questions:

1. What version of Rust are you using (`rustc --version`)?
2. What did you do?
3. What did you expect to see?
4. What did you see instead?

## Report a Bug

Open an issue. Please include descriptions of the following:
- Observations
- Expectations
- Steps to reproduce

## Contributing code

In general, this project follows Rust project conventions. Please make sure
you've linted, formatted, and run your tests before submitting a patch.

## Contribute a Bug Fix

- Report the bug first
- Create a pull request for the fix

## Suggest a New Feature

- Create a new issue to start a discussion around new topic. Label the issue as `new-feature`

## Developer guidelines

### Linting and formatting
To lint the code, run `cargo clippy`. For more information about `clippy`
and its various options, see [here](https://github.com/rust-lang/rust-clippy).

To format the code, run `cargo fmt`. For more information about the various
rules, see [here](https://github.com/rust-lang/rustfmt).

### Generating documentation
To generate the documentation for the code, run `cargo doc`. Once the
documentation has been generated, you can view and poke around the
documentation by opening `$PROJECT_ROOT/target/doc/lcso/index.html`.

### Code coverage
This project requires at least **95%** code coverage. Coverage is enforced
automatically in CI using [`cargo-tarpaulin`](https://github.com/xd009642/tarpaulin)
with the `--fail-under 95` flag.

To run coverage locally:
1. Install `cargo-tarpaulin`:
   ```sh
   cargo install cargo-tarpaulin
   ```
2. Run coverage with the enforcement threshold:
   ```sh
   cargo tarpaulin --fail-under 95 -- --test-threads=1
   ```
3. To generate an HTML report for detailed inspection:
   ```sh
   cargo tarpaulin --out html --fail-under 95 -- --test-threads=1
   open tarpaulin-report.html
   ```