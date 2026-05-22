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

All contributions must maintain at least **95% code coverage**.

#### Using cargo-tarpaulin (recommended)

Install [cargo-tarpaulin](https://github.com/xd009642/tarpaulin) and run:

```bash
cargo install cargo-tarpaulin
cargo tarpaulin --out Html
```

This generates an HTML coverage report in the project root. Open `tarpaulin-report.html` to view results.

#### CI Integration

Coverage is automatically checked on all pull requests via GitHub Actions. The workflow runs `cargo test` and generates a coverage report using `cargo-tarpaulin`.

#### Alternative: CLion IDE

You can also generate coverage reports using the CLion editor:
1. Load the project in [CLion](https://www.jetbrains.com/clion/).
2. Set up a run/debug configuration that runs the tests.
   ![Create run/debug configuration.](create_config.png)
3. Select **Run '\<config name\>' with Coverage**.
   ![Run with coverage](run_with_coverage.png)
4. Export the coverage report as an [`lcov`](https://github.com/linux-test-project/lcov) file.
   ![Export coverage report](export_coverage_report.png)
5. Run `genhtml` on the `.lcov` file to generate an HTML report.