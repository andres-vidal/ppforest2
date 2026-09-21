# ppforest2 monorepo split — record

The split described here is done. This file is kept as the record of what was
decided and where things ended up.

## Result

- **[ppforest2-core](https://github.com/andres-vidal/ppforest2-core)** — this
  repository's tree minus `bindings/`, with no restructuring, so every path
  still matches the one here.
- **[ppforest2-r](https://github.com/andres-vidal/ppforest2-r)** — the R package
  at the repository root, which is what rOpenSci reviews and what would be
  transferred on acceptance. The CRAN package name stays `ppforest2`.
- This repository is the record of the project up to the split, including the
  history and the `v0.1.0`-`v0.1.2` releases.

## Coupling

`ppforest2-r` commits a pruned copy of the core under `src/core/` and the golden
files under `inst/golden/`, refreshed by
`make vendor-core CORE=../ppforest2-core REF=<ref>` and pinned by its
`CORE_VERSION` file. `make vendor-deps` refreshes the vendored nlohmann/json and
pcg headers, reading their pinned versions out of the core repository's
`core/Dependencies.cmake` so the two cannot drift.

`configure` no longer stages anything: it checks `src/core` is present and
generates the `OBJECTS` list. The CRAN tarball was already self-contained, so
the split did not change what CRAN receives.

## Decisions

- **Git history was not carried over.** Both repositories start from four
  snapshot commits — the `v0.1.0`, `v0.1.1` and `v0.1.2` releases plus the tree
  at `429f7bcc` — with the three tags recreated at their original dates. The
  history lives here.
- **This repository is archived, not renamed.** Archiving keeps
  `andres-vidal/ppforest2` resolving, which a rename would leave dependent on
  redirects that break if the name is ever reused.
- **The core repository was not restructured**, so paths line up with this one
  for history lookups.
- **No `ppforest2-site` repository.** The landing page is 52 lines whose two
  cards link the C++ and R documentation; it stays in `ppforest2-core` and links
  out. Revisit if a second binding ever exists, as a user Pages site or a custom
  domain rather than a third project repository.
- **Versions are independent.** `ppforest2-core` keeps the `VERSION` file;
  `DESCRIPTION` is the single source of the R package version, and
  `CORE_VERSION` records which core it was built against. The shared `VERSION`
  file that drove both is gone.
- **`NEWS.md` is curated and hand-maintained**, no longer generated from this
  repository's `CHANGELOG.md`.
- **Future `ppmethods`** remains an umbrella package that would depend on
  `ppforest2`, never a rename of it.

## What is not done

- `secrets.GIST_TOKEN` and `vars.COVERAGE_GIST_ID` are not set on
  `ppforest2-core`, so its coverage badge step fails. Everything before it in
  that workflow passes.
- This repository is not archived yet.
