# Issue #79 APA 7 Compliance Design

## Context

Issue #79 identifies real APA 7 compliance gaps in the thesis paper output. Current source uses `apacite` in `paper/apa/thesis_apa.tex`, which renders APA-6-style reference patterns (including `Retrieved from`) and leaves student-paper front-matter requirements incomplete.

This design defines a focused fix for APA 7 compliance in the paper source and bibliography metadata.

## Scope

### In Scope

- `paper/apa/thesis_apa.tex`
- `paper/apa/references.bib`

### Out of Scope

- `paper/apa/README.md`
- Other repository files outside the two scoped files above

## Selected Approach

Use a full APA 7 migration path:

1. Replace `apacite`-based citation stack with an APA 7-compatible `biblatex-apa` setup in `thesis_apa.tex`.
2. Normalize bibliography metadata in `references.bib` so rendered references follow APA 7 DOI/URL behavior.

This approach is preferred over patching `apacite` behavior because it addresses root causes rather than output symptoms.

## Design

## 1) Architecture and Boundaries

- Keep paper narrative content intact.
- Restrict changes to formatting/tooling and metadata required for APA 7 compliance.
- Maintain existing document layout controls where they are already helpful (margins, doublespacing, page numbering, float controls), and only adjust where APA 7 behavior requires it.

## 2) Component Changes

### `paper/apa/thesis_apa.tex`

- Migrate bibliography engine to `biblatex` with APA 7 style (`biblatex-apa`) and `biber` backend.
- Replace legacy bibliography wiring (`\bibliographystyle`, `\bibliography`) with `\addbibresource{references.bib}` and `\printbibliography`.
- Ensure student title page includes required fields in order:
  - Full title
  - Author names
  - Department/institution affiliation
  - Course number and course name
  - Instructor/adviser name
  - Due/submission date
- Keep abstract page formatting APA-compliant:
  - Centered bold `Abstract`
  - Unindented abstract paragraph
  - `Keywords:` line with italicized label and proper indentation treatment
- Keep figure/table numbering and title presentation APA-consistent; avoid disruptive placement around known trouble zone (roughly current pages 31 to 36).
- Clean appendix presentation so supplementary material is intentional and clearly labeled; remove TODO/FIXME artifacts.
- Normalize problematic source punctuation where needed to reduce encoding artifacts in rendered output.

### `paper/apa/references.bib`

- Normalize entry metadata to support APA 7 rendering:
  - Use canonical `doi` fields where a DOI exists.
  - Keep `url` for non-DOI web sources.
  - Avoid metadata patterns that force legacy-style link phrasing.
- Correct obvious field-quality issues that break output quality (author formatting, malformed values, inconsistent casing where applicable).

## 3) Data and Render Flow

1. LaTeX reads citation configuration from `thesis_apa.tex`.
2. `biber` resolves entries from `references.bib` through `biblatex-apa`.
3. Final PDF renders APA 7-compliant in-text citations and references.
4. Front matter, abstract/keywords, figures/tables, and appendix formatting are validated in generated PDF output.

## 4) Error Handling and Risk Controls

- **Bibliography backend risk:** local environment package differences may affect `biber` output.
  - Mitigation: use conservative, documented `biblatex-apa` options and keep cite command usage consistent.
- **Command compatibility risk:** legacy citation commands may need adjustment after migration.
  - Mitigation: run compatibility pass for citation commands used in document body.
- **Reference metadata risk:** incomplete or malformed entries can still produce weak output.
  - Mitigation: prioritize cleanup for all cited entries and verify representative entry types.
- **Layout regression risk:** front-matter and bibliography changes can shift pagination/floats.
  - Mitigation: explicitly recheck known problematic figure/table region and appendix transition.
- **Encoding artifact risk:** mixed punctuation/encoding can reintroduce visual artifacts.
  - Mitigation: normalize punctuation and re-verify rendered text in affected sections.

## 5) Testing and Validation

- Compile with `biber`-based workflow until references and cross-references stabilize.
- Validate title page against APA 7 student-paper required fields.
- Validate abstract and keywords formatting.
- Validate references:
  - No APA-6-style `Retrieved from` for DOI entries.
  - APA 7 DOI/URL formatting behavior is consistent.
- Validate figure/table presentation and float flow in the previously disruptive region.
- Validate appendix heading and supplementary-material clarity.
- Validate rendered output for encoding artifacts.

## Acceptance Mapping

This design maps directly to issue #79 acceptance targets within the approved scope (`thesis_apa.tex` + `references.bib`), including successful compile, APA 7 front matter/abstract/references behavior, improved figure/table flow, and appendix clarity.
