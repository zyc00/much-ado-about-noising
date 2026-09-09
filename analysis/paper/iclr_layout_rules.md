# ICLR 2027 layout contract

Source of truth: the official ICLR 2027 files in
[`ICLR/Master-Template`](https://github.com/ICLR/Master-Template/tree/master/iclr2027).
Re-check the repository before submission in case the organizers revise it.

## Template constraints

- Use `iclr2027_conference.sty` and `iclr2027_conference.bst` without editing
  their formatting parameters.
- The format is single-column: 5.5-inch text width by 9-inch text height on US
  Letter paper.
- Body text is 10-point Times with 11-point leading.
- The initial submission allows 9 pages of main text; citations may occupy
  additional pages. Rebuttal and camera-ready versions allow 10 main-text
  pages.
- Keep the submission anonymous; do not enable `\iclrfinalcopy` before the
  camera-ready version.
- Include the required AI-use statement. It does not count toward the page
  limit.

## Figures

- Default to a single-column figure sized relative to `\linewidth`.
- Place the figure number and caption below the figure, in sentence case.
- Use vector PDF when possible. Ensure lines, labels, and markers remain legible
  at the final 5.5-inch width.
- Figures and captions must remain meaningful in grayscale; never rely on color
  alone to identify a method or condition.
- Prefer one claim per visual. Move diagnostic panels and exhaustive sweeps to
  the appendix.

## Tables

- Place the table number and title above the table, in sentence case.
- Use the normal body font. Do not solve width problems by shrinking the table
  or applying `\resizebox`.
- Use `booktabs`, no vertical rules, aligned decimals, explicit units, and
  minimal repeated text.
- If a table does not fit `\linewidth`, reduce or group columns, use compact
  labels defined in the caption, or move secondary results to the appendix.

## Validation checklist

- Compile under the official ICLR style after every substantial visual change.
- Check for overfull boxes, clipped labels, detached captions, and float order.
- Inspect the PDF at 100% scale and in grayscale.
- Audit the main-text page count before adding optional ablations.
