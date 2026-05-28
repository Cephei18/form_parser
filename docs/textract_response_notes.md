# Textract Response Notes

## Observed Response Shape

The local sample run on `input/form.pdf` produced these block types:

- `PAGE`
- `LINE`
- `WORD`
- `KEY_VALUE_SET`
- `TABLE`
- `CELL`

No `SELECTION_ELEMENT` blocks were present in that sample, so the checkbox branch of the parser was not exercised by the current document.

## Block Graph Model

Textract responses are graph-shaped:

- each block has an `Id`
- parent blocks point to children through `Relationships`
- text is often split across `WORD` blocks
- form fields are represented by paired `KEY_VALUE_SET` blocks
- tables are represented by `TABLE` blocks with `CELL` children

The parser reconstructs readable output by following those relationships instead of relying on plain OCR ordering.

## Field Extraction Strategy

Form fields are recovered by:

1. finding `KEY_VALUE_SET` blocks tagged as `KEY`
2. following their `VALUE` relationships
3. collecting text from the related child blocks
4. storing the result in a normalized field map

Repeated labels are preserved as lists so the parser does not silently drop duplicates.

## Table Extraction Strategy

Table rows are rebuilt from `TABLE` and `CELL` blocks.
Each cell keeps:

- row and column index
- span information
- confidence
- geometry
- extracted text

The parser also emits a row-major matrix so inspection is easier when the table is regular.

## Checkbox Strategy

Checkboxes are represented through `SELECTION_ELEMENT` blocks when Textract returns them.
The parser records:

- selected or not selected state
- confidence
- geometry
- parent relationships
- any associated field labels that can be inferred from the block graph

This branch is ready for a checkbox-heavy sample even though the current local form did not include one.

## Geometry And Confidence

Geometry is preserved as returned by Textract:

- bounding box
- polygon points

Confidence values are kept on the relevant blocks so downstream evaluation can compare structural certainty across runs.

## Sample Run Notes

The current local sample run produced:

- 162 total blocks
- 12 recovered fields
- 1 table
- 0 checkboxes

That sample is useful for validating table and field reconstruction, but it is not enough to stress the checkbox branch.