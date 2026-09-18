// Shared formatting helpers for the report.
//
// The document follows APA 7 structure (title page, abstract with keywords,
// numbered sections, Table/Figure blocks with the label above the caption) with
// IEEE numeric citations, which is the combination the brief asks for.

#let rule = table.hline(stroke: 0.5pt)

// A table block: bold label, italic caption, then the table itself.
//
// `breakable: false` keeps the three parts together. Left breakable, a table can leave its
// label and caption at the foot of one page and its rows at the head of the next, or split
// its rows so the orphaned ones read as the following table's.
#let apa-table(number, caption, body) = block(
  width: 100%, above: 1.1em, below: 1.1em, breakable: false,
)[
  #set par(first-line-indent: 0pt, justify: false)
  #text(weight: "bold")[Table #number]
  #linebreak()
  #emph(caption)
  #v(0.35em)
  #body
]

// A figure block: bold label, italic caption, then the image.
//
// `width` is the preferred width. Several of these plots are much taller than they are
// wide, and at full text width they overflow the page, which pushes the whole block to
// a fresh page and leaves the previous one nearly empty. `max-height` caps the rendered
// height instead, so a tall figure shrinks to fit rather than displacing a page of text.
// It also floats: a tall figure placed in the flow pushes itself to the next page and
// leaves the remainder of the current one empty, so instead it rises to the top of the
// next page while the text around it keeps filling.
#let apa-figure(number, caption, path, width: 100%, max-height: 6.4in) = place(
  top, float: true, clearance: 1.2em,
)[
  #block(width: 100%, breakable: false)[
  #set par(first-line-indent: 0pt, justify: false)
  #text(weight: "bold")[Figure #number]
  #linebreak()
  #emph(caption)
  #v(0.4em)
  #layout(size => {
    let target = width * size.width
    let drawn = image(path, width: target)
    align(center, if measure(drawn).height > max-height {
      image(path, height: max-height)
    } else {
      drawn
    })
  })
  ]
]

// Table body with APA rules: no vertical lines, a rule above and below the
// header row and one closing the table.
#let apa-body(columns, align: auto, ..cells) = table(
  columns: columns,
  align: align,
  stroke: none,
  inset: (x: 4pt, y: 3.6pt),
  ..cells,
)

// One numbered reference with a hanging indent.
#let ref-entry(number, body) = block(above: 0.55em, below: 0.55em)[
  #set par(first-line-indent: 0pt, justify: false)
  #grid(columns: (0.42in, 1fr), gutter: 0pt, [\[#number\]], body)
]

// The video URL is a build input so the reference can never be published with a
// placeholder: `make report VIDEO_URL=...` sets it, and the plain `make report`
// target refuses to build without it.
#let video-url = sys.inputs.at("video", default: "")
