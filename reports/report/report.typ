#import "lib.typ": *

#set document(
  title: "Comparative Analysis of Sequential Models for One-Step-Ahead Mobile Network Traffic Forecasting",
  author: "Shima Serein",
)
#set page(
  paper: "us-letter",
  margin: 1in,
  numbering: "1",
  number-align: top + right,
)
#set text(font: "Times New Roman", size: 12pt, lang: "en")
// The document this replaces used straight quotes throughout; keep them.
#set smartquote(enabled: false)
// `all: true` indents the first paragraph after a heading as well, which is what
// APA 7 asks for and what the document this replaces did.
#set par(
  justify: false,
  first-line-indent: (amount: 0.5in, all: true),
  leading: 0.62em,
  spacing: 0.62em,
)
#set heading(numbering: (..n) => {
  let v = n.pos()
  if v.len() == 1 { numbering("1.", ..v) } else { numbering("1.1", ..v) }
})
#show heading.where(level: 1): it => block(width: 100%, above: 1.4em, below: 0.7em)[
  #set align(center)
  #set text(size: 12pt, weight: "bold")
  #it
]
#show heading.where(level: 2): it => block(width: 100%, above: 1.1em, below: 0.45em)[
  #set text(size: 12pt, weight: "bold")
  #it
]
#set math.equation(numbering: "(1)")
#show raw: set text(font: "Times New Roman", size: 11pt)

// ---------------------------------------------------------------- title page
#page(numbering: none)[
  #v(2.2in)
  #align(center)[
    #set par(first-line-indent: 0pt, justify: false)
    #text(weight: "bold", size: 14pt)[
      Comparative Analysis of Sequential Models for One-Step-Ahead Mobile
      Network Traffic Forecasting
    ]
    #v(0.8in)
    Shima Serein
    #linebreak()
    Machine Learning Techniques I
    #linebreak()
    September 2026 Term
    #linebreak()
    20 September 2026
  ]
]
#counter(page).update(2)

#include "sections/00_abstract.typ"
#include "sections/01_introduction.typ"
#include "sections/02_related_work.typ"
#include "sections/03_dataset.typ"
#include "sections/04_exploratory.typ"
#include "sections/05_methodology.typ"
#include "sections/06_results.typ"
#include "sections/07_conclusion.typ"
#include "sections/08_disclosure.typ"
#include "sections/09_references.typ"
