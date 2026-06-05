#import "@preview/cetz:0.5.2"

#let dark = sys.inputs.at("dark", default: "false") == "true"
#let dark = true

#let fg = if dark { white } else { black }

#set page(width: 6.8cm, height: 3.5cm, margin: 0.4cm, fill: none)
#set text(font: "IBM Plex Sans", fill: fg)

#cetz.canvas({
  import cetz.draw: *

  let accent = rgb("#ff4136")

  set-style(fill: fg, stroke: (paint: fg))

  // vertical rules between the letters
  line((0.9, 0.0), (0.9, -2.5))
  line((2.3, -0.1), (2.3, -2.6))
  line((3.6, -0.3), (3.6, -2.8))
  line((4.4, -0.2), (4.4, -2.7))

  // slightly slanted horizontal rules
  line((0, -0.5), (6, -0.7))
  line((0, -2.4), (6, -2.3))

  // the word itself, baseline-aligned
  content((0, -2.24), anchor: "base-west", text(size: 2.5cm)[taulu])

  // detected "intersections"
  for x in (0.9, 2.3, 3.6, 4.4) {
    circle((x, -(0.5 + x / 30)), radius: 0.03, fill: accent, stroke: accent)
    circle((x, -(2.4 - x / 60)), radius: 0.03, fill: accent, stroke: accent)
  }
})
