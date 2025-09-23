// FAVICON
#set page(width: 100mm, height: 100mm, margin: 0mm, fill: rgb(1, 1, 1, 0))

#let thicc = 9mm
#let line_config = (
  stroke: (thickness: thicc, paint: black, cap: "round"),
)

#let _line(..args) = place(top + left, line(..args))

#let radius = thicc / 3
#let _circle(x, y, ..args) = place(top + left, move(dx: x - radius, dy: y - radius, circle(
  fill: red,
  radius: thicc / 3,
  ..args,
)))

#_line(start: (18mm, 5mm), length: 90mm, angle: 94deg, ..line_config)
#_line(start: (7mm, 16mm), length: 87mm, angle: 3deg, ..line_config)
#_line(start: (84mm, 10mm), length: 83mm, angle: 90deg, ..line_config)
#_line(start: (4mm, 85mm), length: 88mm, angle: -2deg, ..line_config)

#_circle(17mm, 16mm)
#_circle(84mm, 19mm)
#_circle(12mm, 84mm)
#_circle(83.5mm, 82mm)

#place(horizon + center, text(font: "IBM Plex Sans", weight: 600, size: 70mm)[t])
