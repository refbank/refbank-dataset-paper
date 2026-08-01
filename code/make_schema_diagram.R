# Builds an entity-relationship diagram of the Refbank core + metadata tables
# (derived/pipeline-output tables are intentionally excluded).
#
# Goal: show every field (no types) while avoiding a line-per-foreign-key
# rat's nest. Every FK field is color-tinted to match the header of the table
# it references, and most also get a drawn arrow from that field's row to the
# referenced table's header -- except the six dataset_id fields, which would
# each add a long, low-information arrow back to Datasets (every row belongs
# to *some* dataset; the tint alone says that) and, empirically, turned the
# figure back into a rat's nest when tried. Conditions' dataset_id keeps its
# arrow anyway, as the one visible anchor for where the schema is rooted.
#
# Field lists below were pulled directly from the live Redivis tables via
# refbankr::get_*(max_results = 1) |> names(), against v13 ("current" as of
# v13's release, which carries the renamed fields: target_image, round_num,
# option_set_size, selected_image, selection_id). Re-run against refbankr if
# the schema changes.
#
# PALETTE_VARIANT selects which color family to use for the six table
# identities (Datasets stays neutral gray in both): "solarized" uses four
# Solarized accent hues for the four tables that actually co-occur as FK
# tints inside another table's box (Conditions/Trials/Players/Images) plus
# two more Solarized accents for Messages/Selections (never FK targets
# themselves, so held to a looser bar); "paired" does the same from
# ColorBrewer's Paired-12 palette. Both four-hue "core" sets were chosen by
# testing every 4-combination from their source palette against the
# dataviz skill's validate_palette.js (`--pairs all`, since Trials/Selections
# each stack multiple different FK tints and any pair of them can appear
# side by side) -- most combinations from both source palettes fail (red vs.
# green and red vs. brown are famously CVD-confusable in Paired; Solarized's
# hues sit close enough on the wheel that no 5- or 6-hue subset clears the
# floor at all). The two combinations below are the ones that passed.
PALETTE_VARIANT <- "solarized" # "solarized" or "paired"

library(DiagrammeR)
library(DiagrammeRsvg)
library(rsvg)
library(here)

# table_id -> list(fill, header_fill, border). `border` is the saturated hue
# used for the table outline and arrows. `fill` is border blended ~78%
# toward white -- used for FK-tint cell backgrounds only, i.e. wherever the
# 11pt field text sits, so that text stays comfortably readable regardless
# of text color. `header_fill` is a separate, less-diluted blend (~50%
# toward white) used only for the header band -- darker/more prominent than
# the tints without dragging the tints along with it (they're two different
# blends of the same border hue now, not the same value reused in two
# places). Header text is always black -- deliberately, not per-hue-
# contrast-optimized: several of the saturated border hues (e.g. Paired's
# purple #6A3D9A, black-contrast 2.75) fail badly for black text and would
# need white instead, but a black/white split per table reads as
# inconsistent and is one more thing to get wrong when swapping palettes.
# Both fill and header_fill clear >=9:1 black-text contrast on every hue
# below, comfortably clear of the >=4.5:1 target with room to spare.
solarized_palette <- list(
  datasets   = list(fill = "#E6E6E6", header_fill = "#BABABA", border = "#757575"),
  conditions = list(fill = "#EFE5C7", header_fill = "#DAC480", border = "#B58900"),
  trials     = list(fill = "#F7D2D1", header_fill = "#EE9897", border = "#DC322F"),
  messages   = list(fill = "#CFE5F5", header_fill = "#92C5E8", border = "#268BD2"),
  selections = list(fill = "#E4E9C7", header_fill = "#C2CC80", border = "#859900"),
  players    = list(fill = "#DFE0F2", header_fill = "#B6B8E2", border = "#6C71C4"),
  images     = list(fill = "#D0EAE8", header_fill = "#94D0CC", border = "#2AA198")
)

paired_palette <- list(
  datasets   = list(fill = "#E6E6E6", header_fill = "#BABABA", border = "#757575"),
  conditions = list(fill = "#FFE4C7", header_fill = "#FFBF80", border = "#FF7F00"),
  trials     = list(fill = "#C7E0EE", header_fill = "#8FBCDA", border = "#1F78B4"),
  messages   = list(fill = "#D4EEC9", header_fill = "#99D096", border = "#33A02C"),
  selections = list(fill = "#EDDCC9", header_fill = "#D8AC94", border = "#B15928"),
  players    = list(fill = "#E1D8EC", header_fill = "#B49ECC", border = "#6A3D9A"),
  images     = list(fill = "#F9D2D0", header_fill = "#F18C8E", border = "#E31A1C")
)

palette <- if (PALETTE_VARIANT == "solarized") solarized_palette else paired_palette

# Each field is either a plain string (own field) or c(name, references = "table_id")
# for a foreign key, so the diagram-builder knows which color to tint it.
#
# pos = c(x, y) in inches, node CENTER, for the neato "pinned" layout below
# (pos="x,y!"). Datasets-Conditions-Trials stay top-aligned in a row;
# Selections/Messages form their own right-hand tier, stacked closer together
# than in the earlier layout since removing the Derived table (see below)
# freed the vertical space that used to separate them; Players/Images sit
# below Trials, nudged up to close some of the resulting whitespace.
tables <- list(
  datasets = list(
    label = "Datasets", pk = "dataset_id", pos = c(1.0, 7.7),
    fields = list("dataset_id", "full_cite", "short_cite")
  ),
  conditions = list(
    label = "Conditions", pk = "condition_id", pos = c(4, 7.7),
    fields = list(
      "condition_id", c("dataset_id", "datasets"),
      "condition_label", "group_size", "language",
      "prior_relationship", "partner_constancy", "role_constancy",
      "population", "confederates", "modality", "feedback", "backchannel"
    )
  ),
  trials = list(
    label = "Trials", pk = "trial_id", pos = c(7, 7.7),
    fields = list(
      "trial_id", c("dataset_id", "datasets"), c("condition_id", "conditions"),
      c("describer", "players"), c("matchers", "players"),
      c("target_image", "images"), c("image_options", "images"),
      "game_id", "room_num", "stage_num", "trial_num", "round_num", "order_match",
      "option_set_size", "exclude", "exclusion_reason"
    )
  ),
  messages = list(
    label = "Messages", pk = "message_id", pos = c(10, 8.5),
    fields = list(
      "message_id", c("dataset_id", "datasets"), c("trial_id", "trials"),
      c("player_id", "players"), "role", "text", "message_num",
      "message_irrelevant", "time_stamp"
    )
  ),
  selections = list(
    label = "Selections", pk = "selection_id", pos = c(10, 5.6),
    fields = list(
      "selection_id", c("dataset_id", "datasets"), c("trial_id", "trials"),
      c("player_id", "players"), c("selected_image", "images"), "time_stamp"
    )
  ),
  # Players/Images sit well below Trials' bottom edge, not just below its
  # center: the edge router draws Selections/Messages -> Players and
  # Selections -> Images as a straight horizontal run at the *target's*
  # header height, crossing the full width of whatever sits between the two
  # columns (Trials, geometrically, since Selections/Messages are the
  # rightmost column and Players/Images the leftmost). If that header height
  # falls anywhere inside Trials' tall (16-row) vertical span, the line
  # cuts straight through Trials' interior instead of passing under it --
  # this happened at pos=(1,4.6)/(4,4.2), which left only ~0.1in of
  # clearance under Trials' bottom edge, and the run landed inside it. Kept
  # at least ~0.4in of clearance below Trials' bottom edge here; if Trials
  # gains fields (taller box) or moves, re-check this margin.
  players = list(
    label = "Players", pk = "player_id", pos = c(1, 4),
    fields = list(
      "player_id", c("dataset_id", "datasets"), "age", "gender",
      "native_language", "race", "education"
    )
  ),
  images = list(
    label = "Images", pk = "image_id", pos = c(4, 3.9),
    fields = list(
      "image_id", c("dataset_id", "datasets"), "image_type",
      "kilogram_id", "image_path"
    )
  )
)

field_row <- function(field, table_id) {
  is_fk <- length(field) == 2
  name <- field[1]
  is_pk <- identical(name, tables[[table_id]]$pk)

  text <- if (is_pk) {
    glue::glue("<B>{name} &#128273;</B>")
  } else if (is_fk) {
    glue::glue("<I>{name}</I>")
  } else {
    name
  }
  # FK fields get a background tint (the referenced table's fill color)
  # instead of colored text, so the relationship reads as a highlight rather
  # than requiring the reader to match up font colors.
  bgcolor_attr <- if (is_fk) glue::glue(' BGCOLOR="{palette[[field[2]]]$fill}"') else ""
  glue::glue('<TR><TD ALIGN="LEFT" PORT="{name}"{bgcolor_attr}><FONT POINT-SIZE="11">{text}</FONT></TD></TR>')
}

table_node <- function(table_id) {
  spec <- tables[[table_id]]
  colors <- palette[[table_id]]
  pos <- glue::glue("{spec$pos[1]},{spec$pos[2]}!")
  rows <- purrr::map_chr(spec$fields, field_row, table_id = table_id)

  glue::glue(
    '{table_id} [pos="{pos}", label=<
      <TABLE BORDER="2.25" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" COLOR="{colors$border}" BGCOLOR="white" WIDTH="150" FIXEDSIZE="FALSE">
      <TR><TD WIDTH="150" PORT="header" BGCOLOR="{colors$header_fill}"><FONT COLOR="black" POINT-SIZE="13"><B>{spec$label}</B></FONT></TD></TR>
      {paste(rows, collapse = "\n")}
      </TABLE>
    >]'
  )
}

# Field-level FK edges: one arrow per foreign-key field, from that field's
# own row to the header band of the table it references. Built directly from
# `tables` (the single source of truth for which fields are FKs) rather than
# a hand-maintained edge list, so it can't drift out of sync with the schema.
# Colored to match the tint already on the FK cell, so the arrow and the
# color-coding reinforce the same relationship instead of adding a second,
# uncoordinated visual language.
#
# dataset_id is excluded everywhere except Conditions: every table carries
# it, so drawing all six would add one long arc per table back to Datasets --
# both the least informative relationship in the schema (every row just
# belongs to *some* dataset) and, empirically, the most visually dominant --
# tried it, it turned the figure back into the rat's nest this design avoids.
# The gray tint alone conveys "this points back to Datasets" for the other
# five; Conditions keeps its arrow as the one visible anchor showing where
# the whole schema is rooted.
fk_edges <- tables |>
  purrr::imap(function(spec, table_id) {
    purrr::keep(spec$fields, ~ length(.x) == 2 && (.x[1] != "dataset_id" || table_id == "conditions")) |>
      purrr::map(function(field) {
        list(from = table_id, from_port = field[1], to = field[2])
      })
  }) |>
  purrr::flatten()

dot <- glue::glue('
digraph schema {{
  graph [layout=neato, fontname="Helvetica"]
  node [shape=plaintext, fontname="Helvetica"]

  {paste(purrr::map_chr(names(tables), table_node), collapse = "\n\n")}
}}
')

graph <- DiagrammeR::grViz(dot)
svg <- DiagrammeRsvg::export_svg(graph)

# --- Manual edge routing ------------------------------------------------
# We draw every FK edge ourselves as a left-then-down/up-then-left dogleg,
# rather than asking Graphviz for a node:port:compass edge, because that
# path was tried first and didn't work: Graphviz (via DiagrammeR's viz.js
# renderer) silently drops the port and docks every edge at the plain
# node-level compass point instead -- e.g. a `conditions:dataset_id:w ->
# datasets:header:e` edge rendered as if it had been written
# `conditions:w -> datasets:e`, leaving from Conditions' vertical center
# rather than its dataset_id row. What *is* reliable is the per-cell
# geometry Graphviz already draws: every FK cell and every header cell has
# BGCOLOR set, so each gets its own filled <polygon> in the output SVG, in
# the same top-to-bottom order as the table's HTML rows. We read those
# polygons back out (parse_field_bbox()) and draw the arrows ourselves, so
# each one starts exactly at its named field's row and lands exactly on the
# target's header, with one shared gap and arrowhead size throughout.
GAP <- 4
ARROW_LEN <- 13.5 # 9 * 1.5, matching the 50%-thicker arrows/borders
ARROW_HALF_WIDTH <- 4.725 # 3.15 * 1.5

# The block for one node runs from its <title> to the next top-level closing
# </g> -- i.e. everything up to (but not including) the line that closes
# this node's own <g class="node">, which comes right after its outline
# polygon.
node_block <- function(svg_text, node_id) {
  start <- regexpr(glue::glue("<title>{node_id}</title>"), svg_text, fixed = TRUE)
  rest <- substr(svg_text, start, nchar(svg_text))
  end <- regexpr("\n</g>", rest, fixed = TRUE)
  substr(rest, 1, end - 1)
}

bbox_from_points <- function(pts_str) {
  coords <- do.call(rbind, lapply(strsplit(pts_str, " ")[[1]], function(p) as.numeric(strsplit(p, ",")[[1]])))
  list(xmin = min(coords[, 1]), xmax = max(coords[, 1]), ymin = min(coords[, 2]), ymax = max(coords[, 2]))
}

# A node's own filled polygons include, in row order, one polygon per
# BGCOLOR'd row (header, then each FK field in field order) -- plus a
# variable number of background/outline polygons we don't care about
# (Graphviz draws a different count of these depending on BORDER width: an
# outline polygon at BORDER="1", none at BORDER="2", discovered the hard way
# when thickening the border silently shifted every downstream index). Rather
# than assume a fixed position for the bookend polygons, filter to only the
# polygons whose fill matches one of *this table's own* row colors (header
# fill, then each FK field's referenced-table fill) -- backgrounds are always
# white/none, never one of our saturated or tinted hues, so this reliably
# recovers exactly [header, fk_1, fk_2, ...] regardless of what Graphviz
# wraps around them. Matching by color instead of a raw color->row lookup
# still needs the field list for *order and duplicates*: two FKs can
# reference the same table (e.g. Trials' target_image/image_options both
# tint images-colored) and the row_fills vector preserves that repeat so
# both polygons are kept, one per row, in document order.
parse_field_bbox <- function(svg_text, table_id, port) {
  block <- node_block(svg_text, table_id)
  matches <- gregexpr('<polygon fill="[^"]*" stroke="[^"]*" points="[^"]+"', block)
  polygons <- regmatches(block, matches)[[1]]
  fills <- sub('.*polygon fill="([^"]*)".*', "\\1", polygons)
  points_strs <- sub('.*points="([^"]+)".*', "\\1", polygons)

  row_fills <- c(
    palette[[table_id]]$header_fill,
    purrr::keep(tables[[table_id]]$fields, ~ length(.x) == 2) |>
      purrr::map_chr(~ palette[[.x[2]]]$fill)
  )
  content <- points_strs[tolower(fills) %in% tolower(row_fills)]

  index <- if (identical(port, "header")) {
    1
  } else {
    fk_fields <- purrr::keep(tables[[table_id]]$fields, ~ length(.x) == 2) |> purrr::map_chr(1)
    1 + which(fk_fields == port)
  }
  bbox_from_points(content[index])
}

# Note: in this SVG's coordinate system, y values are negative and *larger*
# (closer to zero) further down the page, so the "bottom" edge is ymax.
attach_point <- function(bbox, side) {
  cx <- (bbox$xmin + bbox$xmax) / 2
  cy <- (bbox$ymin + bbox$ymax) / 2
  switch(side,
    s = c(cx, bbox$ymax),
    n = c(cx, bbox$ymin),
    e = c(bbox$xmax, cy),
    w = c(bbox$xmin, cy)
  )
}

# Outward unit normal for each compass side, i.e. the direction that moves
# *away* from the box -- used to hold every edge to the same GAP on both
# the tail and the head end alike.
outward_normal <- list(s = c(0, 1), n = c(0, -1), e = c(1, 0), w = c(-1, 0))

# Every FK edge leaves its field row heading west, takes a short stub-length
# step before turning, then travels vertically to line up with the target
# header's height, then goes the rest of the way across -- a fixed 3-segment
# "stub, drop/rise, run" shape rather than a single elbow. The turn is what
# makes this robust to boxes sitting between source and target: the stub only
# needs to clear whatever's immediately next to the source (the gap between
# adjacent columns), and the long horizontal leg then runs at the *target's*
# row height, not the source's -- which is what let the describer/matchers/
# target_image/image_options edges (Trials -> Players/Images) stop cutting through
# Conditions' interior, since Conditions sits at Trials' height but Players
# and Images sit a full tier below it.
#
# Stub length is keyed by *target* table, not a single constant: several
# edges can turn at the same x (e.g. everything landing on Trials, or on
# Images) and a shared column of turn-points reads as more deliberate than
# each edge bending at its own default distance. Not in `palette` because
# it's a layout property of the current positions, not a property of the
# table itself -- it'll need revisiting if `tables[[]]$pos` changes.
STUB_DEFAULT <- 15
stub_by_target <- c(images = 30, players = 45)
stub_for <- function(target_table) {
  if (target_table %in% names(stub_by_target)) stub_by_target[[target_table]] else STUB_DEFAULT
}

build_fk_edge_svg <- function(svg_text, edge) {
  from_bbox <- parse_field_bbox(svg_text, edge$from, edge$from_port)
  to_bbox <- parse_field_bbox(svg_text, edge$to, "header")
  color <- palette[[edge$to]]$border

  p1 <- attach_point(from_bbox, "w") + GAP * outward_normal[["w"]]
  tip <- attach_point(to_bbox, "e") + GAP * outward_normal[["e"]]
  p2 <- tip + ARROW_LEN * outward_normal[["e"]] # path/line stops at the arrow's base

  stub_end <- p1 + stub_for(edge$to) * outward_normal[["w"]]
  turn <- c(stub_end[1], p2[2])
  path_d <- glue::glue(
    "M{p1[1]},{p1[2]} L{stub_end[1]},{stub_end[2]} L{turn[1]},{turn[2]} L{p2[1]},{p2[2]}"
  )

  perp <- c(0, 1) * ARROW_HALF_WIDTH # perpendicular to the east-pointing arrowhead
  b1 <- p2 + perp
  b2 <- p2 - perp
  poly_pts <- glue::glue("{b1[1]},{b1[2]} {tip[1]},{tip[2]} {b2[1]},{b2[2]} {b1[1]},{b1[2]}")

  glue::glue(
    '<g class="edge"><title>{edge$from}:{edge$from_port}&#45;&gt;{edge$to}</title>\n',
    '<path fill="none" stroke="{color}" stroke-width="1.95" d="{path_d}"/>\n', # 1.3 * 1.5
    '<polygon fill="{color}" stroke="{color}" stroke-width="1.95" points="{poly_pts}"/>\n',
    "</g>\n"
  )
}

edges_svg <- paste(purrr::map_chr(fk_edges, \(edge) build_fk_edge_svg(svg, edge)), collapse = "\n")
svg <- sub("</g>\n</svg>", glue::glue("{edges_svg}</g>\n</svg>"), svg, fixed = TRUE)
# --------------------------------------------------------------------------

out_stem <- glue::glue("manuscript/figs/schema_v3_{PALETTE_VARIANT}")
writeLines(svg, here(glue::glue("{out_stem}.svg")))
rsvg::rsvg_png(here(glue::glue("{out_stem}.svg")), here(glue::glue("{out_stem}.png")), width = 2400)

message(glue::glue("Wrote {out_stem}.png"))
