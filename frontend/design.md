

**Project:** Rolls-Royce AEGIS · Predictive Maintenance Dashboard

**Stack:** React + Tailwind (or plain HTML/CSS/JS — your choice)

---

**Typography**
Use `IBM Plex Mono` for all data, numbers, labels, status text, and navigation. Use `IBM Plex Sans` (weight 300 for body, 600 for section headers) for all descriptive language. Never use Inter, Roboto, or system fonts. Import both from Google Fonts.

**Colour palette — hardcode these exact values, nothing else:**
```
--ink:     #0D0D0E   /* primary text, borders */
--ink2:    #3A3A3C   /* secondary text */
--ink3:    #8A8A8D   /* muted labels, hints */
--rule:    #DCDCDE   /* dividers, tracks */
--paper:   #F6F5F2   /* page background */
--panel:   #FFFFFF   /* card/pane background */
--red:     #C0302A   /* critical — act now *
--amber:   #B87020   /* warning — monitor */
--green:   #2A6B40   /* nominal — healthy */
--blue:    #1B4F8A   /* informational only */
```
No other colours. No gradients. No shadows. Colour is used semantically, never decoratively.

**Layout rules**
- Page background is `--paper`. Panels are `--panel`.
- Separate sections with `1px solid --ink` top borders, not cards with shadows.
- Use a `1px solid --rule` gap grid for multi-pane layouts, not padding/margin tricks.
- Padding is never uniform — left-heavy asymmetry (e.g. `padding: 20px 28px 20px 0`) creates editorial tension.
- No `border-radius` above `2px` on structural elements. Data is not friendly. It is precise.
- No box shadows anywhere.

**Components to build:**

1. **Nav bar** — monospace brand `RR/AEGIS · Predictive Maintenance`, links in 9px uppercase mono with `letter-spacing: 3px`, live status indicator in `--green` with a `●` prefix. Bottom border is `1px solid --ink`.

2. **Fleet strip** — horizontal row of engine cells. Each cell has: engine ID in 11px mono 500, a large RUL number (22px mono 500) coloured by status, a `label` for confidence, and a `2px` top border in the status colour (`--red` / `--amber` / `--green`). No full background fill — the top border *is* the status indicator.

3. **Degradation chart** — draw with Canvas or SVG. Historical line is `1.2px solid --ink`. Predicted continuation is `1px dashed --ink` with `stroke-dasharray: 3 3`. The confidence interval band is `rgba(184,112,32,0.12)` — warm amber wash, barely visible. A vertical `0.5px --ink3` line marks "now". The failure threshold is `0.5px dashed --red`. The point where prediction crosses threshold gets a hollow `--red` circle. No chart library unless you need one — draw it yourself.

4. **Sensor table** — 4 columns: sensor name (mono 10px), current value (mono 10px 500), drift % coloured by severity, anomaly dot (6px circle, filled `--amber` or `--red` if anomalous, `--rule` if nominal). A `3px` mini bar under each value shows normalised reading against operating range.

5. **Uncertainty decomposition** — three rows: Aleatoric, Epistemic, Combined. Each row has a track with a shaded band (the confidence interval) and a point (the estimate). Track is `--paper` background. Band is `rgba(180,150,100,0.18)`. Point is a `6px --ink` circle. Values in mono 9px right-aligned.

6. **Domain adaptation pills** — small tags, `0.5px solid --rule` border, mono 9px, `letter-spacing: 1px`. Active domains get `border-color: --ink`. Below them: one line of mono 9px muted text showing adaptation metric.

**Typography rules — non-negotiable**
- `class="label"` → mono, 9px, `letter-spacing: 2.5px`, uppercase, `--ink3`
- All data values → mono, varying size, weight 500
- Section headers → sans, 11px, weight 600, `letter-spacing: 3px`, uppercase
- Body text (descriptions, tooltips) → sans, weight 300, 13px, `line-height: 1.7`
- Never bold anything above weight 600. Never use weight 700 or 800.

**What makes this not generic AI slop — enforce these:**
- Section headers are separated by `border-top: 1px solid --ink` with `padding-top: 20px`, not wrapped in coloured header bars.
- The degradation chart's x-axis is a `3px` bar that fills proportionally to current cycle position — not tick marks.
- Numbers that represent predictions are always visually distinguished from measurements (dashed vs solid, or a `~` prefix in mono).
- Uncertainty is always shown as a range, never as a single confidence percentage.
- The word "critical" never appears as a badge with a coloured background fill. Status lives in the top border and the number colour only.
- Spacing between sections is created by `border` rules and `padding`, never by large `margin` gaps that create floating whitespace.

**Tone:** This is a safety-critical aerospace instrument. Every pixel should communicate precision and restraint. If a design decision makes it look friendlier, undo it.