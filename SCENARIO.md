# Drop It Like It's Hot — Visualization Scenario

## Context

A real-time aerial firefighting visualization tool built for the Canadian aerial firefighting context (BC Wildfire Service). The tool shows a fire spreading across terrain and portrays the coordinated aerial response using two aircraft and ATU (Airborne Tracking Unit) data.

---

## The Aircraft

### Bird Dog

- Small fixed-wing aircraft carrying the **Air Attack Officer (AAO)**
- The AAO sits on the **right side** of the Bird Dog cockpit
- The AAO is the tactical brain of the operation — devising the attack plan, reading fire behavior, and coordinating via radio with the tanker pilot, ground crews, and the incident commander
- The Bird Dog first performs a **solo reconnaissance pass** over the fire before the tanker is involved
- During the tanker's drop run, the Bird Dog flies **diagonally behind the tanker, offset to the left** — this positions the AAO (right seat) with a direct sightline to watch where the retardant lands

### Air Tanker

- Large fixed-wing aircraft that carries and drops fire retardant
- Receives drop instructions from the AAO in the Bird Dog via radio
- Flies its own independent straight approach vector toward the fire's leading edge
- Equipped with an **ATU (Airborne Tracking Unit)** that records position, heading, altitude, and — critically — **tanker door open/close events**

---

## ATU Data

ATUs are GPS-enabled devices onboard aircraft that track:

- Aircraft location (lat/lon)
- Heading and altitude
- **Tanker door open event** → drop has started
- **Tanker door close event** → drop is complete

The retardant line drawn on the map is derived from the door open → close window combined with the tanker's flight path. This is the source of truth for where a drop happened.

---

## The Story — "The Attack Run"

The simulation plays out in four acts, driven by ATU events and aircraft behavior.

### Act 1 — Fire Spreading, Bird Dog Recon (Steps 1–4)

- Fire perimeter grows and drifts with the wind
- Bird Dog flies in **alone** and performs a reconnaissance pass over the fire
- The AAO is assessing: leading edge direction, wind, terrain, landmarks, hazards
- No tanker yet — this is purely an observation phase
- Status: `Bird Dog performing reconnaissance`

### Act 2 — Attack Plan Set, Tanker Called In (Steps 5–8)

- AAO identifies the **leading edge** — the side the fire is spreading toward fastest (downwind)
- AAO radios drop instructions to the Tanker using landmarks as reference points
- The Tanker appears from offscreen, arriving from the direction of the nearest airtanker base
- Status: `Tanker en route` → `Tanker on approach`

### Act 3 — The Drop Run (Steps 9–12)

- Tanker flies a straight approach vector toward the fire's leading edge
- Bird Dog positions itself **diagonally behind the tanker, offset to the left** — the AAO watches from the right seat
- **ATU: door open** → retardant line begins drawing on the map ahead of the fire
- **ATU: door close** → retardant line is complete, corridor fixed on the map
- Status: `Drop in progress` → `Drop complete — retardant line laid`

### Act 4 — Hold or Overtake (Steps 13–15)

- The fire reaches the retardant line
- Outcome depends on wind speed and fire behavior:
  - **Strong retardant effect** → fire slows and deflects at the line
  - **Overwhelmed** → fire crosses the line (high wind scenario)
- Tanker departs back toward base
- Status: `Tanker returning to base`

---

## Visual Language

| Element                   | Behavior                                                          | Meaning                        |
| ------------------------- | ----------------------------------------------------------------- | ------------------------------ |
| Bird Dog icon             | Solo recon pass over fire, then diagonal-left trail behind tanker | Reconnaissance then oversight  |
| Tanker icon               | Straight approach → drop run → exit                               | Execution                      |
| Retardant line (red/pink) | Draws during door-open window                                     | Where the drop landed          |
| Fire perimeter (orange)   | Grows and drifts each step                                        | Fire spread                    |
| Status panel              | Event-driven, ATU-sourced                                         | Current state of the operation |

---

## Status Events (ATU-Driven)

| ATU / Aircraft Event                       | Status Label                          |
| ------------------------------------------ | ------------------------------------- |
| Bird Dog solo recon pass                   | `Bird Dog performing reconnaissance`  |
| Tanker airborne, heading to fire           | `Tanker en route`                     |
| Tanker on approach vector                  | `Tanker on approach`                  |
| Bird Dog offset behind-left, tanker on run | `Bird Dog observing drop`             |
| Door open                                  | `Drop in progress`                    |
| Door close                                 | `Drop complete — retardant line laid` |
| Tanker departing                           | `Tanker returning to base`            |

---

## Key Design Decisions

- **No smoke drop from Bird Dog** — in Canada this only happens ~20% of the time; the story does not depend on it
- **Recon before the tanker arrives** — the Bird Dog always goes in first alone; the tanker is called in after the AAO has assessed the fire
- **Bird Dog diagonal-left trail** — during the drop run the Bird Dog is not leading or orbiting; it is offset behind-left so the AAO (right seat) has a direct sightline to assess drop accuracy
- **Retardant line ahead of the fire** — drops are placed in the path of the fire, not on the flames
- **ATU as the source of truth** — door events drive the map drawing and status panel, grounding the visualization in real data infrastructure
