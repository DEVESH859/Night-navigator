# Design System Document

## 1. Overview & Creative North Star: "The Luminous Sanctuary"

This design system is engineered for the "Luminous Sanctuary" aesthetic—a high-tech, night-optimized interface that balances high-visibility safety with a calming, premium atmosphere. We are moving away from the "flat utility" look of standard navigation apps. Instead, we are creating a digital cockpit that feels like a protective beacon in the dark.

The system breaks the traditional grid through **Intentional Asymmetry**. Important navigation cues should feel like they are floating in deep space, utilizing overlapping layers and hyper-rounded forms to soften the high-contrast "Neon" palette. By using tonal depth rather than structural lines, we create an interface that is easy on the eyes during nighttime use while maintaining an authoritative, high-end editorial feel.

---

## 2. Colors: The Neon Void

Our palette is built on a foundation of deep, ink-like navies to preserve night vision, punctuated by high-chroma signals for safety.

### The "No-Line" Rule
**Explicit Instruction:** Designers are prohibited from using 1px solid borders to define sections or containers. Boundaries must be established through:
- **Tonal Shifts:** Placing a `surface-container-low` (#0f131e) card on a `surface` (#0a0e18) background.
- **Negative Space:** Using the spacing scale to create clear separation.
- **Luminous Shadows:** Using light itself (glows) to define the edges of active elements.

### Surface Hierarchy & Nesting
Treat the UI as a physical stack of semi-transparent layers. 
- Use `surface-container-lowest` (#000000) for the base map or background.
- Use `surface-container` (#151926) for primary interface panels.
- Use `surface-bright` (#262c3c) for elevated, interactive elements like floating action buttons or search bars.

### The Glass & Gradient Rule
To achieve a signature look, use **Glassmorphism** for floating overlays. Apply the `surface-variant` (#202534) at 60% opacity with a `20px` backdrop blur. 
**Signature Texture:** Main CTAs should not be flat. Use a subtle linear gradient (Top-Left to Bottom-Right) transitioning from `primary` (#a8ffe1) to `primary-container` (#00fcca) to give the neon elements "soul" and dimension.

---

## 3. Typography: Tech-Forward Authority

We utilize a high-contrast typographic scale to ensure critical information is absorbed instantly at a glance.

*   **Display & Headlines (Space Grotesk):** This is our "Command" voice. Use `display-lg` and `headline-md` for critical navigation instructions (e.g., "TURN LEFT"). The geometric nature of Space Grotesk provides a high-tech, precise feel.
*   **Body & Titles (Be Vietnam Pro):** Our "Human" voice. While headings are technical, the body text is designed for comfort. Use `body-lg` for descriptive safety alerts. This font should feel approachable and calm.
*   **Labels (Space Grotesk):** For micro-data (ETA, Distance), use `label-md` in all-caps with a `0.05em` letter spacing to emphasize the "instrument panel" aesthetic.

---

## 4. Elevation & Depth: Tonal Layering

Traditional shadows represent height; in this system, shadows represent **Energy**.

*   **The Layering Principle:** Instead of standard shadows, stack your containers. Place a `surface-container-highest` (#202534) drawer over a `surface-container-low` (#0f131e) base. The difference in tonal value creates a soft, natural lift that is far more sophisticated than a drop shadow.
*   **Ambient Neon Glows:** For critical "Guardian" elements (like a safety SOS or active navigation line), use a diffused glow. Set the shadow color to `primary` (#a8ffe1) or `tertiary` (#ff6e81) with a 24px blur and only 15% opacity. This mimics the bloom of a neon light in the dark.
*   **The Ghost Border:** If accessibility requires a stroke (e.g., in a complex map environment), use the `outline-variant` (#444854) at 20% opacity. It should be felt, not seen.

---

## 5. Components

### Buttons (Hyper-Rounded)
- **Primary Button:** Pill-shaped (`full` radius: 9999px). Background is the `primary` gradient. Text is `on-primary` (#00654f).
- **Secondary Button:** `surface-bright` background with a subtle glow on hover.
- **Tertiary/Ghost:** No background. Use `primary` text for "Selection" actions and `on-surface-variant` for passive actions.

### Cards & Navigation Modules
- **Style:** Cards must use the `md` radius (1.5rem / 24px). 
- **Content:** Forbid the use of divider lines. Separate content using `surface-container-low` for the card body and `surface-container-high` for the header area of the card.

### Chips & Tags
- **Status Chips:** Use `full` radius. A "Safe Zone" chip uses `secondary-container` (#506600) with `on-secondary` (#455900) text.
- **Interactive Chips:** These should feel like tactile physical buttons; use `surface-variant` with a subtle `2px` inner glow.

### Input Fields
- **Safety Entry:** Inputs should be `surface-container-highest` with a `md` (24px) radius. Use `on-surface-variant` for placeholders. Upon focus, the container should not show a border but rather a `2px` outer glow in `primary`.

### Specialized Component: The Guardian Pulse
- A persistent, high-blur circular gradient behind the main navigation icon using `secondary` (#c3f400) at 10% opacity, subtly pulsing to indicate the "active monitoring" state.

---

## 6. Do’s and Don’ts

### Do:
- **Use Hyper-Rounding:** Every interactive element should feel smooth and organic to the touch.
- **Embrace the Void:** Let the `midnight` background breathe. Large margins (using our `xl` spacing) prevent the UI from feeling claustrophobic at night.
- **Prioritize the "Glow":** Use the `primary` and `secondary` neons only for actionable or life-saving information.

### Don't:
- **No 1px Lines:** Never use a solid white or grey line to separate items in a list. Use vertical white space or a change in surface tone.
- **No Harsh Whites:** Avoid using pure `#FFFFFF` for large blocks of text. Use `on-surface` (#e5e7f6) to reduce glare and eye fatigue.
- **No Sharp Corners:** Avoid the `none` or `sm` roundedness tokens unless it’s for a technical data visualization (like a sparkline).