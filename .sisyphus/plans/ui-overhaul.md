# UI Overhaul: Light Luxurious Premium Redesign

## TL;DR

> **Quick Summary**: Complete visual redesign of Omnibus Legal Compass from flat MVP to premium Light Luxurious theme with heavy Framer Motion animations, card-based Perplexity-style Q&A, and consistent top navbar across all 3 pages.
> 
> **Deliverables**:
> - New design system (colors, typography, spacing, shadows)
> - Framer Motion animation infrastructure
> - Redesigned top navbar (consistent across all pages)
> - Premium Q&A page with card-based streaming answers
> - Premium Compliance Checker page
> - Premium Business Guidance page
> - Typing animations, skeleton loaders, page transitions
> - Ambient gradient background effects
> - Toast notification system
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 (deps) → Task 2 (design system) → Task 3 (layout) → Tasks 4-6 (pages, parallel) → Task 7 (polish)

---

## Context

### Original Request
User wants the UI to look like it's "worth $100,000,000,000,000,000,000,000" — a complete premium redesign of the existing functional MVP.

### Interview Summary
**Key Discussions**:
- **Theme**: Light Luxurious — clean white, rich blue accents, elegant shadows, premium typography (Apple.com meets Notion)
- **Scope**: All 3 pages equally — Q&A, Compliance, Guidance
- **Animations**: Heavy — Framer Motion everywhere
- **Navigation**: Top horizontal navbar with blur backdrop
- **Chat UI**: Card-based like Perplexity (full-width answer cards, expandable sources)
- **Extras**: ALL — typing animation, skeleton loaders, page transitions, ambient background, toast notifications

### Current State (from Playwright exploration)
- **Q&A Page (/)**: Top nav (inconsistent with other pages), hero text, basic text input, example buttons, streaming toggle, placeholder area, footer
- **Compliance (/compliance)**: "Kembali ke Beranda" back link (no navbar!), text/PDF tabs, textarea, button
- **Guidance (/guidance)**: "Kembali ke Beranda" back link (no navbar!), radio buttons for 5 business types, optional fields
- **Tech stack**: Next.js App Router + Tailwind CSS + TypeScript
- **Missing**: No animation library, no design tokens, no component library, inconsistent navigation

---

## Work Objectives

### Core Objective
Transform every page from flat MVP into a world-class premium legal AI interface with consistent design language, fluid animations, and polished micro-interactions.

### Concrete Deliverables
- Updated `tailwind.config.ts` with design tokens (colors, fonts, shadows, animations)
- `globals.css` with custom properties, ambient background styles, glass effects
- Framer Motion + sonner (toast) installed
- `Navbar.tsx` — shared premium navbar component
- `PageTransition.tsx` — animated page wrapper
- `SkeletonLoader.tsx` — beautiful loading placeholders
- `AmbientBackground.tsx` — gradient mesh / animated background
- Redesigned `page.tsx` (Q&A) with card-based chat
- Redesigned `compliance/page.tsx`
- Redesigned `guidance/page.tsx`
- Redesigned `layout.tsx` with navbar + page transitions
- Updated `StreamingAnswerCard.tsx` with typing animation + premium styling
- Updated `CitationCard.tsx` with glass morphism + expand/collapse
- Toast integration for success/error states

### Definition of Done
- [ ] All 3 pages render with new design when visiting localhost:3000
- [ ] Navigation is consistent (top navbar) across all pages
- [ ] Page transitions animate smoothly between routes
- [ ] Streaming answers show typing animation effect
- [ ] Skeleton loaders appear during data fetching
- [ ] Toast notifications appear on success/error
- [ ] Ambient background is visible on hero sections
- [ ] All existing backend API contracts unchanged
- [ ] No TypeScript errors (`npm run build` succeeds)

### Must Have
- Light luxurious color palette (white base, blue accents)
- Plus Jakarta Sans or Inter font
- Framer Motion animations on ALL interactive elements
- Consistent top navbar on ALL pages
- Card-based answer display with citation expansion
- Mobile responsive (375px - 1440px+)

### Must NOT Have (Guardrails)
- NO dark mode (light only — keep scope focused)
- NO backend changes whatsoever
- NO changes to API types/contracts in api.ts (only UI rendering)
- NO new pages or routes (only redesign existing 3)
- NO third-party UI component libraries (shadcn, MUI, etc.) — pure Tailwind + custom components
- NO excessive file restructuring — keep existing file locations, upgrade contents
- NO removing any existing functionality (streaming toggle, PDF upload, business type selection must all still work)

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks are verified by the executing agent using Playwright.

### Test Decision
- **Infrastructure exists**: YES (Next.js build)
- **Automated tests**: NO (UI overhaul — visual verification via Playwright)
- **Framework**: Playwright screenshots + `npm run build`

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

**Verification Tool**: Playwright (playwright skill) for all UI tasks
- Navigate to each page
- Take screenshots for visual verification
- Assert DOM elements exist with correct structure
- Verify animations don't break functionality
- Run `npm run build` to verify no TypeScript errors

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
└── Task 1: Install dependencies (framer-motion, sonner, @fontsource/plus-jakarta-sans)

Wave 2 (After Wave 1):
├── Task 2: Design system (tailwind.config + globals.css + design tokens)
└── Task 3: Shared components (Navbar, PageTransition, SkeletonLoader, AmbientBackground, layout.tsx)

Wave 3 (After Wave 2):
├── Task 4: Q&A page redesign
├── Task 5: Compliance page redesign
└── Task 6: Guidance page redesign

Wave 4 (After Wave 3):
└── Task 7: Final polish, toast integration, build verification
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3 | None |
| 2 | 1 | 3, 4, 5, 6 | None |
| 3 | 2 | 4, 5, 6 | None |
| 4 | 3 | 7 | 5, 6 |
| 5 | 3 | 7 | 4, 6 |
| 6 | 3 | 7 | 4, 5 |
| 7 | 4, 5, 6 | None | None |

---

## Design System Specification

### Color Palette (Light Luxurious)
```
--color-bg-primary: #FFFFFF          (pure white base)
--color-bg-secondary: #F8FAFC        (slate-50 — subtle off-white sections)
--color-bg-tertiary: #F1F5F9         (slate-100 — card backgrounds)
--color-bg-glass: rgba(255,255,255,0.8) (glassmorphism cards)

--color-accent-primary: #2563EB      (blue-600 — primary actions, links)
--color-accent-secondary: #1E40AF    (blue-800 — headers, emphasis)
--color-accent-light: #DBEAFE        (blue-100 — subtle highlights)
--color-accent-gradient: linear-gradient(135deg, #2563EB, #7C3AED) (blue-to-violet hero accents)

--color-text-primary: #0F172A        (slate-900 — body text)
--color-text-secondary: #475569      (slate-600 — secondary text)
--color-text-muted: #94A3B8          (slate-400 — placeholders)

--color-border: #E2E8F0              (slate-200 — borders)
--color-border-accent: #BFDBFE       (blue-200 — active borders)

--color-success: #059669             (emerald-600)
--color-warning: #D97706             (amber-600)
--color-error: #DC2626               (red-600)
--color-confidence-high: #059669     (emerald)
--color-confidence-medium: #D97706   (amber)
--color-confidence-low: #DC2626      (red)
```

### Typography
```
Font Family: 'Plus Jakarta Sans', system-ui, sans-serif
Font Weights: 400 (body), 500 (labels), 600 (subheadings), 700 (headings), 800 (hero)

Scale:
- Hero: 48px/52px, weight 800, tracking -0.02em
- H1: 36px/40px, weight 700, tracking -0.02em  
- H2: 24px/32px, weight 600
- H3: 20px/28px, weight 600
- Body: 16px/24px, weight 400
- Small: 14px/20px, weight 400
- Caption: 12px/16px, weight 500, uppercase, tracking 0.05em
```

### Shadows
```
--shadow-sm: 0 1px 2px rgba(0,0,0,0.04)
--shadow-md: 0 4px 6px -1px rgba(0,0,0,0.06), 0 2px 4px -2px rgba(0,0,0,0.06)
--shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04)
--shadow-xl: 0 20px 25px -5px rgba(0,0,0,0.08), 0 8px 10px -6px rgba(0,0,0,0.04)
--shadow-glow: 0 0 40px rgba(37,99,235,0.15) (blue glow for focus states)
```

### Spacing & Radius
```
Border Radius: 
- sm: 8px (buttons, inputs)
- md: 12px (cards)
- lg: 16px (panels, modals)
- xl: 24px (hero sections)
- full: 9999px (pills, avatars)

Content max-width: 768px (chat area, like Perplexity)
Page padding: 24px mobile, 48px desktop
Section gap: 32px
```

### Glass Morphism
```css
.glass {
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.3);
}
```

---

## TODOs

- [ ] 1. Install Dependencies & Font Setup

  **What to do**:
  - Install: `npm install framer-motion sonner @fontsource-variable/plus-jakarta-sans`
  - Verify installations in package.json
  - Import font in layout.tsx or globals.css: `import '@fontsource-variable/plus-jakarta-sans'`
  - Verify build succeeds after installations

  **Must NOT do**:
  - Do NOT install shadcn, MUI, Chakra, or any UI framework
  - Do NOT install tailwindcss-animate unless needed for specific keyframes
  - Do NOT change any backend files

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Font and dependency installation for Next.js

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (solo)
  - **Blocks**: Tasks 2, 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `frontend/package.json` — Current dependencies list, check what's already installed
  - `frontend/src/app/layout.tsx` — Where to import the font

  **External References**:
  - Framer Motion: `https://www.framer.com/motion/`
  - Sonner (toast): `https://sonner.emilkowal.dev/`
  - Plus Jakarta Sans: `https://fontsource.org/fonts/plus-jakarta-sans`

  **Acceptance Criteria**:
  - [ ] `framer-motion`, `sonner`, `@fontsource-variable/plus-jakarta-sans` appear in package.json dependencies
  - [ ] `npm run build` succeeds with no errors

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Dependencies installed and build passes
    Tool: Bash
    Steps:
      1. cd frontend && cat package.json | grep "framer-motion"
      2. cat package.json | grep "sonner"
      3. cat package.json | grep "plus-jakarta-sans"
      4. npm run build
    Expected Result: All 3 deps found, build succeeds
  ```

  **Commit**: YES (groups with 2)
  - Message: `feat(ui): install framer-motion, sonner, plus-jakarta-sans`
  - Files: `frontend/package.json`, `frontend/package-lock.json`

---

- [ ] 2. Design System: Tailwind Config + Global CSS + Design Tokens

  **What to do**:
  - Update `tailwind.config.ts` with the full color palette, font family, shadow scale, border-radius scale, and animation keyframes from the Design System Specification above
  - Update `globals.css` with:
    - CSS custom properties for all design tokens
    - Plus Jakarta Sans as default font
    - Glass morphism utility classes (`.glass`, `.glass-strong`)
    - Ambient background gradient animations (`@keyframes gradient-shift`)
    - Smooth scrolling, selection colors
    - Custom scrollbar styling (thin, blue accent)
    - Focus ring styles (blue glow)
    - Base component reset styles
  - Ensure all colors reference the design tokens for consistency

  **Must NOT do**:
  - Do NOT add dark mode variables or `dark:` prefixes
  - Do NOT override Tailwind's default spacing scale (extend, don't replace)
  - Do NOT remove any existing CSS that might affect current functionality

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`, `typography`]
    - `frontend-ui-ux`: Design system architecture
    - `typography`: Font scale, line heights, letter spacing precision

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 1)
  - **Blocks**: Tasks 3, 4, 5, 6
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `frontend/tailwind.config.ts` — Current Tailwind config to extend (NOT replace)
  - `frontend/src/app/globals.css` — Current global styles to enhance

  **Design Specification References**:
  - Color Palette section above — exact hex codes
  - Typography section above — exact font sizes, weights, tracking
  - Shadows section above — exact shadow values
  - Glass Morphism section above — exact CSS

  **Acceptance Criteria**:
  - [ ] `tailwind.config.ts` has extended theme with `colors.accent`, `colors.surface`, `fontFamily.sans`, `boxShadow.glow`, `borderRadius.xl`
  - [ ] `globals.css` has CSS custom properties matching design specification
  - [ ] Plus Jakarta Sans is the default body font (visible in browser)
  - [ ] `.glass` class produces frosted glass effect
  - [ ] `npm run build` succeeds

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Design tokens render correctly
    Tool: Playwright (playwright skill)
    Steps:
      1. Navigate to: http://localhost:3000
      2. Evaluate: document.body.style.fontFamily or getComputedStyle
      3. Assert: Font family includes "Plus Jakarta Sans"
      4. Assert: CSS variable --color-accent-primary resolves to #2563EB
      5. Screenshot: .sisyphus/evidence/task-2-design-tokens.png
    Expected Result: Design tokens applied, new font visible
  ```

  **Commit**: YES
  - Message: `feat(ui): add design system tokens, typography, glass morphism`
  - Files: `frontend/tailwind.config.ts`, `frontend/src/app/globals.css`

---

- [ ] 3. Shared Components: Navbar, PageTransition, SkeletonLoader, AmbientBackground, Layout

  **What to do**:
  - Create `frontend/src/components/Navbar.tsx`:
    - Fixed top, full-width, blur backdrop glass effect
    - Logo/brand left ("Omnibus Legal Compass" with compass icon or ⚖ emoji)
    - Navigation links center: "Tanya Jawab", "Kepatuhan", "Panduan Usaha"
    - Active link indicator (bottom border or background pill)
    - Use `usePathname()` from next/navigation for active state
    - Framer Motion: fade in on mount, hover scale on links
    - Height: 64px, z-index: 50
    
  - Create `frontend/src/components/PageTransition.tsx`:
    - Framer Motion `AnimatePresence` wrapper
    - Fade + slight slide up on page enter
    - Fade out on page exit
    - Use `motion.div` with `initial`, `animate`, `exit` props
    
  - Create `frontend/src/components/SkeletonLoader.tsx`:
    - Pulse animation skeleton blocks
    - Variants: `text` (single line), `card` (full card), `paragraph` (3-4 lines)
    - Uses Tailwind `animate-pulse` + custom shimmer gradient
    
  - Create `frontend/src/components/AmbientBackground.tsx`:
    - Subtle animated gradient orbs/mesh in background
    - Uses CSS animations (NOT canvas — keep it lightweight)
    - Absolute positioned, z-index: 0, pointer-events: none
    - Soft blue/violet/cyan gradient blobs that slowly drift
    - Only renders on hero sections, not full page
    
  - Update `frontend/src/app/layout.tsx`:
    - Import and render `<Navbar />` at top
    - Wrap `{children}` with `<PageTransition>`
    - Import Plus Jakarta Sans font
    - Add `<Toaster />` from sonner
    - Remove any existing navigation from layout if present

  **Must NOT do**:
  - Do NOT change any page content — only the shell/wrapper
  - Do NOT make the ambient background heavy/distracting
  - Do NOT use canvas or WebGL for backgrounds (too heavy)
  - Do NOT add more than 3 gradient orbs (performance)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Component architecture, Framer Motion patterns, glass morphism

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 2)
  - **Blocks**: Tasks 4, 5, 6
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `frontend/src/app/layout.tsx` — Current layout structure to modify
  - `frontend/src/app/page.tsx` — Current nav structure in homepage (lines with nav element) — extract to Navbar
  - `frontend/src/components/LoadingSpinner.tsx` — Existing loading component (may replace with SkeletonLoader)

  **API/Type References**:
  - No API changes needed — these are pure UI components

  **External References**:
  - Framer Motion AnimatePresence: `https://www.framer.com/motion/animate-presence/`
  - Sonner Toaster: `https://sonner.emilkowal.dev/`
  - Next.js usePathname: `https://nextjs.org/docs/app/api-reference/functions/use-pathname`

  **Acceptance Criteria**:
  - [ ] Navbar renders on ALL 3 pages (/, /compliance, /guidance)
  - [ ] Active link is visually distinct (highlighted)
  - [ ] Clicking nav links navigates between pages
  - [ ] Page transitions animate (fade + slide)
  - [ ] Ambient background gradient orbs visible on homepage hero
  - [ ] Toast component renders (test with temporary toast.success call)
  - [ ] `npm run build` succeeds

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Navbar present on all pages
    Tool: Playwright (playwright skill)
    Steps:
      1. Navigate to: http://localhost:3000
      2. Assert: nav element exists with "Tanya Jawab", "Kepatuhan", "Panduan Usaha" links
      3. Click: "Kepatuhan" link
      4. Wait for: navigation to /compliance
      5. Assert: Same nav element exists on compliance page
      6. Click: "Panduan Usaha" link
      7. Assert: Same nav element exists on guidance page
      8. Screenshot each page: .sisyphus/evidence/task-3-navbar-{page}.png
    Expected Result: Consistent navbar on all 3 pages

  Scenario: Page transitions animate
    Tool: Playwright (playwright skill)
    Steps:
      1. Navigate to: http://localhost:3000
      2. Click: "Kepatuhan" link
      3. Wait for: /compliance page content visible
      4. Assert: Page content rendered (heading "Pemeriksa Kepatuhan" visible)
    Expected Result: Smooth transition, no flash of unstyled content
  ```

  **Commit**: YES
  - Message: `feat(ui): add Navbar, PageTransition, SkeletonLoader, AmbientBackground components`
  - Files: `frontend/src/components/Navbar.tsx`, `frontend/src/components/PageTransition.tsx`, `frontend/src/components/SkeletonLoader.tsx`, `frontend/src/components/AmbientBackground.tsx`, `frontend/src/app/layout.tsx`

---

- [ ] 4. Q&A Page Premium Redesign

  **What to do**:
  - Completely redesign `frontend/src/app/page.tsx`:
    - **Hero section**: Large heading "Tanya Jawab Hukum Indonesia" with gradient text accent, subtitle, ambient background behind
    - **Search input**: Centered, large, rounded, with glass effect border, blue focus glow, send button with arrow icon
    - **Example questions**: Horizontal scrollable pills/chips with hover effects, Framer Motion stagger entrance
    - **Streaming toggle**: Styled as a premium toggle switch (not raw checkbox)
    - **Conversation area**: Card-based layout (like Perplexity):
      - User question: Right-aligned or full-width subtle card
      - AI answer: Full-width premium card with:
        - Typing animation (blinking cursor during streaming)
        - Smooth text reveal
        - Citation badges inline `[1]`, `[2]` that highlight on hover
        - Expandable source cards below answer
        - Confidence score as elegant progress bar
        - Grounding score visualization
        - Hallucination risk indicator (colored badge)
    - All elements use Framer Motion for entrance animations
    
  - Update `frontend/src/components/StreamingAnswerCard.tsx`:
    - Premium card with rounded-xl, subtle shadow, glass border
    - Typing cursor animation (blinking `|` at end during stream)
    - Answer text with proper prose typography
    - Citation chips that expand to show source details
    - Confidence meter (horizontal bar with gradient fill)
    - Grounding score progress bar (from ML pipeline)
    - Ungrounded claims warning section (if any)
    - Framer Motion: card slides up, content fades in sequentially
    
  - Update `frontend/src/components/CitationCard.tsx`:
    - Glass morphism card style
    - Compact view: [number] + document title + score badge
    - Expanded view: full text excerpt, metadata
    - Click to expand/collapse with Framer Motion layout animation
    - Score displayed as colored badge (green/yellow/red)
    
  - Update `frontend/src/components/QuestionInput.tsx` (if exists, or create):
    - Large, centered input with glass effect
    - Animated placeholder text
    - Blue glow on focus
    - Submit button with icon, loading spinner state
    - Enter key to submit

  - Remove old navigation from page.tsx (now in Navbar)

  **Must NOT do**:
  - Do NOT change API call logic or response handling
  - Do NOT modify the streaming SSE logic
  - Do NOT remove the streaming toggle functionality
  - Do NOT change how citations are parsed from the response
  - Do NOT break the existing question → answer flow

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`, `typography`]
    - `frontend-ui-ux`: Card layout, glass morphism, Framer Motion orchestration
    - `typography`: Answer text typography, citation formatting

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 5, 6)
  - **Blocks**: Task 7
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - `frontend/src/app/page.tsx` — Current Q&A page (COMPLETE FILE — redesign all rendering, keep API logic)
  - `frontend/src/components/StreamingAnswerCard.tsx` — Current streaming card (redesign styling)
  - `frontend/src/components/CitationCard.tsx` — Current citation display (redesign)
  - `frontend/src/components/ChatMessage.tsx` — Current chat message component
  - `frontend/src/components/QuestionInput.tsx` — Current question input (if exists)

  **API/Type References**:
  - `frontend/src/lib/api.ts` — API types (QuestionResponse, ValidationResult, etc.) — DO NOT modify, only consume
  - Streaming endpoint: `POST /api/ask/stream` — SSE response format

  **Acceptance Criteria**:
  - [ ] Hero section has gradient text and ambient background
  - [ ] Search input has glass effect, blue focus glow
  - [ ] Example questions animate in with stagger
  - [ ] Typing a question and pressing Enter/button triggers API call
  - [ ] Streaming answer shows typing cursor animation
  - [ ] Answer card has confidence bar, citation badges
  - [ ] Citation cards expand/collapse on click
  - [ ] Streaming toggle still works
  - [ ] `npm run build` succeeds

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Full Q&A flow with premium UI
    Tool: Playwright (playwright skill)
    Preconditions: Backend running on port 8000
    Steps:
      1. Navigate to: http://localhost:3000
      2. Assert: Hero heading visible with "Tanya Jawab Hukum"
      3. Assert: Search input visible with glass styling
      4. Fill: search input → "Apa itu NIB?"
      5. Click: submit button
      6. Wait for: answer card to appear (timeout: 30s)
      7. Assert: Answer text contains content (length > 50 chars)
      8. Assert: Citation badges visible ([1], [2], etc.)
      9. Assert: Confidence indicator visible
      10. Screenshot: .sisyphus/evidence/task-4-qa-answer.png
    Expected Result: Beautiful answer card with citations and confidence

  Scenario: Example questions work
    Tool: Playwright (playwright skill)
    Steps:
      1. Navigate to: http://localhost:3000
      2. Click: first example question chip
      3. Wait for: answer to appear
      4. Assert: Answer card rendered
    Expected Result: Clicking example question triggers query
  ```

  **Commit**: YES
  - Message: `feat(ui): premium Q&A page with card-based streaming, typing animation, citations`
  - Files: `frontend/src/app/page.tsx`, `frontend/src/components/StreamingAnswerCard.tsx`, `frontend/src/components/CitationCard.tsx`, `frontend/src/components/ChatMessage.tsx`, `frontend/src/components/QuestionInput.tsx`

---

- [ ] 5. Compliance Page Premium Redesign

  **What to do**:
  - Completely redesign `frontend/src/app/compliance/page.tsx`:
    - **Hero section**: Heading "Pemeriksa Kepatuhan" with icon, subtitle, ambient background
    - **Input tabs**: Premium segmented control (text/PDF toggle) with Framer Motion layout animation on active indicator
    - **Text input**: Large textarea with glass border, character count, placeholder
    - **PDF upload**: Drag-and-drop zone with dashed border, file icon, "drop or click" text, file preview after upload
    - **Submit button**: Full-width gradient button with loading state (spinner + "Menganalisis...")
    - **Results display**: Premium compliance result card:
      - Status badge (Compliant/Partially Compliant/Non-Compliant) with color coding
      - Issues list with warning icons
      - Recommendations with checkmark icons
      - Citations with same CitationCard component from Q&A
      - Framer Motion: results card slides up, items stagger in
    - **Skeleton loader**: Show while waiting for compliance analysis
    - Remove "Kembali ke Beranda" link (navbar handles navigation now)

  **Must NOT do**:
  - Do NOT change the compliance API call logic
  - Do NOT modify how PDF files are sent to backend
  - Do NOT change the response handling/parsing
  - Do NOT remove text/PDF mode toggle functionality

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Form design, file upload UX, result card layout

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 4, 6)
  - **Blocks**: Task 7
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - `frontend/src/app/compliance/page.tsx` — Current compliance page (FULL FILE — redesign rendering, keep API logic)
  - `frontend/src/components/StreamingAnswerCard.tsx` — Reuse card patterns for results display

  **API/Type References**:
  - `frontend/src/lib/api.ts` — ComplianceResponse type — DO NOT modify

  **Acceptance Criteria**:
  - [ ] Hero section matches Q&A page design language
  - [ ] Text/PDF tabs are premium segmented control
  - [ ] Textarea has glass effect styling
  - [ ] PDF drag-drop zone is visually clear
  - [ ] Submit button has gradient + loading state
  - [ ] Compliance results display with status badge, issues, recommendations
  - [ ] Skeleton loader shown during analysis
  - [ ] No "Kembali ke Beranda" link (navbar is global now)
  - [ ] `npm run build` succeeds

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Compliance check with text input
    Tool: Playwright (playwright skill)
    Preconditions: Backend running on port 8000
    Steps:
      1. Navigate to: http://localhost:3000/compliance
      2. Assert: Navbar visible at top
      3. Assert: Text/PDF toggle visible
      4. Fill: textarea → "Perusahaan perdagangan umum dengan SIUP dan NIB"
      5. Click: submit button
      6. Wait for: results card (timeout: 30s)
      7. Assert: Status badge visible (any compliance status)
      8. Screenshot: .sisyphus/evidence/task-5-compliance-result.png
    Expected Result: Compliance results with premium card layout

  Scenario: PDF upload zone visible in PDF mode
    Tool: Playwright (playwright skill)
    Steps:
      1. Navigate to: http://localhost:3000/compliance
      2. Click: "Upload PDF" tab/toggle
      3. Assert: Drag-drop upload zone visible
      4. Screenshot: .sisyphus/evidence/task-5-pdf-upload.png
    Expected Result: Upload zone rendered with correct styling
  ```

  **Commit**: YES
  - Message: `feat(ui): premium compliance checker with segmented tabs, drag-drop, result cards`
  - Files: `frontend/src/app/compliance/page.tsx`

---

- [ ] 6. Guidance Page Premium Redesign

  **What to do**:
  - Completely redesign `frontend/src/app/guidance/page.tsx`:
    - **Hero section**: Heading "Panduan Pendirian Usaha" with icon, subtitle, ambient background
    - **Business type selector**: Premium card grid (NOT radio buttons):
      - 5 cards in 2-3 column grid
      - Each card: icon + business type name + short description
      - Selected state: blue border, subtle blue background, checkmark
      - Framer Motion: hover scale, selection animation
    - **Optional fields**: Side-by-side inputs (location + industry) with glass styling
    - **Submit button**: Gradient button matching compliance page
    - **Results display**: Step-by-step guidance card:
      - Numbered steps with timeline/stepper visualization
      - Each step: title, description, estimated time, requirements
      - Required permits section with organized cards
      - Citations section
      - Framer Motion: steps stagger in from left
    - **Skeleton loader**: Show while waiting for guidance
    - Remove "Kembali ke Beranda" link (navbar handles navigation now)

  **Must NOT do**:
  - Do NOT change guidance API call logic
  - Do NOT remove any of the 5 business types (PT, CV, UD, Koperasi, Yayasan)
  - Do NOT remove location/industry optional fields
  - Do NOT change how the guidance response is processed

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Card grid selector, stepper/timeline UX, form design

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 4, 5)
  - **Blocks**: Task 7
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - `frontend/src/app/guidance/page.tsx` — Current guidance page (FULL FILE — redesign rendering, keep API logic)

  **API/Type References**:
  - `frontend/src/lib/api.ts` — GuidanceResponse type — DO NOT modify

  **Acceptance Criteria**:
  - [ ] Hero section matches other pages' design language
  - [ ] Business types displayed as selectable card grid (not radio buttons)
  - [ ] Selecting a card shows visual selection state (blue border + check)
  - [ ] Location and industry fields have glass styling
  - [ ] Submit button has gradient + loading state
  - [ ] Guidance results show numbered steps with timeline
  - [ ] Required permits display as organized cards
  - [ ] Skeleton loader shown during generation
  - [ ] `npm run build` succeeds

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Business guidance flow
    Tool: Playwright (playwright skill)
    Preconditions: Backend running on port 8000
    Steps:
      1. Navigate to: http://localhost:3000/guidance
      2. Assert: Navbar visible at top
      3. Assert: 5 business type cards visible (PT, CV, UD, Koperasi, Yayasan)
      4. Click: "PT (Perseroan Terbatas)" card
      5. Assert: Card shows selected state (blue border or background)
      6. Fill: location input → "Jakarta"
      7. Fill: industry input → "Teknologi"
      8. Click: submit button
      9. Wait for: guidance results (timeout: 30s)
      10. Assert: Steps visible with numbered timeline
      11. Screenshot: .sisyphus/evidence/task-6-guidance-result.png
    Expected Result: Premium guidance with step-by-step timeline

  Scenario: Card selection interaction
    Tool: Playwright (playwright skill)
    Steps:
      1. Navigate to: http://localhost:3000/guidance
      2. Click: "CV" card
      3. Assert: CV card has selection indicator
      4. Click: "PT" card
      5. Assert: PT card has selection indicator, CV does not
    Expected Result: Single selection with visual feedback
  ```

  **Commit**: YES
  - Message: `feat(ui): premium guidance page with card selector, timeline stepper`
  - Files: `frontend/src/app/guidance/page.tsx`

---

- [ ] 7. Final Polish: Toast Integration, Build Verification, Screenshots

  **What to do**:
  - **Toast integration**: Add `toast.success()` and `toast.error()` calls throughout:
    - Q&A: toast.error on API failure, toast.success("Jawaban ditemukan") on completion (subtle)
    - Compliance: toast.error on upload/check failure
    - Guidance: toast.error on guidance generation failure
  - **Micro-interactions audit**: Ensure ALL interactive elements have:
    - Hover state (subtle scale or shadow change)
    - Focus state (blue glow ring)
    - Active/pressed state (slight scale down)
    - Transition: `transition-all duration-200`
  - **Mobile responsiveness check**: Test at 375px width
    - Navbar collapses to hamburger menu or horizontal scroll
    - Cards stack vertically
    - Inputs are full-width
    - Text sizes adjust
  - **Performance check**: Ensure no layout shift, smooth 60fps animations
  - **Full build verification**: `npm run build` — zero errors
  - **Screenshot all pages**: Final screenshots of all 3 pages for evidence

  **Must NOT do**:
  - Do NOT add any new features not in the spec
  - Do NOT make toasts overly verbose or annoying (subtle, helpful only)
  - Do NOT add mobile hamburger menu if it's complex — horizontal scroll nav is acceptable

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`, `playwright`]
    - `frontend-ui-ux`: Micro-interaction polish, responsive design
    - `playwright`: Full visual QA across all pages

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final, solo)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 4, 5, 6

  **References**:

  **Pattern References**:
  - All page files from Tasks 4-6
  - `frontend/src/app/layout.tsx` — Toaster component placement

  **External References**:
  - Sonner toast API: `https://sonner.emilkowal.dev/toast`

  **Acceptance Criteria**:
  - [ ] Toast appears on API error (test by stopping backend temporarily)
  - [ ] All buttons have hover/focus/active states
  - [ ] Pages render correctly at 375px mobile width
  - [ ] `npm run build` succeeds with zero errors
  - [ ] All 3 pages look premium and consistent

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Full visual audit - all pages
    Tool: Playwright (playwright skill)
    Steps:
      1. Set viewport: 1440x900
      2. Navigate to: http://localhost:3000
      3. Screenshot: .sisyphus/evidence/task-7-final-qa-desktop.png
      4. Navigate to: http://localhost:3000/compliance
      5. Screenshot: .sisyphus/evidence/task-7-final-compliance-desktop.png
      6. Navigate to: http://localhost:3000/guidance
      7. Screenshot: .sisyphus/evidence/task-7-final-guidance-desktop.png
      8. Set viewport: 375x812
      9. Navigate to: http://localhost:3000
      10. Screenshot: .sisyphus/evidence/task-7-final-qa-mobile.png
      11. Navigate to: http://localhost:3000/compliance
      12. Screenshot: .sisyphus/evidence/task-7-final-compliance-mobile.png
      13. Navigate to: http://localhost:3000/guidance
      14. Screenshot: .sisyphus/evidence/task-7-final-guidance-mobile.png
    Expected Result: All pages look premium at both breakpoints

  Scenario: Build verification
    Tool: Bash
    Steps:
      1. cd frontend && npm run build
      2. Assert: Exit code 0
      3. Assert: No TypeScript errors in output
    Expected Result: Clean build with zero errors
  ```

  **Commit**: YES
  - Message: `feat(ui): polish micro-interactions, toast notifications, mobile responsiveness`
  - Files: All modified files

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1+2 | `feat(ui): install deps, add design system tokens` | package.json, tailwind.config.ts, globals.css | npm run build |
| 3 | `feat(ui): add Navbar, PageTransition, SkeletonLoader, AmbientBackground` | components/*.tsx, layout.tsx | npm run build |
| 4 | `feat(ui): premium Q&A page with card-based streaming` | page.tsx, StreamingAnswerCard.tsx, CitationCard.tsx | npm run build |
| 5 | `feat(ui): premium compliance checker with segmented tabs` | compliance/page.tsx | npm run build |
| 6 | `feat(ui): premium guidance page with card selector, timeline` | guidance/page.tsx | npm run build |
| 7 | `feat(ui): polish micro-interactions, toasts, mobile` | various | npm run build |

---

## Success Criteria

### Verification Commands
```bash
cd frontend && npm run build   # Expected: Exit code 0, no errors
```

### Final Checklist
- [ ] All 3 pages render with Light Luxurious theme
- [ ] Consistent top navbar across all pages
- [ ] Framer Motion animations on all interactive elements
- [ ] Card-based Q&A with streaming typing animation
- [ ] Skeleton loaders during data fetching
- [ ] Page transitions between routes
- [ ] Ambient gradient background on hero sections
- [ ] Toast notifications on errors
- [ ] Mobile responsive (375px+)
- [ ] Zero TypeScript build errors
- [ ] All existing functionality preserved (streaming, PDF upload, business type selection)
