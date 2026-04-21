"""Patch existing parallel inference HTML reports with the improved diagram."""
import os

OLD = """      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    </head>
    <body>
      <h1>Parallel Inference Attack - Exploration Report</h1>

      <div class="flowchart">
        <h2>Pipeline Architecture</h2>
        <div class="mermaid">
          graph TD
            A["Original Text"] --> B["Attack A: Analytical<br/>(step-by-step deduction)"]
            A --> C["Attack B: Sociolinguistic<br/>(cultural cues, style)"]
            B --> D["Merge & Compare Inferences"]
            C --> D
            D --> E["Anonymization<br/>(informed by merged inferences)"]
            E --> F["Utility Scoring"]
            F --> G["Attack A on anonymized text"]
            F --> H["Attack B on anonymized text"]
            G --> I["Compare: Was attack defeated?"]
            H --> I
        </div>
      </div>"""

NEW = """      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      <script>mermaid.initialize({ startOnLoad: true, theme: 'base', themeVariables: { primaryColor: '#4b86b4', primaryTextColor: '#fff', primaryBorderColor: '#2a4d69', lineColor: '#555', secondaryColor: '#f0f5f9', tertiaryColor: '#fff' } });</script>
    </head>
    <body>
      <h1>Parallel Inference Attack - Exploration Report</h1>

      <div class="flowchart">
        <h2>Pipeline Architecture</h2>
        <div class="mermaid">
          flowchart TD
            P(["Reddit Profiles"])

            P --> A1
            P --> B1

            subgraph STAGE1["Stage 1 — Parallel Inference on Original Text"]
              A1["Attack A<br/><b>Analytical</b><br/>Step-by-step logical deduction<br/>from explicit text evidence"]
              B1["Attack B<br/><b>Sociolinguistic</b><br/>Implicit cues: style, vocabulary,<br/>cultural references, slang"]
              A1 --> M1
              B1 --> M1
              M1["Merge<br/>Union guesses · Pick higher certainty<br/>Compute agreement"]
            end

            M1 -->|"merged inferences stored into profiles"| ANON

            subgraph STAGE2["Stage 2 — Anonymization"]
              ANON["Anonymizer<br/>Informed by merged inferences<br/>Must defeat both attack surfaces"]
              ANON --> UTIL
              UTIL["Utility Scorer<br/>Measures text quality loss"]
            end

            UTIL --> A2
            UTIL --> B2

            subgraph STAGE4["Stage 4 — Parallel Inference on Anonymized Text"]
              A2["Attack A<br/><b>Analytical</b>"]
              B2["Attack B<br/><b>Sociolinguistic</b>"]
              A2 --> M2
              B2 --> M2
              M2["Merge"]
            end

            M2 --> CMP

            subgraph ANALYSIS["Analysis — Compare Before vs After Anonymization"]
              CMP["Certainty drop per attack<br/>Attacks defeated: A only / B only / Both<br/>Agreement stats before and after"]
            end

            classDef input fill:#2a4d69,color:#fff,stroke:#1a3049
            classDef attack fill:#4b86b4,color:#fff,stroke:#2a4d69
            classDef merge fill:#e8a838,color:#fff,stroke:#b07820
            classDef process fill:#5ba55b,color:#fff,stroke:#3d7a3d
            classDef analysis fill:#9b59b6,color:#fff,stroke:#7d3c98

            class P input
            class A1,B1,A2,B2 attack
            class M1,M2 merge
            class ANON,UTIL process
            class CMP analysis
        </div>
      </div>"""

root = os.path.dirname(os.path.abspath(__file__))
patched = 0
for dirpath, _, files in os.walk(root):
    for fname in files:
        if fname == "parallel_inference_report.html":
            path = os.path.join(dirpath, fname)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            if OLD in content:
                content = content.replace(OLD, NEW)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Patched: {path}")
                patched += 1
            else:
                print(f"Already up to date or pattern not found: {path}")

print(f"\nDone. {patched} file(s) patched.")
