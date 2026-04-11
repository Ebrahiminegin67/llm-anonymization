import json
import difflib
from pathlib import Path
from src.reddit.reddit_utils import load_data

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "anonymized_results" / "test_real_profile"
OUTPUT_HTML = ROOT / "anonymization_report.html"
USERNAME = "31male"

def load_profile(path: Path, username: str):
    for profile in load_data(str(path)):
        if profile.username == username:
            return profile
    raise ValueError(f"Username {username} not found in {path}")

def comments_text(annotated):
    return "\n\n".join(
        [f"{i+1}. {comment.text}" for i, comment in enumerate(annotated.comments)]
    )

def json_block(title: str, obj):
    return f"""
    <div class="block">
      <h3>{title}</h3>
      <pre>{json.dumps(obj, indent=2, ensure_ascii=False)}</pre>
    </div>
    """

def make_diff_table(text_a: str, text_b: str, label_a: str, label_b: str):
    a_lines = text_a.splitlines()
    b_lines = text_b.splitlines()
    diff = difflib.HtmlDiff(tabsize=4, wrapcolumn=80)
    return diff.make_table(a_lines, b_lines, fromdesc=label_a, todesc=label_b, context=True, numlines=1)

def render_stage(title: str, text: str, diff_html: str, metadata_html: str):
    return f"""
    <section class="stage">
      <h2>{title}</h2>
      <div class="text-block">
        <h3>Text</h3>
        <pre>{text}</pre>
      </div>
      <div class="diff-block">
        <h3>Diff</h3>
        {diff_html}
      </div>
      {metadata_html}
    </section>
    """

def main():
    orig_anon_profile = load_profile(RESULTS / "anonymized_1.jsonl", USERNAME)
    inf0 = load_profile(RESULTS / "inference_0.jsonl", USERNAME)
    inf1 = load_profile(RESULTS / "inference_1.jsonl", USERNAME)
    inf2 = load_profile(RESULTS / "inference_2.jsonl", USERNAME)
    util0 = load_profile(RESULTS / "utility_0.jsonl", USERNAME)
    util1 = load_profile(RESULTS / "utility_1.jsonl", USERNAME)

    original = comments_text(orig_anon_profile.comments[0])
    anonymized_0 = comments_text(orig_anon_profile.comments[1])
    anonymized_1 = comments_text(orig_anon_profile.comments[2])

    diff_orig_anon0 = make_diff_table(original, anonymized_0, "Original", "Anonymized 0")
    diff_anon0_anon1 = make_diff_table(anonymized_0, anonymized_1, "Anonymized 0", "Anonymized 1")

    metadata0 = json_block("Inference 0 (original)", inf0.comments[0].predictions)
    metadata1 = json_block("Inference 1 (after anonymized_0)", inf1.comments[0].predictions)
    metadata2 = json_block("Inference 2 (after anonymized_1)", inf2.comments[0].predictions)
    utility0 = json_block("Utility 0", util0.comments[0].utility)
    utility1 = json_block("Utility 1", util1.comments[0].utility)

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>Anonymization Pipeline Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
          h1 {{ color: #2a4d69; }}
          h2 {{ color: #4b86b4; }}
          h3 {{ color: #adcbe3; }}
          .stage {{ margin-bottom: 40px; border: 1px solid #d1d1d1; padding: 16px; border-radius: 8px; background: #fbfbfb; }}
          .text-block, .diff-block {{ margin-bottom: 20px; }}
          pre {{ background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; }}
          .block {{ margin-top: 16px; }}
          table.diff {{ width: 100%; border-collapse: collapse; font-size: 0.95em; }}
          table.diff th {{ padding: 8px; background: #2a4d69; color: white; }}
          table.diff td {{ padding: 6px; vertical-align: top; }}
          .diff_header {{ background: #4b86b4; color: white; }}
          .diff_next {{ background: #f3f3f3; }}
          .diff_add {{ background: #e2ffe2; }}
          .diff_chg {{ background: #fff5b1; }}
          .diff_sub {{ background: #ffe2e2; }}
          .flowchart-container {{ display: flex; gap: 30px; align-items: flex-start; }}
          .flowchart-container > div {{ flex: 1; }}
          .explanation {{ padding: 16px; background: #f9f9f9; border-left: 4px solid #4b86b4; line-height: 1.6; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      </head>
      <body>
        <h1>Anonymization Process Report</h1>
        <p>Dataset: <strong>test_profile.jsonl</strong></p>
        <p>Results folder: <strong>anonymized_results/test_real_profile</strong></p>

        <section class="stage">
          <h2>Pipeline Flowchart</h2>
          <div class="flowchart-container">
            <div>
              <div class="mermaid">
                graph TD
                A["Original Text"] --> B["Inference (GPT-3.5)"]
                B --> C["Anonymization 1"]
                C --> D["Utility Scoring 1"]
                D --> E["Anonymization 2"]
                E --> F["Utility Scoring 2"]
              </div>
            </div>
            <div class="explanation">
              <h3>Process Overview</h3>
              <p>The anonymization pipeline starts with the original text from Reddit comments. First, an inference model (GPT-3.5) analyzes the text to identify potentially sensitive information. Then, an anonymization model replaces or removes identifiable details to protect privacy. Utility scoring evaluates how well the anonymized text retains meaning, readability, and avoids hallucinations. This process can iterate multiple times, as shown in the stages above, to achieve better anonymization while maintaining utility.</p>
            </div>
          </div>
        </section>

        <section class="stage">
          <h2>Stage 0: Original text + initial inference</h2>
          <div class="text-block">
            <h3>Original text</h3>
            <pre>{original}</pre>
          </div>
          {metadata0}
        </section>

        {render_stage("Stage 1: First anonymization", anonymized_0, diff_orig_anon0, utility0 + metadata1)}
        {render_stage("Stage 2: Second anonymization", anonymized_1, diff_anon0_anon1, utility1 + metadata2)}

        <p>The anonymization pipeline starts with the original text from Reddit comments. First, an inference model (GPT-3.5) analyzes the text to identify potentially sensitive information. Then, an anonymization model replaces or removes identifiable details to protect privacy. Utility scoring evaluates how well the anonymized text retains meaning, readability, and avoids hallucinations. This process can iterate multiple times, as shown in the stages above, to achieve better anonymization while maintaining utility.</p>
      </body>
    </html>
    """

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print("Wrote HTML report to", OUTPUT_HTML)

if __name__ == "__main__":
    main()