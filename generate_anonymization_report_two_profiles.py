import json
import difflib
from pathlib import Path
from src.reddit.reddit_utils import load_data

ROOT = Path(__file__).resolve().parent
RESULTS_GPT35 = ROOT / "anonymized_results" / "test_real_profile"
RESULTS_GPT4O = ROOT / "anonymized_results" / "test_real_profile_gpt4o"
TEST_DATA = ROOT / "data" / "base_inferences" / "synthetic" / "test_profile.jsonl"
OUTPUT_HTML = ROOT / "anonymization_report.html"

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

def highlight_changes(text_a: str, text_b: str) -> tuple:
    """Create highlighted versions of text showing changes"""
    a_words = text_a.split()
    b_words = text_b.split()
    
    matcher = difflib.SequenceMatcher(None, a_words, b_words)
    
    a_highlighted = []
    b_highlighted = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            a_highlighted.extend(a_words[i1:i2])
            b_highlighted.extend(b_words[j1:j2])
        elif tag == 'delete':
            a_highlighted.extend([f'<span style="background: #ffe2e2; padding: 2px 4px; border-radius: 3px;">{word}</span>' for word in a_words[i1:i2]])
        elif tag == 'insert':
            b_highlighted.extend([f'<span style="background: #e2ffe2; padding: 2px 4px; border-radius: 3px;">{word}</span>' for word in b_words[j1:j2]])
        elif tag == 'replace':
            a_highlighted.extend([f'<span style="background: #ffe2e2; padding: 2px 4px; border-radius: 3px;">{word}</span>' for word in a_words[i1:i2]])
            b_highlighted.extend([f'<span style="background: #e2ffe2; padding: 2px 4px; border-radius: 3px;">{word}</span>' for word in b_words[j1:j2]])
    
    return ' '.join(a_highlighted), ' '.join(b_highlighted)

def render_stage_highlighted(title: str, text_original: str, text_anonymized: str, metadata_html: str):
    """Render stage with highlighted changes side-by-side"""
    a_highlighted, b_highlighted = highlight_changes(text_original, text_anonymized)
    
    return f"""
    <section class="stage">
      <h3>{title}</h3>
      <div class="comparison-texts">
        <div class="text-column">
          <h4>Original</h4>
          <div class="highlighted-text">{a_highlighted}</div>
        </div>
        <div class="text-column">
          <h4>Anonymized</h4>
          <div class="highlighted-text">{b_highlighted}</div>
        </div>
      </div>
      {metadata_html}
    </section>
    """

def generate_profile_report(username: str, results_dir: Path, model_name: str):
    try:
        orig_anon_profile = load_profile(results_dir / "anonymized_1.jsonl", username)
        inf0 = load_profile(results_dir / "inference_0.jsonl", username)
        inf1 = load_profile(results_dir / "inference_1.jsonl", username)
        inf2 = load_profile(results_dir / "inference_2.jsonl", username)
        util0 = load_profile(results_dir / "utility_0.jsonl", username)
        util1 = load_profile(results_dir / "utility_1.jsonl", username)

        original = comments_text(orig_anon_profile.comments[0])
        anonymized_0 = comments_text(orig_anon_profile.comments[1])
        anonymized_1 = comments_text(orig_anon_profile.comments[2])

        metadata0 = json_block("Inference 0 (original)", inf0.comments[0].predictions)
        metadata1 = json_block("Inference 1 (after anonymized_0)", inf1.comments[0].predictions)
        metadata2 = json_block("Inference 2 (after anonymized_1)", inf2.comments[0].predictions)
        utility0 = json_block("Utility 0", util0.comments[1].utility)
        utility1 = json_block("Utility 1", util1.comments[1].utility)

        profile_html = f"""
        <div class="model-report">
          <span class="model-badge">{model_name}</span>
          
          <section class="stage">
            <h4>Stage 0: Original text + initial inference</h4>
            <div class="text-block">
              <h5>Original text</h5>
              <pre>{original}</pre>
            </div>
            {metadata0}
          </section>

          {render_stage_highlighted("Stage 1: First anonymization", original, anonymized_0, utility0 + metadata1)}
          {render_stage_highlighted("Stage 2: Second anonymization", anonymized_0, anonymized_1, utility1 + metadata2)}
        </div>
        """
        return profile_html
    except Exception as e:
        return f"<div class='error'><p>Error processing {username} with {model_name}: {str(e)}</p></div>"

def main():
    # Get all usernames from test_profile.jsonl
    usernames = [profile.username for profile in load_data(str(TEST_DATA))]
    
    # Generate reports for each profile and both models
    profiles_html = ""
    for username in usernames:
        gpt35_report = generate_profile_report(username, RESULTS_GPT35, "GPT-3.5-Turbo")
        gpt4o_report = generate_profile_report(username, RESULTS_GPT4O, "GPT-4o")
        
        profiles_html += f"""
        <div class="profile-section">
          <h2 class="profile-title">Profile: {username}</h2>
          <div class="model-comparison">
            {gpt35_report}
            <hr class="model-separator"/>
            {gpt4o_report}
          </div>
        </div>
        <hr class="profile-separator"/>
        """

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>Anonymization Pipeline Report - Model Comparison</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
          h1 {{ color: #2a4d69; }}
          h2 {{ color: #4b86b4; }}
          h3 {{ color: #4b86b4; }}
          h4 {{ color: #4b86b4; }}
          h5 {{ color: #adcbe3; }}
          .stage {{ margin-bottom: 40px; border: 1px solid #d1d1d1; padding: 16px; border-radius: 8px; background: #fbfbfb; }}
          .text-block, .diff-block {{ margin-bottom: 20px; }}
          pre {{ background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; max-height: 400px; overflow-y: auto; }}
          .block {{ margin-top: 16px; }}
          .flowchart-container {{ display: flex; gap: 30px; align-items: flex-start; }}
          .flowchart-container > div {{ flex: 1; }}
          .explanation {{ padding: 16px; background: #f9f9f9; border-left: 4px solid #4b86b4; line-height: 1.6; }}
          .profile-section {{ margin-bottom: 80px; }}
          .profile-title {{ color: #2a4d69; border-bottom: 3px solid #4b86b4; padding-bottom: 10px; margin-bottom: 20px; font-size: 1.5em; }}
          .profile-separator {{ margin: 60px 0; border: 2px solid #2a4d69; }}
          .model-comparison {{ margin: 30px 0; }}
          .model-report {{ padding: 20px; background: #f0f5f9; border-radius: 8px; margin-bottom: 30px; border-left: 4px solid #4b86b4; }}
          .model-badge {{ display: inline-block; background: #4b86b4; color: white; padding: 8px 16px; border-radius: 4px; margin-bottom: 16px; font-weight: bold; }}
          .model-separator {{ margin: 50px 0; border: 2px dashed #4b86b4; padding: 20px 0; background: #f9f9f9; }}
          .comparison-note {{ background: #fff9e6; padding: 12px; border-left: 4px solid #ff9800; margin-bottom: 20px; }}
          .comparison-texts {{ display: flex; gap: 30px; margin-bottom: 20px; }}
          .text-column {{ flex: 1; }}
          .text-column h4 {{ margin-top: 0; color: #4b86b4; }}
          .highlighted-text {{ background: #f4f4f4; padding: 12px; border-radius: 6px; line-height: 1.8; word-wrap: break-word; border: 1px solid #d1d1d1; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      </head>
      <body>
        <h1>Anonymization Pipeline Report - Model Comparison</h1>
        <p>Dataset: <strong>test_profile.jsonl</strong></p>
        <p>Number of profiles: <strong>{len(usernames)}</strong></p>
        <p style="color: #2a4d69; font-weight: bold; font-size: 1.1em;">Comparing: GPT-3.5-Turbo vs GPT-4o</p>

        <div class="comparison-note">
          <strong>Color Legend:</strong>
          <ul>
            <li><span style="background: #ffe2e2; padding: 2px 4px; border-radius: 3px;">Pink/Red</span> = Removed or changed text</li>
            <li><span style="background: #e2ffe2; padding: 2px 4px; border-radius: 3px;">Green</span> = Added or new text</li>
          </ul>
        </div>

        <section class="stage">
          <h2>Pipeline Flowchart</h2>
          <div class="flowchart-container">
            <div>
              <div class="mermaid">
                graph TD
                A["Original Text"] --> B["Inference (Model)"]
                B --> C["Anonymization 1"]
                C --> D["Utility Scoring 1"]
                D --> E["Anonymization 2"]
                E --> F["Utility Scoring 2"]
              </div>
            </div>
            <div class="explanation">
              <h3>Process Overview</h3>
              <p>The anonymization pipeline starts with the original text from Reddit comments. First, an inference model analyzes the text to identify potentially sensitive information. Then, an anonymization model replaces or removes identifiable details to protect privacy. Utility scoring evaluates how well the anonymized text retains meaning, readability, and avoids hallucinations. This process can iterate multiple times.</p>
              <p><strong>Below:</strong> Results from both GPT-3.5-Turbo and GPT-4o models applied to the same profiles, allowing direct comparison of their anonymization strategies and effectiveness.</p>
            </div>
          </div>
        </section>

        {profiles_html}

      </body>
    </html>
    """

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote HTML report comparing both models for {len(usernames)} profiles to {OUTPUT_HTML}")

if __name__ == "__main__":
    main()