"""Generate paper/appendix/A_prompts.tex from cgsc/prompts/."""
import os


def clean_prompt(text: str) -> str:
    lines = text.splitlines()
    # Leave out comment lines (e.g. lines starting with #)
    filtered = [l for l in lines if not l.strip().startswith("#")]
    content = "\n".join(filtered).strip("\n")
    return content


def main():
    prompts_dir = "cgsc/prompts"

    sections = [
        ("A0 GSM8K", os.path.join(prompts_dir, "a0_gsm8k.txt"), None),
        ("A0 HotpotQA", os.path.join(prompts_dir, "a0_hotpotqa.txt"), None),
        ("Verbalized confidence", os.path.join(prompts_dir, "verb.txt"), None),
        (
            "Self-verification P(True)",
            os.path.join(prompts_dir, "verif_gsm8k.txt"),
            "The assistant turn is prefilled with ``The proposed answer is: ('' and the next-token probabilities of ``A'' and ``B'' are read."
        ),
        ("Revision", os.path.join(prompts_dir, "revision.txt"), None),
        (
            "IoE",
            os.path.join(prompts_dir, "ioe.txt"),
            'Adapted from Li et al.~\\cite{li2024confidence}.'
        ),
    ]

    out_lines = [
        r"\section{Prompts}",
        r"\label{app:prompts}",
        r"All prompts are given verbatim; \{...\} marks a placeholder filled per question. Each prompt is sent as a user turn with the model's own chat template.",
        ""
    ]

    for title, filepath, note in sections:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
        cleaned = clean_prompt(raw)
        # \paragraph is a run-in heading; end its line so the framed listing starts on its own line
        out_lines.append(f"\\paragraph{{{title}}}\\mbox{{}}\\par\\nopagebreak")
        out_lines.append(r"\begin{lstlisting}")
        out_lines.append(cleaned)
        out_lines.append(r"\end{lstlisting}")
        if note:
            out_lines.append(note)
        out_lines.append("")

    out_path = "paper/appendix/A_prompts.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"Generated {out_path} successfully.")


if __name__ == "__main__":
    main()
