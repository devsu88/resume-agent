import pytest

from resume_agent.profile import load_profile


def make_pdf(text):
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        None,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET"
    objects[3] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    return out + f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


def test_text_resume_summary_and_name(tmp_path):
    (tmp_path / "resume.txt").write_text("Skills: Python\n")
    (tmp_path / "summary.txt").write_text("Likes data.\n")
    (tmp_path / "profile.json").write_text('{"name": "Sam Doe"}')
    profile = load_profile(tmp_path)
    assert (profile.name, profile.summary, profile.resume) == ("Sam Doe", "Likes data.", "Skills: Python")


def test_summary_and_name_are_optional(tmp_path):
    (tmp_path / "resume.md").write_text("# CV")
    profile = load_profile(tmp_path)
    assert profile.name == "the candidate" and profile.summary == ""


def test_pdf_resumes_are_read(tmp_path):
    (tmp_path / "resume.pdf").write_bytes(make_pdf("Hello Resume"))
    assert "Hello Resume" in load_profile(tmp_path).resume


def test_a_folder_without_a_resume_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="resume"):
        load_profile(tmp_path)
