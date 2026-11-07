from src.quality_gate import check

def test_quality_gate_passes_good_candidate(tmp_path, monkeypatch):
    import yaml
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    Path("config.yaml").write_text(yaml.safe_dump({"quality_gate":{"accuracy":0.9,"f1_macro":0.9}}))
    assert check({"accuracy":0.99,"f1_macro":0.98})
