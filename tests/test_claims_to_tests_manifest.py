from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = Path("claims-to-tests.json")


def test_claims_manifest_structure_and_references_exist() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    claims = payload.get("claims", [])
    assert isinstance(claims, list)
    assert claims

    seen_ids: set[str] = set()
    for claim in claims:
        assert isinstance(claim, dict)
        claim_id = str(claim.get("id", "")).strip()
        assert claim_id
        assert claim_id not in seen_ids
        seen_ids.add(claim_id)

        statement = str(claim.get("statement", "")).strip()
        assert statement

        tests = claim.get("tests", [])
        artifacts = claim.get("artifacts", [])
        assert isinstance(tests, list) and tests
        assert isinstance(artifacts, list) and artifacts

        for test_path in tests:
            assert Path(str(test_path)).exists(), f"Missing test reference in {claim_id}: {test_path}"
        for artifact_path in artifacts:
            assert Path(str(artifact_path)).exists(), f"Missing artifact reference in {claim_id}: {artifact_path}"
