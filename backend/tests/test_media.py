"""Phase 2 P7: authorized media path (probe, audio extraction, frame sampling, visual signal
evaluator, upload endpoint).

All media here is SYNTHETIC test media generated with ffmpeg (test patterns and tones); no real
video is processed in tests.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from avd.content import media
from avd.evaluation.evaluators.visual import VisualSignalEvaluator, score_visual, summarize
from avd.evaluation.profiles import get_profile
from avd.evaluation.schema import Dimension, DimensionStatus, Genre, ReasonCode

pytestmark = pytest.mark.skipif(not media.available(), reason="ffmpeg/ffprobe not installed")


def make_media(
    path: Path, vf: str = "", seconds: int = 4, size: str = "1280x720", audio: bool = True
) -> Path:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate=10:duration={seconds}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest"]
    cmd += ["-c:a", "aac"] if audio else ["-an"]
    subprocess.run([*cmd, str(path)], check=True, timeout=120)
    return path


def test_probe_extract_and_sample(tmp_path: Path) -> None:
    f = make_media(tmp_path / "clip.mp4")
    info = media.probe(f, root=tmp_path)
    assert info.has_audio and info.has_video and (info.width, info.height) == (1280, 720)
    assert 3.5 < info.duration_seconds < 4.6
    wav = media.extract_audio_wav(f, tmp_path / "a.wav", root=tmp_path)
    assert wav.stat().st_size > 16000 * 2 * 3  # ~4 s of 16 kHz 16-bit mono
    frames = media.sample_frames(f, info, n=6, root=tmp_path)
    assert len(frames) == 6 and frames[0].shape == (180, 320)


def test_paths_outside_media_dir_and_non_media_are_refused(tmp_path: Path) -> None:
    (tmp_path / "inside").mkdir()
    outside = make_media(tmp_path / "outside.mp4", seconds=1, audio=False)
    with pytest.raises(media.MediaError, match="inside the media directory"):
        media.probe(outside, root=tmp_path / "inside")
    junk = tmp_path / "inside" / "notes.mp4"
    junk.write_text("this is not a video")
    with pytest.raises(media.MediaError):
        media.probe(junk, root=tmp_path / "inside")
    # A playlist pointing at a URL must not be fetched (file protocol only).
    pl = tmp_path / "inside" / "list.m3u8"
    pl.write_text("#EXTM3U\n#EXTINF:1,\nhttp://127.0.0.1:9/x.ts\n")
    with pytest.raises(media.MediaError):
        media.probe(pl, root=tmp_path / "inside")


def test_visual_metrics_separate_sharp_from_blurred_dark(tmp_path: Path) -> None:
    good = make_media(tmp_path / "good.mp4", audio=False)
    bad = make_media(tmp_path / "bad.mp4", vf="boxblur=10:2,eq=brightness=-0.45", size="640x360", audio=False)
    mg = summarize(media.sample_frames(good, media.probe(good), n=8))
    mb = summarize(media.sample_frames(bad, media.probe(bad), n=8))
    assert mg["sharpness"] > 3 * mb["sharpness"] and mg["brightness"] > mb["brightness"]
    sg, _ = score_visual(mg, 720)
    sb, issues = score_visual(mb, 360)
    assert sg > sb and any("blurry" in i or "soft" in i for i in issues)


def test_visual_evaluator_scores_quality_and_abstains_on_editing(tmp_path: Path) -> None:
    from tests.conftest import make_input

    f = make_media(tmp_path / "v.mp4", audio=False)
    inp = make_input().model_copy(update={"video_path": str(f), "video_sha256": "x", "video_source_id": "s"})
    out = VisualSignalEvaluator().evaluate(inp, get_profile(Genre.LECTURE))
    by = {d.dimension: d for d in out.dimensions}
    assert by[Dimension.VISUAL_QUALITY].status == DimensionStatus.SCORED
    assert (
        by[Dimension.VISUAL_QUALITY].confidence is not None and by[Dimension.VISUAL_QUALITY].confidence <= 0.4
    )
    for d in (Dimension.EDITING, Dimension.VISUAL_READABILITY):
        assert (
            by[d].status == DimensionStatus.INSUFFICIENT_EVIDENCE
            and by[d].reason_code == ReasonCode.METHOD_CANNOT_JUDGE
        )
    none = VisualSignalEvaluator().evaluate(make_input(), get_profile(Genre.LECTURE))
    assert {d.dimension: d.status for d in none.dimensions}[
        Dimension.VISUAL_QUALITY
    ] == DimensionStatus.REQUIRES_MEDIA


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from tests.test_ops import _client

    yield from _client(tmp_path, monkeypatch, AVD_MAX_MEDIA_UPLOAD_MB="5")


def test_media_upload_end_to_end(api: TestClient, tmp_path: Path) -> None:
    tok = api.post("/v1/users").json()["token"]
    h = {
        "Authorization": f"Bearer {tok}",
        "Content-Type": "video/mp4",
        "X-Content-Attestation": "I created this video myself.",
    }
    clip = make_media(tmp_path / "upload.mp4").read_bytes()
    r = api.put("/v1/videos/user_media_01/media", content=clip, headers=h)
    assert r.status_code == 202, r.text
    assert set(r.json()["modalities"]) == {"audio", "video"}
    assert api.put("/v1/videos/user_media_01/media", content=clip, headers=h).json()["created"] is False
    out = api.post("/v1/admin/worker/run", headers={"X-Admin-Token": "test-admin"}).json()["outcomes"]
    assert out.get("succeeded", 0) >= 2
    ev = api.get("/v1/videos/user_media_01/evaluation").json()
    dims = {d["dimension"]: d for d in ev["dimensions"]}
    assert dims["audio_quality"]["status"] == "scored" and dims["visual_quality"]["status"] == "scored"
    assert dims["depth"]["status"] == "requires_media"  # no transcript: content is not judged
    assert ev["overall"]["score"] is None  # content coverage too low for an overall summary
    # Refusals: wrong type, missing attestation, too large, not media.
    assert (
        api.put(
            "/v1/videos/user_media_01/media", content=b"x", headers={**h, "Content-Type": "text/plain"}
        ).status_code
        == 415
    )
    no_att = {k: v for k, v in h.items() if k != "X-Content-Attestation"}
    assert api.put("/v1/videos/user_media_01/media", content=clip, headers=no_att).status_code == 422
    assert (
        api.put("/v1/videos/user_media_02/media", content=b"0" * (6 * 1024 * 1024), headers=h).status_code
        == 413
    )
    bad = api.put("/v1/videos/user_media_03/media", content=b"definitely not an mp4 file", headers=h)
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "validation_error"
