"""Exercise release immutability with a fake registry client."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "deploy/promote-release.sh"
DIGEST = "sha256:" + "a" * 64
REVISION = "b" * 40

FAKE_SKOPEO = """#!/bin/sh
set -eu
case "$1 $2" in
    'inspect --config')
        printf '{"os":"linux","architecture":"amd64","config":{"Labels":{"org.opencontainers.image.revision":"%s","org.opencontainers.image.source":"%s/%s"}}}\n' \
            "$CANDIDATE_REVISION" "$GITHUB_SERVER_URL" "$GITHUB_REPOSITORY"
        ;;
    'inspect --no-tags')
        if [ -n "$INSPECT_ERROR" ]; then
            echo "$INSPECT_ERROR" >&2
            exit 1
        fi
        if [ -f "$REGISTRY_STATE" ]; then
            cat "$REGISTRY_STATE"
        else
            echo 'manifest unknown' >&2
            exit 1
        fi
        ;;
    'copy --all')
        [ "$3" = --preserve-digests ]
        [ "$4" = --digestfile ]
        [ "$6" = "docker://ghcr.io/$GITHUB_REPOSITORY@$EXPECTED_DIGEST" ]
        [ "$7" = "docker://ghcr.io/$GITHUB_REPOSITORY-release:v0.1.0-beta.1" ]
        printf '%s\n' "$EXPECTED_DIGEST" > "$5"
        printf '%s\n' "$EXPECTED_DIGEST" > "$REGISTRY_STATE"
        touch "$REGISTRY_STATE.copied"
        ;;
    *) exit 90 ;;
esac
"""


class ReleasePromotionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        executable = directory / "skopeo"
        executable.write_text(FAKE_SKOPEO)
        executable.chmod(0o700)
        self.state = directory / "state"
        self.copied = directory / "state.copied"
        self.env = {
            **os.environ,
            "PATH": str(directory) + ":" + os.environ["PATH"],
            "GITHUB_REPOSITORY": "nuuenkdaniel/tralens",
            "GITHUB_SERVER_URL": "https://github.com",
            "REGISTRY_STATE": str(self.state),
            "EXPECTED_DIGEST": DIGEST,
            "CANDIDATE_REVISION": REVISION,
            "INSPECT_ERROR": "",
        }

    def promote(self):
        return subprocess.run(
            ["bash", str(SCRIPT), "v0.1.0-beta.1", DIGEST, REVISION],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_copy_preserves_digest_and_retry_does_not_copy(self):
        result = self.promote()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.state.read_text().strip(), DIGEST)
        self.assertTrue(self.copied.exists())
        self.copied.unlink()
        result = self.promote()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.copied.exists())

    def test_refuses_existing_version_at_another_digest(self):
        self.state.write_text("sha256:" + "c" * 64)
        result = self.promote()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already points", result.stderr)
        self.assertFalse(self.copied.exists())

    def test_registry_error_does_not_count_as_missing_version(self):
        self.env["INSPECT_ERROR"] = "unauthorized"
        result = self.promote()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unauthorized", result.stderr)
        self.assertFalse(self.copied.exists())

    def test_refuses_candidate_built_from_another_git_revision(self):
        self.env["CANDIDATE_REVISION"] = "d" * 40
        self.assertNotEqual(self.promote().returncode, 0)
        self.assertFalse(self.copied.exists())
