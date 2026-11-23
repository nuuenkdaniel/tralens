# Release Tralens

Open a PR against `dev` and wait for **Reader checks** to pass. Its merge builds
`ghcr.io/nuuenkdaniel/tralens-dev:dev`. Promote it through a PR from `dev` into
`main`. The main push build publishes a candidate to
`ghcr.io/nuuenkdaniel/tralens:main`. Each run records the pushed image digest
in its job summary. Use the digest from the successful main run for acceptance
and release. Keep both candidate packages private.

## Configure GitHub

Require PRs and the Reader checks on `dev` and `main`. Use the actual check
names from a successful PR, `clean-install` and `container / image`. Restrict
creation of `v*` tags to release maintainers and block tag updates and deletions.
Enable private vulnerability reporting in the repository settings.
[GitHub rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)

Create an Actions environment named `public-beta`. Restrict its deployments to
`main`, add a required reviewer, and disable administrator bypass of protection
rules. For a solo maintainer, allow the reviewer to approve their own run.
The staging workflow verifies that required reviewers are configured before it
writes to the release registry. [GitHub deployment environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)

## Approve the exact candidate

On a fresh Linux x86-64 host, check out the commit from the candidate's
`org.opencontainers.image.revision` label. Set `TRALENS_IMAGE` to
`ghcr.io/nuuenkdaniel/tralens@sha256:<candidate digest>` in `.env`. Authenticate
to GHCR with read access to the private candidate, then follow the
[Compose setup](../README.md#start-the-reader). Record the digest and results.

- Import a page you have permission to use and complete a translation with
  local Ollama. Check the rendered result and original.
- Sign in through the actual HTTPS endpoint. Connect your eligible ChatGPT
  account, translate a page, and verify that disconnect works.
- Stop the reader, back up its data and test restoration. If there is a
  previous release, exercise upgrade and rollback with a copy of its data.
- Review the [third-party notices](../THIRD_PARTY_NOTICES.md) against the
  exact image. Check the Debian source inventory and the installed wheels'
  notices, including corresponding source for covered bundled binaries.
  Keep a copy of the review with the candidate digest.

CI exercises fresh Compose startup, the owner key, real CPU OCR, and recovery
of a completed job from a stopped backup.

## Stage and publish the beta

1. Create the beta tag, such as `v0.1.0-beta.1`, on the approved commit from
   `main`. Run **Stage public beta** from `main` with the tag and exact
   candidate digest. Confirm that the acceptance checklist is complete and
   approve the environment deployment.
2. The workflow pulls and tests that digest, then copies it into the separate
   `ghcr.io/nuuenkdaniel/tralens-release` package. It refuses to replace a
   version that points to different content. It creates a draft prerelease
   with a digest-pinned Compose bundle and checksum.
3. After redistribution is cleared, make only the release package public.
   GitHub treats this change as permanent. Verify that the repository has
   Actions access to the package. [GitHub package visibility](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility)
4. Run **Verify public beta image** from `main` with the tag and digest. It
   tests anonymous pulls of the tag and digest on a fresh runner. Download
   the draft's Compose bundle and check that its `.env.example` contains
   the same digest. Publish the draft prerelease after these checks pass.

Use a new version tag for each release and retain the exact source and digest
for every distributed image. Keep the previous bundle and a stopped backup for
[rollback](self-hosted-reader.md#data-upgrades-and-recovery).
