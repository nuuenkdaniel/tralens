# Release Tralens

Open the release-preparation PR against `dev` and wait for **Reader checks** to pass. After merging, its push build publishes `ghcr.io/nuuenkdaniel/tralens-dev:dev` and a commit-specific tag. Test that image with Compose and a fresh data directory. Promote by opening a PR from `dev` into `main`. A successful push build on `main` publishes `:main` and its commit-specific tag.

For the first beta, verify on a clean Linux x86-64 host that Compose can pull and start the candidate. Import a page you can redistribute, translate it with local Ollama, and repeat with an eligible ChatGPT account you control. Verify session login over HTTPS, OAuth disconnect, an upgrade from the previous data layout, and a stopped backup and restore. The workflow covers the locked test suite, actual CPU OCR, model downloads, and owner sign-in in Docker.

Move the container and lock to a security-supported Python release before public distribution. The current locked build uses Python 3.10. Re-run the full suite and container smoke test after updating it. [Python release status](https://devguide.python.org/versions/)

Review the [third-party notices](../THIRD_PARTY_NOTICES.md) and each exact packaged dependency before public image redistribution. Confirm the release commit is publicly accessible and any separately licensed binaries have the required notices and corresponding source available. Keep third-party sample pages, checkpoints without confirmed permission, and private datasets outside the release.

GitHub Packages starts new GHCR packages as private. In the package settings, confirm the package is linked to this repository, the repository has Actions access, and change the production package visibility to Public when cleared for redistribution. Test `docker pull` using a Docker configuration without GitHub credentials. Use [GitHub's package settings guide](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility).

Once the release commit is on `main`, create tag `v0.1.0-beta.1` there. Its successful workflow publishes the versioned image used by the quick start. Create the GitHub release from that exact tag and mark it as a **prerelease**. Include the image reference, digest, supported platform, known limitations, and the setup link. Use new version tags for subsequent builds. Retain the source for every distributed image.

Enable private vulnerability reporting in the repository settings. For `dev` and `main`, require PRs and passing Reader checks. Keep workflow publishing permissions limited to the package jobs and restrict release tags to maintainers. [GitHub rulesets guide](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)
