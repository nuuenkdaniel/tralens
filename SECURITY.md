# Security policy

The Tralens beta supports one owner per Linux instance and optional reader accounts with private libraries. Set `TRALENS_REGISTRATION=closed` before exposing a personal server to other people. Each registered reader can consume local inference and storage capacity. The reader and Ollama should bind to loopback. For remote access, use an SSH tunnel or configure the reader's HTTPS owner mode behind a reverse proxy. See the [self-hosting guide](docs/self-hosted-reader.md).

Report suspected vulnerabilities privately using the repository's [private vulnerability reporting](https://github.com/nuuenkdaniel/tralens/security/advisories/new). If reporting is unavailable, use the repository owner's private contact information. Include the image tag or Git commit, affected component, impact, and steps that help confirm the issue. Remove any account tokens, owner keys, or private library content from the report.
