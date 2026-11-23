# Tralens privacy notice

Tralens is software you run on a server you control. Your library, original uploads, translated pages, transcripts, reading positions, and job diagnostics are stored on that server. Reader preferences and the selected translator are stored in your browser. The Tralens project does not receive this information from your instance.

When you select Local, translation requests go to the Ollama endpoint configured by the operator. Review that service's configuration, including cloud settings. The setup service contacts Hugging Face to download the selected OCR exports. OCR itself runs on your server.

## Sign in with ChatGPT

Signing in is optional. When you agree and continue, the OAuth flow connects directly to OpenAI. Tralens stores the account identifier, email address, and OAuth tokens in an owner-only configuration file on your server. It uses those credentials to determine available models and send authorized translation requests. OAuth credentials stay in your server configuration and are exchanged with OpenAI to refresh or revoke access.

For imports and retries you start with a ChatGPT model, Tralens sends page images, detected text, and related translation prompts to OpenAI. Tralens processes those jobs in the background, including when you close the reader. OpenAI handles those requests under its own terms and privacy policy. Use a runtime only you control with your own account.

Disconnect in Tralens Settings to revoke the connection. You can also manage connections in ChatGPT Settings. Disconnecting removes the locally stored sign-in tokens after revocation succeeds. The local registration retains the account identifier and email for reconnecting. For full removal, disconnect first, then remove the ChatGPT account file from the stopped server's configuration directory. The operator manages copies in backups.

## Retention

Library entries remain until you delete them. Job diagnostics can contain copies of inputs and outputs and remain on disk after library deletion. The operator can remove those files and backups separately. See the self-hosting guide for storage paths. If you contact the project for support, share only files you intend to disclose.

OpenAI Privacy Policy
https://openai.com/policies/privacy-policy/

Sign in with ChatGPT Terms
https://openai.com/policies/sign-in-with-chatgpt-terms/
