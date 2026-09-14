# Screenshots — how to add them

Screenshots are the one thing a README cannot fake: they prove the UI exists and runs.
They also become the social preview on LinkedIn, Slack and Twitter.

## Taking them

```bash
cd projects/01_mcp_redteam_platform
uvicorn webapp.app:app --reload        # or: streamlit run ui/app.py
```

Open the page, screenshot it, and save as **PNG** into `docs/` using this naming:

```
docs/images/01-redteam-chat.png
docs/images/01-redteam-audit.png
docs/images/03-bfcl-results.png
```

Keep each under ~300 KB (PNG, resize to 1600px wide). Large binaries bloat every future
clone of the repo permanently.

## Referencing them

In the project README, directly under the headline paragraph:

```markdown
![Red-team audit results](../docs/images/01-redteam-audit.png)
```

## Social preview

GitHub → repo **Settings** → **Social preview** → upload a 1280×640 image. That is what
renders when the link is pasted anywhere, and it is the single highest-leverage image in
the repo.
