# Research runs

A static explorer for research runs, served at https://invi-bhagyesh.github.io/research/. It reads each project's run
log directly from a Hugging Face dataset in the browser; no data is stored in this repo.

- **Private datasets:** paste a read token on the site's Settings page; it stays in your browser.
- **Manual review:** in the sample reader, `1`/`2`/`3` set lie / not lie / unsure, `j`/`k` move to the next / previous
  sample, and the run's Samples tab exports your verdicts as `<run>.reviewed.jsonl`.

## Add a project

Add an entry to `projects.json`:
- `hf_dataset`, `prefix`: where its runs live, in the layout `<prefix>/runs/<run_id>/{meta.json,status.json,progress.log,...}`
  written by `hublog.py` in the lying-organism repo;
- `transcript_runs`: runs holding `episodes.jsonl.gz`, used to show the transcript behind a sample;
- `groups`: how runs are grouped on the project page;
- `links`, `highlights`: shown on the project page.

The run views appear based on the files present: `samples.jsonl.gz`, `episodes.jsonl.gz`, `rewrites.jsonl.gz`,
`train.jsonl.gz`, `train_log.jsonl.gz`, `summary.md`, `progress.log`, and `job.log`/`boot.log`.
Review verdicts are read from `<prefix>/reviewed/<run>.jsonl`.
