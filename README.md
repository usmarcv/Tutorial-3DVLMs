# An overview of 3D Vision-Language Models

Source of the project page for the SIBGRAPI 2026 tutorial
**An overview of 3D Vision-Language Models**.

**Live site:** https://usmarcv.github.io/Tutorial-3DVLMs/

[Paper (arXiv)](https://arxiv.org/abs/2609.05583) |
[Hands-on code (`main` branch)](https://github.com/usmarcv/Tutorial-3DVLMs/tree/main)

## Pages

| File | Content |
|---|---|
| `index.html` | Home: abstract, outline, taxonomy, tasks, when and where, citation |
| `schedule.html` | Program of Part 1 and Part 2 |
| `code.html` | Slides (PDF) and how to run the hands-on notebook |
| `people.html` | Presenters and co-authors |

Shared files: `style.css`, `script.js`, and `assets/` (figures, people photos,
and the slide PDFs in `assets/slides/`).

## Preview locally

The site is plain HTML, CSS and JavaScript, with no build step:

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000.

## Branches

- `page` (this branch) - the website.
- `main` - the hands-on notebook, helper code and data.

## Credits

Website design inspired by [DA-Flow](https://cvlab-kaist.github.io/DA-Flow/) and
[AgentRVOS](https://cvlab-kaist.github.io/AgentRVOS/).
