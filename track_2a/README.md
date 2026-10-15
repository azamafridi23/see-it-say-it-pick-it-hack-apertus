# Academia Challenges

> **Our project: See It, Say It, Pick It (ZHAW), team Asian Prompters.** `make run` runs our pipeline
> in Docker on the held-out scenes. The write-up is in [technical_report.md](technical_report.md) and the
> full run guide in [docs/USAGE.md](docs/USAGE.md).

Submissions must use the Apertus model family.
For Track 2 this means that submitted solutions must be built with Apertus. Other open-weights models can be used to support development, e.g. as automatic judges during evaluation. Their role must be clearly described in the submission report.

💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”

## How it works
Pick from 5 academia challenges provided by Swiss academic institutions:

- **FHGR:** AI-Powered Job Interview Coach
- **OpenParlData:** Extracting Parliamentary Affairs from PDFs into One Common Structure
- **OST:** Multilingual Natural Language Inference over Swiss Official Voting Booklets
- **UZH:** Detecting Cross-Lingual Semantic Differences in Swiss Government Websites
- **ZHAW:** See It, Say It, Pick It: Vision-Language Grounding for a Real Robot Arm

The challenges incl. submission and judging criteria are described in our **Getting Started guide**:
https://hackapertus.notion.site/getting-started-guide-onlinehack

## Run it

Keep `track_2a/` as it is: don't rename it or move its files, just delete the
other track directories.

From the root of the project:

```bash
make run
```

Fill in the [Makefile](Makefile) so that it works on a clean checkout. It is
expected to run the project in a Docker container, since that is how the judges
will run it, without relying on anything already installed on your machine.

Requirements:

- **Runtime:** Docker. `make run` builds the image and runs Path A on `data/heldout/`; the answers go
  to `results/results_heldout.json`. `make run DATA=path/to/scenes` runs any folder of scenes.
- **Hardware:** any machine that runs Docker, with at least 8 GB of memory for the container. No GPU
  is needed: the image uses PyTorch's CPU-only build.
- **API keys:** `LLM_API_KEY` for the Apertus endpoint, in the environment or in `.env` (see
  `.env.example`). `LLM_BASE_URL` and `LLM_NAME` default to the hosted endpoint and
  `apertus-v1.5-70b`.
- **Model weights:** Meta SAM 3 (from the ungated, byte-identical copy `jetjodh/sam3`) and Depth
  Anything V2 Small, both downloaded into the image during the build; no Hugging Face token is needed.

## Data
The `data/` directory must not exceed 100 MB.


## 📦 Submission Requirements & Deliverables
❗️ Submissions are not handled on Devpost. Submit through our website only:
http://hackapertus.ch/online-hack/submissions

Requirements differ by challenge. See the description of the challenge you are entering for the exact deliverables.


## ⚖️ Judging Criteria
Judging criteria also differ by challenge. See the respective challenge description.


## Support

**Licensing requirements**
Please check our Terms & Conditions (6. What you build is open source):
https://hackapertus.ch/terms-and-conditions

## FAQ
💡 https://hackapertus.ch/faq

## Contact
💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”
