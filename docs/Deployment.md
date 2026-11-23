# Deployment

The repository contains the same three selected seed-42 models through a checksum-pinned runtime release. Raw UTKFace images, participant data, credentials, training caches, and optimizer states are excluded.

## Local installation

Use Python 3.12. Install `requirements.txt` for CPU inference, then run `python scripts/download_runtime.py` and `python scripts/serve_api.py`. Open http://127.0.0.1:8000. For the verified Windows CUDA research environment, follow the root README instead of the CPU requirements file.

The runtime downloader verifies the archive SHA-256, size, inventory and every entry digest before writing controlled runtime paths. Model weights retain their original checkpoint bytes. The downloaded archive includes model and detector attribution. Review `THIRD_PARTY_NOTICES.md`; the selected ConvNeXt weights are noncommercial.

## GitHub

Source is intended for the public `GinnoLmao/VisAge` repository. Large inference weights are supplied as a versioned release asset instead of placing a file larger than GitHub's ordinary file limit into Git history. The release archive contains no photo fixtures. GitHub Pages can host the documentation and report portal, but cannot execute the Python inference API.

## Vercel

`api/index.py` exposes the FastAPI application. `vercel.json` includes the runtime models, detector and compact evaluation records; its build command restores the pinned runtime release. `requirements.txt` selects CPU-only PyTorch wheels. The hosted adapter uses temporary cache locations and loads the same selected weights once per warm function process.

The PyTorch bundle requires Vercel's eligible Large Functions/Fluid Compute support. The project is configured with `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`; it does not upgrade a billing plan. The cloud upload limit is **4 MiB** to stay below Vercel's 4.5 MB function-body limit. The local app retains **8 MiB**. The hosted models endpoint advertises the smaller limit, and the browser displays and enforces it. Both paths retain the original 20-megapixel decoded-image limit and shared geometry. Cold starts reload all models; timings from the local CPU benchmark are not cloud performance measurements.

Deployment is complete only after the production URL serves the model metadata, all three real predictions, and the evaluation dashboard successfully. A static preview, successful Git push, build completion alone, or localhost test does not prove hosted inference.

## Container alternative

When the selected hosting account cannot run the large Python function, the local package remains functional. Use a container-capable host with the included Dockerfile. Set a single worker so models are not duplicated, bind to the host's `PORT`, provision enough memory for all three models, and use HTTPS for camera access. No new paid hosting account or subscription is created automatically.

## Verification and limits

Public endpoint checks use a known validation fixture for functionality, not a new accuracy experiment. Do not upload private photos as deployment tests. Pilot photo testing and further tuning are excluded by the user's decision. Physical webcam operation was confirmed by the user; phone hardware testing is not required.

Primary platform references, checked November 23, 2026:

- https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site
- https://vercel.com/changelog/vercel-functions-can-now-be-up-to-5-gb-in-package-size
- https://vercel.com/docs/functions/limitations
- https://vercel.com/docs/functions/runtimes/python
