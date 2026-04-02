# 2026-04-02 Vast Under $6/Hour Matrix

Search date: 2026-04-02

This matrix reflects current live `vastai search offers` results under the maintained TF contracts. It is intentionally stricter than older ETA notes.

## Current Picks

| TF | Contract mode | Hard floor | Preferred profile | Recommended current fit | Budget fit | Why |
|---|---|---|---|---|---|---|
| `1w` | CPU-first trimmed | `16c / 64GB / 1 GPU` | `32c / 128GB / GPU optional` | Local `13900K + 3090` first. Cloud fallback: offer `33198342` (`1x RTX 4090`, `256c`, `513GB`, Norway) at about `$0.47/hr` | Yes | Weekly is trimmed and no-cross. Cheap high-RAM single-GPU boxes are enough. |
| `1d` | CPU-first | `32c / 128GB / 1 GPU` | `128c / 256GB / GPU optional` | Local first. Cloud fallback: offer `33198342` (`1x RTX 4090`, `256c`, `513GB`, Norway) at about `$0.47/hr` | Yes | `1d` remains CPU-first and benefits more from RAM/cores than from large GPU count. |
| `4h` | Hybrid transition | `48c / 256GB / 1 GPU` | `128c / 512GB / 1 GPU preferred` | Offer `33198342` (`1x RTX 4090`, `256c`, `513GB`, Norway) at about `$0.47/hr` | Yes | Meets the preferred RAM target with enough GPU to make the hybrid lane worthwhile. |
| `1h` | GPU-required | `64c / 512GB / 1 GPU` | `128c / 768GB / 2 GPU preferred` | Offer `30818874` (`4x A40`, `256c`, `2051GB`, Belgium) at about `$1.15/hr` | Yes | Not the fastest DLPerf candidate, but it comfortably meets the RAM/GPU contract and stays cheap. |
| `15m` | GPU-required same-machine | `96c / 768GB / 1 GPU` | `128c / 1024GB / 2 GPU preferred` | Offer `33432644` (`8x RTX 6000 Ada`, `128c`, `1032GB`, Finland) at about `$4.80/hr` | Yes | This is the first live under-`$6/hr` offer in the search that actually matches the preferred same-machine RAM/GPU posture. |

## Notable Alternate Fits

- `34007697`: `2x RTX 5090`, `128c`, `516GB`, Japan, about `$1.13/hr`
  - good `1h` speed candidate
  - does not hit the `1h` preferred `768GB` RAM target
- `33394740`: `4x RTX 5090`, `384c`, `516GB`, Taiwan, about `$1.60/hr`
  - strong raw speed candidate
  - still below the preferred `1h/15m` RAM target
- `26597007`: `8x RTX 3090`, `128c`, `774GB`, Sichuan, about `$1.12/hr`
  - interesting budget lower-TF option
  - meets `1h` hard floor and nearly reaches `15m` RAM needs
  - weaker than the `8x RTX 6000 Ada` fit for the strict `15m` lane

## Notes

- `1w` and `1d` remain local-first. The Vast rows above are fallback/certification candidates, not a claim that cloud is always the best value.
- This report uses contract fit first, not just TFLOPs per dollar.
- If future inventory does not include a true under-`$6/hr` fit for a TF, the correct answer is “no certified fit,” not a looser recommendation.
