# EdgeLUT-Agent

**EdgeLUT-Agent: A Lightweight Graph-Aware LUT Agent for On-Device Image Restoration**

A compact agentic system for edge-side image restoration using dynamic 3D LUTs with graph-aware routing and closed-loop feedback.

## Key Results

| Metric | Value |
|---|---:|
| Total parameters | **48,478** (~194 KB FP32) |
| CPU inference latency | **0.99 ms** (measured) |
| MNIST PSNR / SSIM | **26.78 dB / 0.949** |
| Average routed path FLOPs | ~1.09 MFLOP |

### Qwen2.5-0.5B Router Comparison

| Router | PSNR (dB) | SSIM | Latency |
|---|---:|---:|---:|
| Rule-based (threshold) | 23.73 | 0.904 | <0.1 ms |
| Qwen2.5-0.5B-Instruct (FP32, CPU) | 23.64 | 0.898 | 6,412 ms |

**Finding**: The 0.5B model exhibits output collapse across 100 test samples — it produces nearly identical routing JSON regardless of input degradation levels, effectively memorizing the few-shot example rather than reasoning from numeric observations. This is a critical limitation of on-device small LLMs for structured routing tasks.

## Architecture

```
Degraded image → [PerceptionNet CNN] → p = [p_noise, p_blur, p_lowlight]
                                          ↓
                                    [Router: Rule / Qwen] → LUT action sequence
                                          ↓
Degraded image → [LUT_denoise?] → [LUT_deblur?] → [LUT_lowlight?] → Restored
                                          ↓
                                    [Feedback loop] → quality metrics → re-route or stop
```

### Component Breakdown

| Component | Params | Description |
|---|---:|---|
| PerceptionNet | 3,763 | 3-layer CNN → 3-dim degradation vector |
| LUT_denoise | 14,905 | 17³×3 base LUT + FiLM conditioner (3→16→6) |
| LUT_deblur | 14,905 | 17³×3 base LUT + FiLM conditioner |
| LUT_lowlight | 14,905 | 17³×3 base LUT + FiLM conditioner |
| **Total** | **48,478** | |

### Dynamic 3D LUT (FiLM Conditioning)

```
scale = tanh(MLP(p)[:3]) × 0.5    # ∈ [-0.5, 0.5]
bias  = tanh(MLP(p)[3:]) × 0.1    # ∈ [-0.1, 0.1]
effective_LUT = base_LUT × (1 + scale) + bias
output = trilinear_sample(effective_LUT, input_pixel)
```

## Ablation Study

| Variant | PSNR (dB) | ΔPSNR |
|---|---:|---:|
| Full model (rule router + feedback) | 26.78 | — |
| w/o feedback re-route | 24.44 | -2.34 |
| w/o FiLM (static base LUT) | 21.27 | -5.51 |
| w/o routing (all 3 serial) | 26.66 | -0.12 |

## Files

```
├── demo.html                    # Single-file HTML demo (weights inlined, mobile-friendly)
├── report.tex / report.pdf      # English LaTeX paper (ICLR style)
├── buildozer.spec               # Android APK build configuration
├── src/
│   ├── dataset.py                # MNIST composite degradation dataset
│   ├── model.py                  # PerceptionNet, DynamicLUT, EdgeLUTAgent
│   ├── train.py                  # Training loop
│   ├── evaluate.py              # Main comparison + ablation experiments
│   ├── export_weights.py         # Export .npz weights
│   ├── inject_weights.py         # Inject real weights into demo.html
│   ├── qwen_router.py            # Qwen2.5-0.5B routing comparison
│   └── real_image_eval.py        # Set12 sliding-window evaluation
├── weights/
│   ├── edgelut_agent_best.pt     # Best PyTorch checkpoint
│   └── edgelut_agent.npz         # Numpy export (26 arrays)
├── results/
│   ├── metrics.json              # MNIST experiment metrics
│   ├── qwen_vs_rule.json        # Qwen vs rule router comparison
│   ├── real_image_metrics.json   # Set12 evaluation results
│   ├── visual_samples.png        # MNIST restoration visualization
│   ├── real_image_comparison.png # Set12 sliding-window comparison
│   ├── training_curve.png
│   ├── comparison_bar.png
│   ├── ablation_bar.png
│   └── perception_examples.png
├── router_samples/              # 10 prompt→Qwen decision→PSNR samples
├── test_images/                  # Set12 clean and degraded images
└── data/                         # MNIST idx files
```

## Quick Start

### Train (CPU, ~5 min)
```bash
python src/train.py
```

### Run Experiments
```bash
python src/evaluate.py
python src/qwen_router.py     # Requires transformers + ~2.5GB RAM
python src/real_image_eval.py # Requires Set12 images
```

### Mobile Demo
Open `demo.html` in any browser — works offline, weights are inlined.

### Build APK (optional)
```bash
pip install buildozer
buildozer android debug
```

## Limitations

- Trained on 28×28 MNIST only; poor generalization to real photos (Set12 PSNR drops to 9.8 dB due to domain gap)
- Qwen2.5-0.5B shows output collapse on numeric routing tasks; rule-based threshold router is more practical for edge deployment
- Feedback loop uses heuristic quality metrics, not learned

## Requirements

- Python 3.10+, PyTorch 2.0+, numpy, Pillow, matplotlib, scipy
- Optional: transformers (for Qwen router evaluation)

> （注：部分内容由豆包 AI 生成）
