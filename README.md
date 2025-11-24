# RatioWaveNet2
RatioWaveNet2 is an EEG motor-imagery classifier built on top of the TCFormer architecture. The model preserves the temporal convolutional transformer core from TCFormer while adding a learnable graph neural network (GNN) front-end to capture spatial dependencies between EEG channels, resulting in a spatially aware temporal model for robust brain–computer interface research.

## Overview
- **TCFormer-inspired backbone:** The model directly adapts the TCFormer design and training strategy, crediting the original repository (https://github.com/altaheri/TCFormer) for the temporal-convolutional transformer baseline.
- **Graph-enhanced encoder:** A Chebyshev GCN-based EEGGraphLayer embeds spatial structure before the transformer stack, paired with squeeze-and-excitation channel gating and residual tuning to start from an identity mapping.
- **Flexible EEG pipeline:** Configuration-driven preprocessing, augmentation, and model hyperparameters enable quick replication of TCFormer settings with RadioWaveNet-specific additions.

## Architecture highlights
- **TCFormer heritage:** The implementation documents its TCFormer lineage and reuse of the temporal-convolutional transformer head inside `models/ratiowavenet.py`.
- **EEGGraphLayer:** A Chebyshev graph convolution over normalized adjacency (with optional symmetry and self-loops) precedes the transformer, includes channel-wise SE attention, and can revert to identity via a learnable residual scale.
- **Grouped attention and TCN head:** The configuration exposes grouped query/key attention depths and TCN head depth to mirror TCFormer while enabling RadioWaveNet tuning.

## Setup
1. Create a Python 3.10 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training and evaluation
1. Choose or edit `configs/ratiowavenet.yaml` to match your dataset and hardware.
2. Run the end-to-end pipeline (example for LOSO on HGD):
   ```bash
   python train_pipeline.py --model ratiowavenet --dataset hgd --loso --interaug --subject 13
   ```
   The script resolves dataset-specific DataModules, instantiates the RadioWaveNet model with matching channels/classes, and trains with PyTorch Lightning.
3. Outputs (checkpoints, curves, confusion matrices, latency estimates, and YAML copies of the run configuration) are written under `results/` with timestamped subfolders.

## Results

**Accuracy Summary (Subject-Dependent vs. LOSO)**

The table reports **mean accuracy (%)** for all models across **BCI IV-2a**, **BCI IV-2b**, and **HGD** in both **subject-dependent (Sub-Dep)** and **Leave-One-Subject-Out (LOSO)** settings, **with (+aug)** and **without (–aug)** augmentation, plus model **parameter counts (k)**. Parameter counts are referenced from the IV-2a configuration and may vary slightly with dataset/channel count.

<table>
  <thead>
    <tr>
      <th rowspan="3">Model</th>
      <th rowspan="3">Params (k)</th>
      <th colspan="4"><a href="https://www.bbci.de/competition/iv/#dataset2a">BCI Comp IV-2a</a></th>
      <th colspan="4"><a href="https://www.bbci.de/competition/iv/#dataset2b">BCI Comp IV-2b</a></th>
      <th colspan="4"><a href="https://github.com/robintibor/high-gamma-dataset">HGD (High-Gamma)</a></th>
    </tr>
    <tr>
      <th colspan="2">Sub-Dep</th>
      <th colspan="2">LOSO</th>
      <th colspan="2">Sub-Dep</th>
      <th colspan="2">LOSO</th>
      <th colspan="2">Sub-Dep</th>
      <th colspan="2">LOSO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>EEGNet</td>
      <td align="right">1.7</td>
      <td align="right">72.62</td>
      <td align="right">52.03</td>
      <td align="right">83.65</td>
      <td align="right">77.89</td>
      <td align="right">85.94</td>
      <td align="right">60.12</td>
    </tr>
    <tr>
      <td>ShallowNet</td>
      <td align="right">44.6</td>
      <td align="right">65.72</td>
      <td align="right">47.31</td>
      <td align="right">81.45</td>
      <td align="right">75.58</td>
      <td align="right">91.54</td>
      <td align="center">—</td>
    </tr>
    <tr>
      <td>BaseNet</td>
      <td align="right">3.7</td>
      <td align="right">78.58</td>
      <td align="right">56.89</td>
      <td align="right">86.11</td>
      <td align="right">78.61</td>
      <td align="right">95.40</td>
      <td align="center">—</td>
    </tr>
    <tr>
      <td>EEGTCNet</td>
      <td align="right">4.1</td>
      <td align="right">78.82</td>
      <td align="right">55.99</td>
      <td align="right">86.74</td>
      <td align="right">80.56</td>
      <td align="right">93.54</td>
      <td align="center">—</td>
    </tr>
    <tr>
      <td>TS-SEFFNet</td>
      <td align="right">334.8</td>
      <td align="center">—</td>
      <td align="center">—</td>
      <td align="center">—</td>
      <td align="center">—</td>
      <td align="center">—</td>
      <td align="center">—</td>
    </tr>
    <!-- CTNet split into two rows (two configurations) -->
    <tr>
      <td rowspan="2">CTNet,&nbsp;&nbsp;F1=20<br/><span style="opacity:.7">CTNet, F1=8</span></td>
      <td rowspan="2" align="right">152.7<br/><span style="opacity:.7">27.3</span></td>
      <!-- conf-1 -->
      <td align="right">81.91</td>
      <td align="right">60.09</td>
      <td align="right">86.91</td>
      <td align="right">80.29</td>
      <td align="right">94.21</td>
      <td align="right">64.60</td>
    </tr>
    <tr>
      <!-- conf-2 -->
      <td align="right">79.24</td>
      <td align="right">56.17</td>
      <td align="right">87.50</td>
      <td align="right">80.15</td>
      <td align="right">92.22</td>
      <td align="center">—</td>
    </tr>
    <tr>
      <td>MSCFormer</td>
      <td align="right">150.7</td>
      <td align="right">79.16</td>
      <td align="right">54.27</td>
      <td align="right">87.60</td>
      <td align="right">79.20</td>
      <td align="right">94.31</td>
      <td align="right">61.19</td>
    </tr>
    <tr>
      <td>EEGConformer</td>
      <td align="right">789.6</td>
      <td align="right">75.39</td>
      <td align="right">45.59</td>
      <td align="right">81.89</td>
      <td align="right">75.25</td>
      <td align="right">94.67</td>
      <td align="right">69.92</td>
    </tr>
    <tr>
      <td>ATCNet</td>
      <td align="right">113.7</td>
      <td align="right">83.78</td>
      <td align="right">59.66</td>
      <td align="right">86.26</td>
      <td align="right">80.94</td>
      <td align="right">95.08</td>
      <td align="center">—</td>
    </tr>
    <tr>
      <td><strong>TCFormer</strong></td>
      <td align="right">77.8</td>
      <td align="right"><strong>84.79</strong></td>
      <td align="right"><strong>63.00</strong></td>
      <td align="right"><strong>87.71</strong></td>
      <td align="right"><strong>81.34</strong></td>
      </td><td align="right"><strong>96.27</strong></td>
      <td align="right"><strong>72.83<sup>1</sup></strong></td>
    </tr>
    <tr>
      <td><strong>RatioWaveNet</strong></td>
      <td align="right">77.8</td>
      <td align="right">83.06</td><td align="right"><strong>84.79</strong></td>
      <td align="right"><strong>62.44</strong></td><td align="right"><strong>63.00</strong></td>
      <td align="right"><strong>87.11</strong></td><td align="right"><strong>87.71</strong></td>
      <td align="right">79.73</td><td align="right"><strong>81.34</strong></td>
      <td align="right"><strong>95.62</strong></td><td align="right"><strong>96.27</strong></td>
      <td align="right">71.90<sup>1</sup></td><td align="right"><strong>72.83<sup>1</sup></strong></td>
    </tr>
  </tbody>
</table>

> <sup>1</sup> Using a deeper TCFormer encoder (**N = 5**, ≈131 k params). See the paper for details.  
> Reported accuracies were averaged over 5 runs (BCI IV-2a/2b) or 3 runs (HGD) using the final (last-epoch) checkpoint; no early stopping or validation-based model selection.

## Attribution
RatioWaveNetis derived from TCFormer, retaining its temporal-convolutional transformer core and repository structure. Please cite TCFormer alongside this work when reporting results.
