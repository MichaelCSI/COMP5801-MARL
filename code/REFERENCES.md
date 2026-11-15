# References

Works and software used to build this project.

## Simulator and data

[1] T. Fonseca, L. L. Ferreira, A. Sousa, A. Mohammadi, Z. Nagy, "CityLearn v3:
A Configurable Simulation and Evaluation Framework for Realistic Control Studies
of Renewable Energy Communities," arXiv:2609.21570, 2026.
*Used for:* the simulator (`citylearn==3.0.2`), wrapped by
`marl/envs/citylearn_adapter.py` and `marl/envs/district_env.py`.

[2] K. Nweye, M. Langtry, R. Choudhary, G. Z. Nagy, "The CityLearn Challenge
2023 Dataset," Texas Data Repository, 2024, doi:10.18738/T8/SXFWTI.
*Used for:* the six-building Phase 3.3 dataset and schema.

## Algorithms and code

[10] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, Y. Wu, "The
Surprising Effectiveness of PPO in Cooperative Multi-Agent Games," NeurIPS 2022.
*Used for:* MAPPO, through the authors' reference implementation
[marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy) @ `de66d7a`
(MIT license), vendored in `third_party/onpolicy` (see its `VENDORED.md`).

[8] W. Kim, M. Cho, Y. Sung, "Message-Dropout: An Efficient Training Method for
Multi-Agent Deep Reinforcement Learning," AAAI 2019.
*Used for:* random-loss training (M1-RND).

[6] A. Khouja, I. Jendoubi, O. Mahjoub, O. Mahfoudhi, R. de Kock, S. Singh,
C. Formanek, "Characterizing MARL for Energy Control: A Multi-KPI Benchmark on
the CityLearn Environment," arXiv:2602.19223, 2026.
*Used for:* the IPPO arm (I0) and the critic-input fallback.

## Design and related work

[3] K. Nweye, B. Liu, P. Stone, Z. Nagy, "Real-world challenges for multi-agent
reinforcement learning in grid-interactive buildings," Energy and AI 10, 100202,
2022 (arXiv:2112.06127).
*Used for:* motivation (missing observations, challenge C5) and the comparison
with shared-forecast agents.

[4] S. Sajid, B. Li, B. Berehman, Q. Guo, Y. Kang, M. Athar, A. Muqtadir,
"Decentralized Multi-Agent Reinforcement Learning Control of Residential Battery
Storage for Demand Response," Energies 18(21), 5712, 2025.
*Used for:* the battery-synchrony motivation.

[5] C. C. Costea, "Multi-Agent Reinforcement Learning Building Energy
Optimization: Taxonomy and Experimental Evaluation in CityLearn," M.Sc. thesis,
Politecnico di Torino, 2026, https://webthesis.biblio.polito.it/40396/.
*Used for:* related work on MAPPO in CityLearn.

[7] S. Zaregarizi, K. Yavari, "Multi-agent storage dispatch in building
communities: Information availability, inter-building dynamics, and the
distribution of benefits," Energy and Buildings, 2026,
doi:10.1016/j.enbuild.2026.118218.
*Used for:* information availability and reporting effect-size bounds.

[9] H. Zhou, A. Aral, I. Brandić, M. Erol-Kantarci, "Multiagent Bayesian Deep
Reinforcement Learning for Microgrid Energy Management Under Communication
Failures," IEEE Internet of Things Journal, 2021.
*Used for:* related work on message loss in energy management.

## Software

PyTorch, NumPy, pandas, Gymnasium, SciPy, Matplotlib and pytest, at the versions
pinned in `requirements.txt`.

## AI assistance

Claude (Anthropic), 2026. Claude assisted in parts of the implementation 
and created most of the test cases in `tests/`.
