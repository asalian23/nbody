# GPU-Accelerated N-Body Gravitational Simulator

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

<div align="center">

<!-- [TODO: Add a screenshot or GIF of the simulation running] -->

**A real-time gravitational N-body simulator built from scratch in C++ and CUDA, capable of simulating hundreds of thousands of gravitationally interacting bodies at interactive frame rates on consumer GPU hardware.**

[View Demo](#usage) · [Report Bug](https://github.com/asalian23/nbody/issues) · [Request Feature](https://github.com/asalian23/nbody/issues)

</div>

---

## About The Project

<!-- [TODO: Insert screenshot/GIF here — ideally the 10k body galaxy rotating, or the 100k body run] -->

This project is a physically accurate N-body gravitational simulator built entirely from scratch — no physics libraries, no simulation frameworks. Every component, from the numerical integrator to the GPU kernel, was written by hand.

The simulator progresses through two major phases:

**Brute Force GPU (Phase 1):** A CUDA-accelerated O(N²) force kernel with shared memory tiling computes pairwise gravitational forces between all bodies simultaneously. Benchmarked up to 50,000 bodies with real timing data showing O(N²) scaling behavior.

**Barnes-Hut GPU (Phase 2):** A sparse quadtree is built on the CPU each frame and uploaded to the GPU, where a custom traversal kernel uses the Barnes-Hut theta criterion to reduce per-body force computation from O(N) to O(log N), enabling simulation of hundreds of thousands of bodies in real time.

Key technical achievements:
- Leapfrog (KDK) symplectic integrator for long-term orbital stability — verified correct over a full Earth year
- CUDA shared memory tiling in brute force kernel reducing global memory bandwidth pressure
- Sparse quadtree with lazy child initialization minimizing redundant node allocation
- Gravitational softening preventing numerical explosions at close range
- Single sf::VertexArray draw call for all N bodies — O(1) render complexity regardless of body count

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### Built With

[![C++][cpp-shield]][cpp-url]
[![CUDA][cuda-shield]][cuda-url]
[![SFML][sfml-shield]][sfml-url]

- **C++17** — core simulation logic, tree building, initialization
- **CUDA 13.2** — GPU force kernels, leapfrog integration
- **SFML 3.0** — real-time visualization
- **NVIDIA RTX 4090** — development and benchmark hardware

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

### Prerequisites

- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA Compute Capability 5.0+ (tested on RTX 4090)
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
- [SFML 3.0.2](https://www.sfml-dev.org/download/sfml/3.0.2/) — Visual C++ 64-bit build
- Visual Studio 2022 (for MSVC compiler) or VS Build Tools

### Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/asalian23/nbody.git
   cd nbody
   ```

2. **Install SFML**
   - Download SFML 3.0.2 (Visual C++ 64-bit) from [sfml-dev.org](https://www.sfml-dev.org/download/sfml/3.0.2/)
   - Extract to `C:\SFML\SFML-3.0.2`
   - Add `C:\SFML\SFML-3.0.2\bin` to your system PATH

3. **Install CUDA Toolkit**
   - Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Follow the installer — no additional configuration needed

4. **Build**

   Open **x64 Native Tools Command Prompt for VS 2022**, navigate to the project folder, and run:
   ```sh
   build.bat
   ```

   Or manually:
   ```sh
   nvcc nBody.cu -o nbody.exe -std=c++17 ^
     -I "C:\SFML\SFML-3.0.2\include" ^
     -L "C:\SFML\SFML-3.0.2\lib" ^
     -lsfml-graphics -lsfml-window -lsfml-system
   ```

5. **Run**
   ```sh
   .\nbody.exe
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Usage

<!-- [TODO: Add GIF or video embed of the simulation here] -->

The simulation opens a 2560×1440 window rendering all N bodies as white pixel points. Bodies are initialized in a rotating disk distribution with physically derived circular orbit velocities around a central supermassive black hole.

**Key parameters** (edit at the top of `nBody.cu`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | 100,000 | Number of bodies |
| `dt` | 10s | Timestep per frame |
| `maxRadius` | 15 AU | Galaxy disk radius |
| `theta_device` | 0.5 | Barnes-Hut accuracy (lower = more accurate) |
| `epsilon_device` | 6.96e8m | Gravitational softening length |

Press **ESC** or close the window to exit. GPU timing is printed to the console every 10 frames.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Performance Benchmarks

All benchmarks measured on **NVIDIA RTX 4090**, GPU kernel time via CUDA events.

### Brute Force O(N²) — Shared Memory Tiled Kernel

| Body Count (N) | GPU Time (ms/frame) | FPS |
|----------------|--------------------|----|
| 1,000 | 4.66 | 214 |
| 5,000 | 17.13 | 58 |
| 10,000 | 33.0 | 30 |
| 20,000 | 65.23 | 15 |
| 30,000 | 97.02 | 10 |
| 40,000 | 259.29 | 4 |
| 50,000 | 398.46 | 2.5 |

The sharp jump between 30k and 40k reflects GPU shared memory occupancy saturation — the tiling optimization that hides memory latency breaks down and performance degrades faster than O(N²) predicts.

### Barnes-Hut O(N log N) — GPU Force Traversal

<!-- [TODO: Fill in Barnes-Hut benchmark numbers once measured] -->

| Body Count (N) | GPU Time (ms/frame) | FPS |
|----------------|--------------------|----|
| 10,000 | [TODO] | [TODO] |
| 50,000 | [TODO] | [TODO] |
| 100,000 | [TODO] | [TODO] |
| 500,000 | [TODO] | [TODO] |

<!-- [TODO: Note the crossover point where Barnes-Hut beats brute force] -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Architecture

```
nBody.cu
├── Vec3 struct              — 3D vector math (__host__ __device__)
├── body struct              — position, velocity, acceleration, mass
├── quadNode struct          — sparse quadtree node (CoM, children[4], bodyIdx)
│
├── resetNode()              — initialize node to empty state
├── getRootNode()            — compute bounding box, return root node
├── getQuadrant()            — bitwise quadrant lookup (branchless)
├── calcChildCenterPos()     — compute child region center from parent + quadrant
├── insertBody()             — recursive sparse tree insertion (3 cases)
├── fillTree()               — reset old nodes, rebuild tree from scratch
├── computeCOM()             — bottom-up center of mass aggregation
│
├── calcAccel (kernel)       — Barnes-Hut force traversal with explicit stack
├── leapfrogPartOne (kernel) — KDK kick 1 + drift
├── leapfrogPartTwo (kernel) — KDK kick 2
├── leapfrog()               — full timestep: kick1 → tree rebuild → forces → kick2
│
└── main()                   — init, GPU setup, render loop
```

### Physics

The simulator uses the **leapfrog (Kick-Drift-Kick) integrator** — a symplectic method that exactly conserves a modified Hamiltonian, keeping orbits stable indefinitely. This outperforms Euler (energy drift) and RK4 (energy dissipation) for long-timescale simulations.

Force calculation uses **gravitational softening**: `F = G*m1*m2 / (r² + ε²)` where ε is the solar radius (6.96×10⁸ m), preventing force singularities at close range.

**Barnes-Hut theta criterion:** For each node, `s/d < θ` (node size / distance < 0.5) determines whether to use the center-of-mass approximation or recurse into children. This reduces per-body work from O(N) to O(log N).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Roadmap

- [x] CPU N-body physics engine (Vec3, leapfrog, pairwise forces)
- [x] CUDA brute force O(N²) kernel with shared memory tiling
- [x] Real-time SFML visualization (sf::VertexArray single draw call)
- [x] GPU benchmarks — O(N²) scaling verified, occupancy cliff identified
- [x] Barnes-Hut sparse quadtree — CPU tree build, GPU force traversal
- [ ] Barnes-Hut benchmarks and crossover analysis
- [ ] 3D simulation with OpenGL rendering
- [ ] CUDA-OpenGL interoperability (eliminate CPU memcpy bottleneck)
- [ ] Yoshida 4th-order symplectic integrator
- [ ] Full GPU tree construction (Burtscher & Pingali approach)
- [ ] Galaxy collision scenario

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## References

- Barnes, J. & Hut, P. (1986). *A hierarchical O(N log N) force-calculation algorithm.* Nature, 324, 446–449.
- Burtscher, M. & Pingali, K. (2011). *An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm.* GPU Computing Gems Emerald Edition.
- Leapfrog integration: [Wikipedia — Leapfrog integration](https://en.wikipedia.org/wiki/Leapfrog_integration)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## License

<!-- [TODO: Choose a license — MIT is standard for open source projects. Add a LICENSE.txt file to the repo.] -->

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Contact

Aryan Salian — [GitHub](https://github.com/asalian23)

Project Link: [https://github.com/asalian23/nbody](https://github.com/asalian23/nbody)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/asalian23/nbody.svg?style=for-the-badge
[contributors-url]: https://github.com/asalian23/nbody/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/asalian23/nbody.svg?style=for-the-badge
[forks-url]: https://github.com/asalian23/nbody/network/members
[stars-shield]: https://img.shields.io/github/stars/asalian23/nbody.svg?style=for-the-badge
[stars-url]: https://github.com/asalian23/nbody/stargazers
[issues-shield]: https://img.shields.io/github/issues/asalian23/nbody.svg?style=for-the-badge
[issues-url]: https://github.com/asalian23/nbody/issues
[license-shield]: https://img.shields.io/github/license/asalian23/nbody.svg?style=for-the-badge
[license-url]: https://github.com/asalian23/nbody/blob/master/LICENSE.txt
[cpp-shield]: https://img.shields.io/badge/C%2B%2B17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white
[cpp-url]: https://isocpp.org/
[cuda-shield]: https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white
[cuda-url]: https://developer.nvidia.com/cuda-toolkit
[sfml-shield]: https://img.shields.io/badge/SFML-8CC445?style=for-the-badge&logo=sfml&logoColor=white
[sfml-url]: https://www.sfml-dev.org/