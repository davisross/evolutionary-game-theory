# Evolutionary game theory

This repo contains a Python script which implements a genetic algorithm and performs the iterated prisoner's dilemma. It supports introducing noise into the process subject to a configurable noise rate.

The script computes:
- Best strategy found and its fitness
- Time to compute
- Fitness plot over generations

And performs:
- Tournament selection
- Elitism
- Ordered crossover
- Uniform crossover
- Swap mutation
- Scramble mutation

### Prerequisites

```
pip3 install matplotlib
```

### Usage

```
python3 gametheory.py
```

Parameters that control the search process are defined within the script's main function at [line 217](https://github.com/davisross/evolutionary-game-theory/blob/main/gametheory.py#L217). The number of rounds and the noise rate are set at [line 16](https://github.com/davisross/evolutionary-game-theory/blob/main/gametheory.py#L16). The noise rate is set to 5% by default.
