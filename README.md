# cat-solver

A naive solver for the game [_Catch the Cat_](https://github.com/ganlvtech/phaser-catch-the-cat) in Deep Q-Networks.

## Prepare the Environments

```bash
$ brew install pdm
$ pdm use
$ pdm install
```

## Play by Yourself

```bash
$ pdm run play
```

## Train a Model

```bash
$ pdm run train [path/to/checkpoint/model]
```

## Evaluate a Model

```bash
$ pdm run eval path/to/model
```
