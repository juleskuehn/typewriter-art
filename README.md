# Algorithmic typewriter art: Can 1000 words paint a picture?

![Process overview: Input photograph and character set transformed into 4 typeable layers, then typed in overlapping fashion.](https://github.com/juleskuehn/typewriter-art/raw/main/process.png)

## Abstract

We present an optimization-based algorithm for converting input photographs into typewriter art. 
Taking advantage of the typist's ability to move the paper in the typewriter, 
the optimization algorithm selects characters for four overlapping, staggered layers of type. 
By typing the characters as instructed, the typist can reproduce the image on the typewriter. 
Compared to text-mode ASCII art, allowing characters to overlap greatly increases tonal range 
and spatial resolution, at the expense of exponentially increasing the search space. 
We use a simulated annealing search to find an approximate solution in this highdimensional 
search space. Considering only one dimension at a time, we measure the effect of changing a 
single character in the simulated typed result, repeatedly iterating over all the characters 
composing the image. Both simulated and physical typed results have a high degree of detail, 
while still being clearly recognizable as type art. The accuracy of the physical typed result is 
largely limited by human error and the mechanics of the typewriter.

## Try it out

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juleskuehn/typewriter-art/blob/main/typewriter_demo.ipynb)

## Learn more

[Watch the 5 minute conference presentation](https://www.youtube.com/watch?v=usa6kupyCjA)

[Read the conference paper](https://github.com/juleskuehn/typewriter-art/gi2021-13.pdf)

*Presented at [Graphics Interface 2021](https://graphicsinterface.org/proceedings/gi2021/gi2021-13/)*
