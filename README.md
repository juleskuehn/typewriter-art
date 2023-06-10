# Algorithmic typewriter art: Can 1000 words paint a picture?

![Process overview: Input photograph and character set transformed into 4 typeable layers, then typed in overlapping fashion.](https://github.com/juleskuehn/typewriter-art/raw/master/process.png)

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

## Usage

Requires Python **3.10**. It seems that Numba has some bugs with Python 3.11.

```
git clone https://github.com/juleskuehn/typewriter-art
pip install -r requirements.txt
cd src
python optimize.py
```

Once optimization is finished, press any key to dismiss the mockup windows.

Final mockup, typeable layers and `choices.json` will be output to `src/results`.

### Command line options
```
> python optimize.py -h

usage: optimize.py [-h] [--charset CHARSET] [--target TARGET] [--row_length ROW_LENGTH] [--num_loops NUM_LOOPS] [--init_mode INIT_MODE]
                   [--asymmetry ASYMMETRY] [--search SEARCH] [--init_temp INIT_TEMP] [--layers LAYERS] [--display DISPLAY] [--shuffle SHUFFLE]

options:
  -h, --help            show this help message and exit
  --charset CHARSET, -c CHARSET
                        Path to charset folder containing config.json and image (default sc-2)
  --target TARGET, -t TARGET
                        Path to target image in ./images (default mwdog_crop.png)
  --row_length ROW_LENGTH, -r ROW_LENGTH
                        Number of characters per row; determines image size (default 30)
  --num_loops NUM_LOOPS, -n NUM_LOOPS
                        Number of times to optimize each layer (default 20)
  --init_mode INIT_MODE, -i INIT_MODE
                        Start with random or blank image (default random)
  --asymmetry ASYMMETRY, -a ASYMMETRY
                        Asymmetry of mean squared error function (default 0.1)
  --search SEARCH, -s SEARCH
                        Search algorithm. Options: 'simAnneal', 'greedy' (default simAnneal)
  --init_temp INIT_TEMP, -temp INIT_TEMP
                        Initial temperature for simulated annealing (default 0.001)
  --layers LAYERS, -l LAYERS
                        Key to layers.json for offsets - how many layers, where to place them (default 16x1)
  --display DISPLAY, -d DISPLAY
                        Display the mockup every X iterations (1 == most often) or 0 to not display (default 1)
  --shuffle SHUFFLE, -sh SHUFFLE
                        Shuffle the order of the layers each optimization loop (default True)
```

## Learn more

The descriptions below are a bit dated (do not reflect subsequent improvements to the codebase).

[Watch the 5 minute conference presentation](https://www.youtube.com/watch?v=usa6kupyCjA)

[Read the conference paper](https://github.com/juleskuehn/typewriter-art/gi2021-13.pdf)

*Presented at [Graphics Interface 2021](https://graphicsinterface.org/proceedings/gi2021/gi2021-13/)*
