from kword import kword

kword(
    # Charset
    sourceFn="sc-3tone.png",
    slicesX=50,
    slicesY=34,
    xPad=0,  # Usually 0
    yPad=0,  # Usually 0
    # Target
    targetFn="maisie-williams.png",
    rowLength=20,  # Width of the output typable in character columns.
    # Find optimal crop?
    autoCrop=False,
    # Manually crop
    zoom=0,  # Between 0 and 3 (at 4, image is same as rowLength+1 at zoom 0)
    shiftLeft=0,  # Between 0 and 3
    shiftUp=0,  # Between 0 and 3
    # Initial state
    initMode="euclidean",  # ['euclidean', 'blend', 'angular', 'random', None]
    initOnly=False,  # Stop after initializing (don't optimize)
    initPriority=True,  # Initialize in priority order (priority only calculated once)
    initComposite=True,  # Subtract already placed ink from target image
    subtractiveScale=64,  # Between 0 (full subtraction from target) and 255 (same as initComposite=False)
    initBrighten=0,  # Raises black level, compressesing dynamic range. Between 0 (no brightening) and 1 (completely white)
    # Add an overtype pass
    resume=None,  # ['pass_0', None]
    # Similarity metric
    mode="amse",  # ['amse', 'mse', 'ssim', 'blend']
    asymmetry=0.2,  # asymmetry > 0 penalizes "sins of commission" more harshly
    # Search technique
    search="firstBetter",  # ['greedy', 'firstBetter', 'incrK', 'simAnneal']
    maxVisits=5,  # Stopping condition, necessary for stochastic search
    initTemp=0.1,  # For simAnneal
    initK=5,  # For incrK: k = initK * numVisits
    # Saving
    genLayers=False,  # Save the 4 typable layers
    saveChars=False,  # Save the chopped character set
    # Logging
    printEvery=10,  # How many selections between progress updates?
)
