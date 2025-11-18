# Test scripts 

## Loader Testing

```bash
# Basic loading
python debug/loader.py load --city beijing --sample

# Shuffled data
python debug/loader.py load-shuffled --city boston --val-ratio 0.2

# Test processor directly
python debug/loader.py test-processor --city london

# Test file handlers
python debug/loader.py test-file-handler --city tokyo ## Fails -- dimension error`file-handler`

# Performance testing
python debug/loader.py test-performance --city beijing

# Schema validation
python debug/loader.py test-schema --city moscow # Multiple errors --no errors, missing required columns, check data

# Data analysis
python debug/loader.py analyze --city beijing --loglevel DEBUG
```



## Coords Testing

Script testing for the various functions for vector coordinate system representation. 

```bash
    # Conversion from cartesian to spherical
    python coords.py --cart-to-sph

    # Conversion from spherical to cartesian
    python coords.py --sph-to-cart

    # Converts cartesian to spherical and convert back
    python coords.py --roundtrip

    # Adding two angles together     
    python coords.py --add-angles

    # Subtracting two angles for performance testing
    python coords.py --sub-angles

    # Rotation first adds the angles then subtract the angle once again
    python coords.py --rotate
```
-----

Additional operations for customize the test further, these allow some modifications to each test.
```bash
    # Integer value of samples to generate
    --n-samples 1000

    # precision definition to each generated vector
    --dtype float64 / float32

    --seed 42 
    --debug
    --trials 100
```