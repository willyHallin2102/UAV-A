# Test scripts 

## Loader Testings

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