# Rubidium

Rubidium is a Klipper extension for printing and scanning pressure advance test patterns

It provides a framework to:
- generate and print test patterns
- "run them over" as scans or prints
- record video of that
- analyze the results


## Core concepts

### Patterns
The template, basically something with settings assigned a name

### Modes: scan vs print

Rubidium distinguishes between different modes:

- **Scan mode**  
  Used for recording video of the patterns

- **Print mode**  
  Used for printing the patterns

Both modes use the same underlying pattern definitions.


## Video capture

Rubidium records video during a scan using `ffmpeg`.

right now:
- a continuous session video is recorded
- motion events emit timestamped markers
- markers and metadata are written to a JSON file
- per-segment video clips are generated

---

## Install
To install this plugin, run the installation script using the following command over SSH. 
```
wget -O - https://raw.githubusercontent.com/Contomo/rubidium/main/install.sh | bash
```
This script will download this GitHub repository to your RaspberryPi home directory, and symlink the files in the Klipper extra folder.


## Status

Rubidium is an active work in progress.

> printing works fine  
> single pattern definition works fine  
> scanning works fine*ish*  

> still a timing issue in video recording  
> multiple print/scan/pattern sections untested
> analysis not tested at all just gibberish rn

---

Inspired by [Rubedo](<https://github.com/furrysalamander/rubedo>)  
