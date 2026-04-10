## Offline AI file sorter.
This is a Python project, focused on using actual *context* of the files that is used for sorting.

In its core a Gemma 4 SLM is used (E2B/E4B on user's choice).

---

## Usage
Simply clone this github repo into the place of your choosing. Then, install all dependenciies from ```requirements.txt``` via pip.

After that run ```python3 vsort.py```.

Follow the onboarding prompts, and then the sorting will start.

It can take from 15-60 minutes, depending on your device specs (SLM is hosted on **your** hardware, your data is not going anywhere!).

---

## Additional flags
```--reset```: resets onboarding for changing config.

```--verbose```: shows more info about the process.

```--sort```: for headless launch (in task scheduler).

```--think```: shows reasoning of the SLM, so you know how exactly the model is sorting your files.

:warning: ```--yolo```: EXPERIMENTAL, yeets all the files in one take, while increasing the maximum context. (Default - batches of 15 files, then finalise all contexts)


## Supported OS
Should be every OS that has python 3 on it?
