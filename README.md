
# Design E 344 Test Station Application

This is a Python application developed to perform various audio signal processing tasks, including generating and measuring audio signals. The application uses `PyAudio` for audio I/O and `Tkinter` for the GUI.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [License](#license)

## Features

- Generate and measure audio signals.
- Visualize signals in both time and frequency domains.
- Perform testing on stereo-to-mono conversions, pre-amplifiers, and tone controls.

## Requirements

- Python 3.7 (due to PyAudio wheel compatibility)
- `venv` for managing the virtual environment
- Windows 64-bit OS (as the PyAudio wheel is specific to this platform)

## Installation

Follow the steps below to set up the environment and install the required dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/joshuasello/design-e344-test-station.git
cd design-e344-test-station
```

### 2. Create a Virtual Environment

To avoid conflicts with other Python projects, it's recommended to create a virtual environment.

```bash
python3.7 -m venv venv
```

### 3. Activate the Virtual Environment

- On Windows:
  ```bash
  venv\Scripts\activate
  ```

- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies

#### Install PyAudio

You will need to install PyAudio using a precompiled wheel, as it might not install directly via `pip` due to missing build tools.

Download the wheel file:
[PyAudio-0.2.11-cp37-cp37m-win_amd64.whl](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

Install it using `pip`:

```bash
pip install path\to\PyAudio-0.2.11-cp37-cp37m-win_amd64.whl
```

Replace `path\to\PyAudio-0.2.11-cp37-cp37m-win_amd64.whl` with the actual path to the downloaded file.

#### Install Other Dependencies

```bash
pip install -r requirements.txt
```

If the `requirements.txt` file doesn't exist, create it and add the following dependencies:

```text
numpy
matplotlib
pyaudio
```

Then, run the installation:

```bash
pip install -r requirements.txt
```

## Running the Application

To start the application, ensure that your virtual environment is activated and run the following command:

```bash
python main.py
```

Replace `main.py` with the actual name of your main script file if it's different.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Explanation:

- **Requirements**: It highlights the requirement for Python 3.7 and Windows 64-bit, which are necessary for the provided PyAudio wheel.
- **Installation**: Detailed steps to set up a virtual environment, install dependencies, and specifically install the PyAudio wheel.
- **Running the Application**: Instructions on how to run the application after setting everything up.
- **License**: Placeholder for your project's license details.