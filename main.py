""" Design E 344 Test Station Application

"""

__author__ = "Joshua Sello"
__email__ = "joshuasello@gmail.com"


# --------------------------------------------------
#   Imports
# --------------------------------------------------

import json
import time
import queue
import pyaudio
import threading
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# --------------------------------------------------
#   Classes
# --------------------------------------------------

CHUNK_SIZE = 2_048
FORMAT = pyaudio.paFloat32
NUM_INPUT_CHANNELS = 1
NUM_OUTPUT_CHANNELS = 2
RATE = 44_100
TIME_DOMAIN_PLOT_MIN = -30
TIME_DOMAIN_PLOT_MAX = 30
INPUT_PERIOD = CHUNK_SIZE / RATE

# --------------------------------------------------
#   Classes
# --------------------------------------------------

class Application:
    def __init__(self, root: tk, config: dict) -> None:
        self.root = root
        self.root.title("Design E 344 Test Station")
        # Maximize the window on startup
        self.root.state("zoomed")

        self.__reference_voltage = config["reference_voltage"]
        self.__stereo_to_mono_right_output_frequency = config["stereo_to_mono_right_output_frequency"]
        self.__stereo_to_mono_left_output_frequency = config["stereo_to_mono_left_output_frequency"]
        self.__stereo_to_mono_output_peak = config["stereo_to_mono_output_peak"]
        self.__pre_amplifier_output_frequency = config["pre_amplifier_output_frequency"]
        self.__pre_amplifier_output_peak = config["pre_amplifier_output_peak"]
        self.__tone_control_low_output_frequency = config["tone_control_low_output_frequency"]
        self.__tone_control_mid_output_frequency = config["tone_control_mid_output_frequency"]
        self.__tone_control_high_output_frequency = config["tone_control_high_output_frequency"]

        # Initialize pyaudio
        self.audio = pyaudio.PyAudio()
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=NUM_INPUT_CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        self.output_stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=NUM_OUTPUT_CHANNELS,
            rate=RATE,
            output=True,
            stream_callback=self.__output_stream_callback
        )

        self.__output_on = False
        self.__is_measuring = False
        self.__output_thread = None

        self.__is_outputting = True
        self.__output_buffer = np.zeros(CHUNK_SIZE * NUM_OUTPUT_CHANNELS, dtype=np.float32)
        self.__output_left_channel = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.__output_right_channel = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.__output_thread = threading.Thread(target=self.__output_loop, daemon=True)
        self.__output_thread.start()

        self.__audio_data = np.zeros(CHUNK_SIZE * NUM_INPUT_CHANNELS, dtype=np.float32)

        # Initialise Matplotlib Figures displaying the measured signal in the time and frequency domain
        self.fig, (self.ax_time, self.ax_freq) = plt.subplots(2, 1, figsize=(10, 8))

        # Time-Domain Plot
        self.time_data = np.linspace(0, CHUNK_SIZE / RATE, CHUNK_SIZE, endpoint=False)
        self.time_line, = self.ax_time.plot(self.time_data, np.zeros(CHUNK_SIZE))
        self.__setup_time_domain_plot()

        # Frequency-Domain Plot
        self.freq_data = np.zeros(CHUNK_SIZE // 2)
        self.freq_line, = self.ax_freq.semilogx(self.freq_data)
        self.__setup_frequency_domain_plot()

        # Configure the grid layout for the main window
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Create Control Side Panel
        self.control_side_panel = ttk.Frame(root, width=300)
        self.control_side_panel.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

        # Embed the plot in the tkinter window
        self.plots_canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.plots_canvas.get_tk_widget().grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

        # Create Results Side Panel
        self.results_side_panel = ttk.Frame(root, width=300)
        self.results_side_panel.grid(row=0, column=2, sticky="nswe", padx=10, pady=10)
        self.results_side_panel.grid_propagate(False)

        self.__control_instructions_section()

        # Setup Section
        # Stores the gain to apply to the output such that the device's audio is exactly 1 volt-peak
        self.max_output_button = None
        self.output_peak_var = tk.DoubleVar(value=1.0)
        self.input_gain_var = tk.DoubleVar(value=1.0)
        self.__control_setup_section()

        self.__control_outputting_section()

        self.__control_measuring_section()

        self.__control_measurements_section()

        self.__testing_stereo_to_mono_section()

        self.__testing_pre_amplifier_section()

        self.__tone_control_testing_mode_var = tk.StringVar()
        self.__tone_control_testing_mode_var.set("TESTING-LOW")
        self.__testing_tone_control_testing_section()

    def on_closing(self) -> None:
        self.__is_measuring = False
        self.__is_outputting = False
        self.__output_on = False

        self.output_stream.stop_stream()
        self.output_stream.close()

        self.input_stream.stop_stream()
        self.input_stream.close()

        self.root.destroy()
        self.audio.terminate()

    def __setup_time_domain_plot(self) -> None:
        sampling_period = 1 / RATE
        self.ax_time.set_title("Time-Domain Signal")
        self.ax_time.set_xlabel("Samples")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.set_xlim(0, CHUNK_SIZE * sampling_period)
        self.ax_time.set_ylim(TIME_DOMAIN_PLOT_MIN, TIME_DOMAIN_PLOT_MAX)
        self.ax_time.grid(True)

    def __setup_frequency_domain_plot(self) -> None:
        self.ax_freq.set_title("Frequency-Domain Signal (FFT)")
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_ylabel("Magnitude")
        self.ax_freq.set_xlim(20, RATE / 2)
        self.ax_freq.set_ylim(0, 1000)
        self.ax_freq.grid(True)

    def __control_instructions_section(self) -> None:
        instructions_frame = ttk.LabelFrame(self.control_side_panel, text="Instructions")
        instructions_frame.pack(fill=tk.X, pady=8, padx=8)
        ttk.Label(instructions_frame, text="⚠️ Make sure no other applications are outputting audio.").pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(instructions_frame, text="1. Calibrate input voltage").pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(instructions_frame, text="2. Calibrate output voltage").pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(instructions_frame, text="3. Turn on application audio output").pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(instructions_frame, text="4. Turn on measuring").pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(instructions_frame, text="5. Test stereo-to-mono").pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(instructions_frame, text="6. Test pre-amplifier").pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(instructions_frame, text="7. Test tone control").pack(fill=tk.X, padx=8, pady=2)

    def __control_setup_section(self) -> None:
        section_frame = ttk.LabelFrame(self.control_side_panel, text="Setup")
        section_frame.pack(fill=tk.X, pady=8, padx=8)

        ttk.Label(section_frame, text="(1) Measure device audio peak voltage").pack(fill=tk.X, padx=8, pady=2)

        ttk.Button(
            section_frame,
            text="Generate Calibration Output",
            command=self.__generate_calibration_output
        ).pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(section_frame, text="Output peak voltage:").pack(fill=tk.X, padx=8, pady=4)
        ttk.Entry(section_frame, textvariable=self.output_peak_var).pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(section_frame, text="Input gain:").pack(fill=tk.X, padx=8, pady=4)
        ttk.Entry(section_frame, textvariable=self.input_gain_var).pack(fill=tk.X, padx=8, pady=4)

    def __control_outputting_section(self) -> None:
        section_frame = tk.LabelFrame(self.control_side_panel, text="Output Signal")
        section_frame.pack(fill=tk.X, pady=8, padx=8)

        self.toggle_button = ttk.Button(section_frame, text="Turn on", command=self.__toggle_signal)
        self.toggle_button.pack(pady=4)

    def __toggle_signal(self) -> None:
        self.__output_on = not self.__output_on
        if self.__output_on:
            self.toggle_button.config(text="Turn off")
        else:
            self.toggle_button.config(text="Turn on")

    def __output_loop(self) -> None:
        while self.__is_outputting:
            if self.__output_on:
                self.__output_buffer[0::2] = self.__output_left_channel
                self.__output_buffer[1::2] = self.__output_right_channel
            else:
                self.__output_buffer[:] = 0.0
            time.sleep(CHUNK_SIZE / RATE)

    def __control_measuring_section(self) -> None:
        section_frame = tk.LabelFrame(self.control_side_panel, text="Measuring")
        section_frame.pack(fill=tk.X, pady=8, padx=8)

        # Create Start and Stop buttons in the side panel
        start_button = ttk.Button(section_frame, text="Start", command=self.__start_measuring)
        start_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4, pady=4)

        stop_button = ttk.Button(section_frame, text="Stop", command=self.__stop_measuring)
        stop_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4, pady=4)

    def __control_measurements_section(self) -> None:
        measurements_frame = ttk.LabelFrame(self.control_side_panel, text="Measurements")
        measurements_frame.pack(fill=tk.X, pady=8, padx=8)

        self.__peak_voltage_label = tk.Label(measurements_frame, text="Peak Voltage: -- Vp")
        self.__peak_voltage_label.pack(side=tk.TOP, anchor='w', padx=8, pady=2)

        self.__thd_label = tk.Label(measurements_frame, text="THD: --%")
        self.__thd_label.pack(side=tk.TOP, anchor='w', padx=8, pady=2)

        self.__peak_frequencies_label = tk.Label(measurements_frame, text="Peak Frequencies: -- Hz")
        self.__peak_frequencies_label.pack(side=tk.TOP, anchor='w', padx=8, pady=2)

    def __testing_stereo_to_mono_section(self) -> None:
        section_frame = ttk.LabelFrame(self.results_side_panel, text="Stereo-to-Mono Testing")
        section_frame.pack(fill=tk.X, pady=8, padx=8)

        tk.Label(
            section_frame,
            text="Generate and apply a 1 kHz/1V peak signal to the left input and a 2 kHz/1V peak "
                 "signal to the right input",
            wraplength=200
        ).pack(fill=tk.X, side=tk.TOP, anchor='w', padx=8, pady=4)

        generate_output_button = ttk.Button(
            section_frame, text="Generate Output", command=self.__generate_stereo_to_mono_testing_output
        )
        generate_output_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=8, pady=4)

    def __testing_pre_amplifier_section(self) -> None:
        section_frame = ttk.LabelFrame(self.results_side_panel, text="Pre-Amplifier Testing")
        section_frame.pack(fill=tk.X, pady=8, padx=8)

        tk.Label(
            section_frame,
            text="Apply a 1 kHz, 1V peak signal to the inputs, measure the pre-amp output, and ensure "
                 "the peak voltage exceeds 20V",
            wraplength=200
        ).pack(fill=tk.X, side=tk.TOP, anchor='w', padx=8, pady=4)

        generate_output_button = ttk.Button(
            section_frame, text="Generate Output", command=self.__generate_pre_amplifier_testing_output
        )
        generate_output_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=8, pady=4)

    def __testing_tone_control_testing_section(self) -> None:
        section_frame = ttk.LabelFrame(self.results_side_panel, text="Tone Control Testing")
        section_frame.pack(fill=tk.X, pady=8, padx=8)

        tk.Label(
            section_frame,
            text=(
                "For each tone control:\n"
                "- Low: Apply a 100 Hz, 1V peak signal, adjust the Bass control, and verify output voltage is "
                "within specified ranges.\n"
                "- Mid: Apply a 1 kHz, 1V peak signal, adjust the Mid control, and verify output voltage is "
                "within specified ranges.\n"
                "- High: Apply a 10 kHz, 1V peak signal, adjust the High control, and verify output voltage is "
                "within specified ranges."
            ),
            wraplength=200
        ).pack(fill=tk.X, side=tk.TOP, anchor='w', padx=8, pady=4)

        options = ["TESTING-LOW", "TESTING-MID", "TESTING-HIGH"]
        dropdown = tk.OptionMenu(section_frame, self.__tone_control_testing_mode_var, *options)
        dropdown.pack(side=tk.TOP, expand=True, fill=tk.X, padx=8, pady=4)

        generate_output_button = ttk.Button(
            section_frame, text="Generate Output", command=self.__generate_tone_control_testing_output
        )
        generate_output_button.pack(side=tk.TOP, expand=True, fill=tk.X, padx=8, pady=4)

        ttk.Separator(section_frame, orient="horizontal").pack(fill=tk.X, pady=8)

        (
            ttk.Button(section_frame, text="Check Cut", command=self.__check_tone_control_cut)
            .pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)
        )
        (
            ttk.Button(section_frame, text="Check Boost", command=self.__check_tone_control_boost)
            .pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)
        )
        (
            ttk.Button(section_frame, text="Check Low Adj. Isolation", command=self.__check_tone_control_isolation_low)
            .pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)
        )
        (
            ttk.Button(section_frame, text="Check High Adj. Isolation", command=self.__check_tone_control_isolation_high)
            .pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)
        )

    def __generate_calibration_output(self) -> None:
        amplitudes = [1.0]
        frequencies = [1_000.0]
        self.__set_left_channel_sine_waves(
            amplitudes=amplitudes,
            frequencies=frequencies,
        )
        self.__set_right_channel_sine_waves(
            amplitudes=amplitudes,
            frequencies=frequencies,
        )

    def __start_measuring(self) -> None:
        if self.__is_measuring:
            return

        self.__is_measuring = True
        self.__update_measurements()

    def __stop_measuring(self) -> None:
        self.__is_measuring = False

        if self.__output_thread:
            self.__output_thread.join()
            self.__output_thread = None

    def __update_measurements(self) -> None:
        uses_triggering = True
        if self.__is_measuring:
            data = self.input_stream.read(CHUNK_SIZE)
            audio_data = np.frombuffer(data, dtype=np.float32)

            # Apply gain to the audio data
            input_gain = self.input_gain_var.get()
            audio_data = audio_data * input_gain

            self.__display_peak_voltage(audio_data.max())

            # Compute FFT and corresponding frequencies
            fft_data = np.abs(np.fft.rfft(audio_data[:CHUNK_SIZE]))
            frequencies = np.fft.rfftfreq(CHUNK_SIZE, 1 / RATE)

            peak_frequencies = self.__find_top_n_peaks(frequencies, fft_data, n=3)
            self.__display_peak_frequencies(peak_frequencies)

            self.__audio_data = audio_data

            if uses_triggering:
                # Smooth the signal slightly to avoid detecting false peaks due to noise
                smoothed_data = self.__smooth_signal(audio_data)

                # Find the zero-crossing closest to the peak
                trigger_index = self.__find_trigger_point(smoothed_data)

                # Shift the audio data so that the trigger point is at the beginning
                audio_data = np.roll(audio_data, -trigger_index)

            # Update the time-domain plot
            self.time_line.set_ydata(audio_data[:CHUNK_SIZE])

            # Ensure that the data lengths match
            fft_data = fft_data[:len(frequencies)]

            # Update frequency-domain plot
            self.freq_line.set_xdata(frequencies)
            self.freq_line.set_ydata(fft_data)
            self.ax_freq.relim()
            self.ax_freq.autoscale_view()

            # Calculate and display THD
            thd_value = self.__calculate_thd(fft_data, frequencies)
            self.__display_thd(thd_value)

            self.plots_canvas.draw()
            self.root.after(10, self.__update_measurements)  # Update the plot after 10ms

    def __display_peak_voltage(self, value: float) -> None:
        thd_text = f"Peak Voltage: {value: 04.2f} Vp"
        self.__peak_voltage_label.config(text=thd_text)

    def __display_thd(self, value: float) -> None:
        thd_text = f"THD: {value: 06.2%}"
        self.__thd_label.config(text=thd_text)

    def __display_peak_frequencies(self, frequencies: np.ndarray) -> None:
        frequency_strings = [f"{frequency: 08.2f}" for frequency in frequencies]
        self.__peak_frequencies_label.config(text=f"Peak Frequencies: \n{', '.join(frequency_strings)} Hz")

    def __reset_output(self) -> None:
        self.__is_playing = False

    def __reset_plots(self) -> None:
        self.is_plotting = False

    def __get_gain(self) -> float:
        output_peak = self.output_peak_var.get()  # In volts
        return self.__reference_voltage / output_peak

    def __output_stream_callback(self, in_data, frame_count, time_info, status):
        # Provide the buffer to the audio stream
        return self.__output_buffer.tobytes(), pyaudio.paContinue

    def __set_left_channel_sine_waves(
            self,
            frequencies: list[float],
            amplitudes: list[float],
    ) -> None:
        if len(frequencies) != len(amplitudes):
            raise ValueError("frequencies and amplitudes lists must have the same length.")

            # Initialize the buffer to zero
        self.__output_left_channel[:] = 0.0

        # Time vector for one chunk
        t = np.arange(CHUNK_SIZE) / RATE

        output_gain = self.output_peak_var.get()

        # Sum sine waves based on given frequencies and amplitudes
        for freq, amp in zip(frequencies, amplitudes):
            self.__output_left_channel += output_gain * amp * np.sin(2 * np.pi * freq * t).astype(np.float32)

    def __set_right_channel_sine_waves(
            self,
            frequencies: list[float],
            amplitudes: list[float],
    ) -> None:
        if len(frequencies) != len(amplitudes):
            raise ValueError("frequencies and amplitudes lists must have the same length.")

        # Initialize the buffer to zero
        self.__output_right_channel[:] = 0.0

        # Time vector for one chunk
        t = np.arange(CHUNK_SIZE) / RATE

        # Sum sine waves based on given frequencies and amplitudes
        for freq, amp in zip(frequencies, amplitudes):
            self.__output_right_channel += amp * np.sin(2 * np.pi * freq * t).astype(np.float32)

    def __generate_stereo_to_mono_testing_output(self) -> None:
        self.__set_left_channel_sine_waves(
            amplitudes=[self.__stereo_to_mono_output_peak],
            frequencies=[self.__stereo_to_mono_left_output_frequency],
        )
        self.__set_right_channel_sine_waves(
            amplitudes=[self.__stereo_to_mono_output_peak],
            frequencies=[self.__stereo_to_mono_right_output_frequency],
        )

    def __generate_pre_amplifier_testing_output(self) -> None:
        self.__set_left_right_channel_sine_waves([1.0], [1_000])

    def __generate_tone_control_testing_output(self) -> None:
        match self.__tone_control_testing_mode_var.get():
            case "TESTING-LOW":
                self.__set_left_right_channel_sine_waves([1.0], [100])
            case "TESTING-MID":
                self.__set_left_right_channel_sine_waves([1.0], [1_000])
            case "TESTING-HIGH":
                self.__set_left_right_channel_sine_waves([1.0], [10_000])

    def __check_tone_control_cut(self) -> None:
        peak_input = self.__audio_data.max()
        test_passed = peak_input <= 0.5
        messagebox.showinfo(
            "Tone Control Cut Check",
            f"Outcome: {'PASS' if test_passed else 'FAILED'}. \n"
            f"Measured: {peak_input: .2f} Vp\n"
            "Require less than or equal to 0.5 Vp.",
        )

    def __check_tone_control_boost(self) -> None:
        peak_input = self.__audio_data.max()
        test_passed = 2.0 <= peak_input
        messagebox.showinfo(
            "Tone Control Boost Check",
            f"Outcome: {'PASS' if test_passed else 'FAILED'}. \n"
            f"Measured: {peak_input: .2f} Vp\n"
            "Require greater than or equal to 2.0 Vp.",
        )

    def __check_tone_control_isolation_low(self) -> None:
        peak_input = self.__audio_data.max()
        upper_threshold = 0.9
        lower_threshold = 1.1
        test_passed = lower_threshold <= peak_input <= upper_threshold
        messagebox.showinfo(
            "Tone Control Low Adjustment Isolation Check",
            f"Outcome: {'PASS' if test_passed else 'FAILED'}. \n"
            f"Measured: {peak_input: .2f} Vp\n"
            f"Require {lower_threshold} <= {peak_input: .2f} <= {upper_threshold} Vp.",
        )

    def __check_tone_control_isolation_high(self) -> None:
        peak_input = self.__audio_data.max()
        upper_threshold = 0.9
        lower_threshold = 1.1
        test_passed = lower_threshold <= peak_input <= upper_threshold
        messagebox.showinfo(
            "Tone Control High Adjustment Isolation Check",
            f"Outcome: {'PASS' if test_passed else 'FAILED'}. \n"
            f"Measured: {peak_input: .2f} Vp\n"
            f"Require {lower_threshold} <= {peak_input: .2f} <= {upper_threshold} Vp.",
        )

    def __set_left_right_channel_sine_waves(self, amplitudes: list[float], frequencies: list[float]) -> None:
        self.__set_left_channel_sine_waves(
            amplitudes=amplitudes,
            frequencies=frequencies,
        )
        self.__set_right_channel_sine_waves(
            amplitudes=amplitudes,
            frequencies=frequencies,
        )

    @staticmethod
    def __smooth_signal(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply a simple moving average filter to the signal."""
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

    @staticmethod
    def __find_trigger_point(signal: np.ndarray) -> int:
        """
        Find the index of the zero-crossing near the highest peak.
        This will act as the trigger point for aligning the sine wave.
        """
        # Find the peak index
        peak_index = np.argmax(np.abs(signal))

        # Look for the nearest zero-crossing before the peak
        for i in range(peak_index, 0, -1):
            if signal[i] >= 0 > signal[i - 1]:
                return i

        # Fallback to the peak if no zero-crossing is found
        return peak_index

    @staticmethod
    def __find_top_n_peaks(frequencies: np.ndarray, fft_data: np.ndarray, n: int = 3) -> np.ndarray:
        """
        Find the frequencies corresponding to the top N peaks in the FFT data.
        """
        # Sort the FFT data in descending order and get the indices of the top N peaks
        peak_indices = np.argsort(fft_data)[-n:][::-1]

        # Get the corresponding frequencies
        peak_freqs = frequencies[peak_indices]

        # Calibrate
        scalar = 1_000 / 990.53
        peak_freqs *= scalar

        return peak_freqs

    @staticmethod
    def __calculate_thd(fft_data: np.ndarray, frequencies: np.ndarray) -> float:
        """
        Calculate the Total Harmonic Distortion (THD) by extracting amplitudes only at harmonic frequencies.
        """
        # Identify the fundamental frequency component (highest peak)
        fundamental_index = np.argmax(fft_data)
        fundamental_frequency = frequencies[fundamental_index]
        fundamental_power = fft_data[fundamental_index] ** 2

        # Initialize the harmonic power sum
        harmonic_power_sum = 0.0

        if fundamental_index == 0:
            return 0.0

        # Calculate the harmonics (2nd harmonic and beyond)
        for n in range(2, len(frequencies) // fundamental_index):
            harmonic_frequency = n * fundamental_frequency
            harmonic_index = np.argmin(np.abs(frequencies - harmonic_frequency))

            if harmonic_index < len(fft_data):
                harmonic_power_sum += fft_data[harmonic_index] ** 2

        # Calculate THD
        thd = np.sqrt(harmonic_power_sum) / np.sqrt(fundamental_power)

        return thd


# --------------------------------------------------
#   Entry Point
# --------------------------------------------------

if __name__ == '__main__':
    with open('config.json', 'r') as file:
        config = json.load(file)
    root = tk.Tk()
    app = Application(root, config)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
