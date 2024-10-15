import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model
model = load_model('tesla_stock_model.h5')  # Your saved model

# Create a Tkinter window
root = tk.Tk()
root.title("Stock Price Prediction")
root.geometry("700x600")  # Increased size to accommodate plot

# Label for the window
label = tk.Label(root, text="Stock Price Prediction Using LSTM", font=("Arial", 16))
label.pack(pady=10)

# Function to browse and load the CSV file
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        file_entry.delete(0, tk.END)  # Clear the entry field
        file_entry.insert(0, file_path)

# Function to predict future stock prices
def predict_price():
    file_path = file_entry.get()
    try:
        # Load the dataset
        new_df = pd.read_csv(file_path)
        new_data = new_df[['Close']].values

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_new_data = scaler.fit_transform(new_data)

        # Prepare the data
        time_step = 60
        X_new = []
        for i in range(time_step, len(scaled_new_data)):
            X_new.append(scaled_new_data[i - time_step:i, 0])

        X_new = np.array(X_new)
        X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)

        # Show loading progress bar
        progress_bar.start()

        # Make predictions
        predictions = model.predict(X_new)
        predictions = scaler.inverse_transform(predictions)

        # Stop the progress bar
        progress_bar.stop()

        # Plot results in the GUI
        plot_results(new_df, new_data, predictions, time_step)

    except Exception as e:
        progress_bar.stop()
        messagebox.showerror("Error", str(e))

# Function to plot the results inside the GUI
def plot_results(new_df, new_data, predictions, time_step):
    # Clear previous plot if exists
    for widget in frame.winfo_children():
        widget.destroy()

    # Create a new figure
    plt.figure(figsize=(8, 4))
    plt.plot(new_df['Date'][time_step:], new_data[time_step:], label="Actual Price")
    plt.plot(new_df['Date'][time_step:], predictions, label="Predicted Price", color="orange")
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.grid()

    # Embed the plot in the Tkinter GUI
    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Label and Entry to upload the stock dataset
file_label = tk.Label(root, text="Upload CSV file with stock data:")
file_label.pack(pady=5)
file_entry = tk.Entry(root, width=40)
file_entry.pack(pady=5)

# Browse Button
browse_button = tk.Button(root, text="Browse", command=load_file)
browse_button.pack(pady=5)

# Create a progress bar
progress_bar = ttk.Progressbar(root, length=200, mode='indeterminate')
progress_bar.pack(pady=10)

# Predict Button
predict_button = tk.Button(root, text="Predict Price", command=predict_price)
predict_button.pack(pady=20)

# Frame for the plot
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Run the Tkinter loop
root.mainloop()
